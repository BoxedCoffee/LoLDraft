"""
Draft encoder model.

Architecture:
  Champion ID → 64-dim embedding (role-aware via positional encoding)
  5 champion embeds per team → concat (320) → 3-layer MLP (512, residual) → 256-dim team vector
  Cross-team Lane Matchup & Intra-Team Synergy block:
    - Explicit lane matchup diffs (blue_role - red_role) and Hadamard products (blue_role * red_role)
    - Intra-team pairwise role synergy interactions (10 pairs per side)
  Blue team vec + Red team vec (+ optional patch embed) → project + gated interaction residual → 256-dim draft embedding

Output heads (multi-task):
  1. Win prediction: BCE
  2. Gold curve regression: MSE at 5-min intervals
  3. Objective prediction: CE per objective type

Loss balancing: uncertainty-weighted (Kendall et al. 2018)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """MLP block with residual connection."""

    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.net(x))


class TeamEncoder(nn.Module):
    """Encode 5 champion embeddings into a team vector."""

    def __init__(self, input_dim: int, hidden_dim: int = 512, output_dim: int = 256,
                 num_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout) for _ in range(num_layers)
        ])
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, 5 * champion_embed_dim)"""
        h = F.gelu(self.input_proj(x))
        for block in self.blocks:
            h = block(h)
        return self.output_proj(h)


class MatchupSynergyBlock(nn.Module):
    """
    Explicitly encodes role-aligned lane matchups (blue_role vs red_role) and
    intra-team pairwise synergies across the 10 role pairs per team.
    """

    def __init__(self, champion_dim: int = 64, num_roles: int = 5, output_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.champion_dim = champion_dim
        self.num_roles = num_roles
        # Pair indices for C(num_roles, 2) = 10 intra-team pairs
        pairs = [(i, j) for i in range(num_roles) for j in range(i + 1, num_roles)]
        self.register_buffer(
            "pair_left", torch.tensor([p[0] for p in pairs], dtype=torch.long), persistent=False
        )
        self.register_buffer(
            "pair_right", torch.tensor([p[1] for p in pairs], dtype=torch.long), persistent=False
        )

        # Features:
        #   1) 5 lane diffs (blue_r - red_r): num_roles * champion_dim
        #   2) 5 lane Hadamard products (blue_r * red_r): num_roles * champion_dim
        #   3) Mean intra-team pairwise Hadamard for Blue + Red: 2 * champion_dim
        #   4) Pairwise scaled dot products (10 for Blue + 10 for Red + 5 lane dots): 2 * len(pairs) + num_roles
        num_pairs = len(pairs)
        in_features = (2 * num_roles * champion_dim) + (2 * champion_dim) + (2 * num_pairs + num_roles)

        self.proj = nn.Sequential(
            nn.Linear(in_features, output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, blue_roles: torch.Tensor, red_roles: torch.Tensor) -> torch.Tensor:
        """
        blue_roles, red_roles: (batch, 5, champion_dim)
        Returns: (batch, output_dim)
        """
        batch_size = blue_roles.shape[0]
        scale = self.champion_dim ** -0.5

        # 1. Role-aligned lane matchup interactions
        lane_diff = (blue_roles - red_roles).reshape(batch_size, -1)
        lane_prod = (blue_roles * red_roles).reshape(batch_size, -1)
        lane_dots = (blue_roles * red_roles).sum(dim=-1) * scale  # (batch, 5)

        # 2. Intra-team pairwise synergies (10 role pairs per team)
        b_left = blue_roles[:, self.pair_left, :]
        b_right = blue_roles[:, self.pair_right, :]
        r_left = red_roles[:, self.pair_left, :]
        r_right = red_roles[:, self.pair_right, :]

        blue_pair_prod = (b_left * b_right).mean(dim=1)  # (batch, champion_dim)
        red_pair_prod = (r_left * r_right).mean(dim=1)   # (batch, champion_dim)

        blue_pair_dots = (b_left * b_right).sum(dim=-1) * scale  # (batch, 10)
        red_pair_dots = (r_left * r_right).sum(dim=-1) * scale   # (batch, 10)

        features = torch.cat(
            [
                lane_diff,
                lane_prod,
                blue_pair_prod,
                red_pair_prod,
                lane_dots,
                blue_pair_dots,
                red_pair_dots,
            ],
            dim=-1,
        )
        return self.proj(features)


class DraftEncoder(nn.Module):
    """
    Full draft encoder: 10 champion IDs → 256-dim draft embedding.

    Args:
        num_champions: size of champion vocabulary (including <UNK> at 0)
        champion_dim: embedding dimension per champion (default 64)
        num_roles: number of roles (5: top, jng, mid, bot, sup)
        team_hidden: hidden dim in team MLP
        draft_dim: output draft embedding dimension
        num_patches: number of unique patches (0 to disable patch embedding)
        patch_dim: patch embedding dimension
        num_team_layers: depth of team MLP
        dropout: dropout rate
        use_matchup_interactions: whether to add explicit lane matchup & pairwise synergy block
    """

    def __init__(
        self,
        num_champions: int,
        champion_dim: int = 64,
        num_roles: int = 5,
        team_hidden: int = 512,
        draft_dim: int = 256,
        num_patches: int = 0,
        patch_dim: int = 16,
        num_team_layers: int = 3,
        dropout: float = 0.1,
        use_matchup_interactions: bool = True,
    ):
        super().__init__()
        self.draft_dim = draft_dim
        self.num_roles = num_roles
        self.use_matchup_interactions = use_matchup_interactions

        # Champion embeddings (shared across both teams)
        self.champion_embed = nn.Embedding(num_champions, champion_dim, padding_idx=0)

        # Role positional encoding (added to champion embedding)
        self.role_embed = nn.Embedding(num_roles, champion_dim)

        # Team encoders (shared weights for blue and red)
        team_input_dim = num_roles * champion_dim  # 5 * 64 = 320
        self.team_encoder = TeamEncoder(
            input_dim=team_input_dim,
            hidden_dim=team_hidden,
            output_dim=draft_dim,
            num_layers=num_team_layers,
            dropout=dropout,
        )

        # Draft projection: combine blue + red team vectors
        combine_dim = draft_dim * 2  # 512
        if num_patches > 0:
            self.patch_embed = nn.Embedding(num_patches, patch_dim)
            combine_dim += patch_dim
        else:
            self.patch_embed = None

        self.draft_proj = nn.Sequential(
            nn.Linear(combine_dim, draft_dim),
            nn.GELU(),
            nn.LayerNorm(draft_dim),
        )

        if use_matchup_interactions:
            self.matchup_synergy = MatchupSynergyBlock(
                champion_dim=champion_dim,
                num_roles=num_roles,
                output_dim=draft_dim,
                dropout=dropout,
            )
            self.interaction_gate = nn.Parameter(torch.tensor(0.5))
            self.out_norm = nn.LayerNorm(draft_dim)
        else:
            self.matchup_synergy = None
            self.interaction_gate = None
            self.out_norm = None

    def get_role_embeddings(self, champion_ids: torch.Tensor) -> torch.Tensor:
        """
        Compute role-aware embeddings for one team's 5 champions.
        Returns: (batch, 5, champion_dim)
        """
        batch_size = champion_ids.shape[0]
        champ_embeds = self.champion_embed(champion_ids)
        role_indices = torch.arange(self.num_roles, device=champion_ids.device).unsqueeze(0).expand(batch_size, -1)
        role_embeds = self.role_embed(role_indices)
        # Keep open slots (id == 0) zeroed out so partial drafts don't inject phantom role bias
        open_mask = (champion_ids != 0).unsqueeze(-1).float()
        return (champ_embeds + role_embeds) * open_mask

    def encode_team(self, champion_ids: torch.Tensor) -> torch.Tensor:
        """
        Encode one team's 5 champions.
        champion_ids: (batch, 5) — integer champion IDs ordered by role
        """
        batch_size = champion_ids.shape[0]
        combined = self.get_role_embeddings(champion_ids)
        flat = combined.view(batch_size, -1)
        return self.team_encoder(flat)

    def forward(
        self,
        blue_champs: torch.Tensor,
        red_champs: torch.Tensor,
        patch_ids: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Encode a full draft.
        blue_champs: (batch, 5) — blue team champion IDs [top, jng, mid, bot, sup]
        red_champs: (batch, 5) — red team champion IDs [top, jng, mid, bot, sup]
        patch_ids: (batch,) — optional patch indices

        Returns: (batch, draft_dim) — 256-dim draft embedding
        """
        blue_roles = self.get_role_embeddings(blue_champs)
        red_roles = self.get_role_embeddings(red_champs)

        batch_size = blue_champs.shape[0]
        blue_vec = self.team_encoder(blue_roles.view(batch_size, -1))
        red_vec = self.team_encoder(red_roles.view(batch_size, -1))

        combined = torch.cat([blue_vec, red_vec], dim=-1)  # (batch, draft_dim * 2)

        if self.patch_embed is not None and patch_ids is not None:
            patch_vec = self.patch_embed(patch_ids)  # (batch, patch_dim)
            combined = torch.cat([combined, patch_vec], dim=-1)

        base_draft = self.draft_proj(combined)  # (batch, draft_dim)

        if self.matchup_synergy is not None and self.interaction_gate is not None:
            interaction_vec = self.matchup_synergy(blue_roles, red_roles)
            return self.out_norm(base_draft + self.interaction_gate * interaction_vec)

        return base_draft


# ── Output Heads ──────────────────────────────────────────────

class WinHead(nn.Module):
    """Binary win prediction."""
    def __init__(self, draft_dim: int = 256, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(draft_dim, hidden), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hidden, 1),
        )

    def forward(self, draft_embed: torch.Tensor) -> torch.Tensor:
        return self.net(draft_embed).squeeze(-1)  # (batch,)


class GoldCurveHead(nn.Module):
    """Predict gold differential at 6 time points (5,10,15,20,25,30 min)."""
    def __init__(self, draft_dim: int = 256, hidden: int = 128, num_points: int = 6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(draft_dim, hidden), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hidden, num_points),
        )

    def forward(self, draft_embed: torch.Tensor) -> torch.Tensor:
        return self.net(draft_embed)  # (batch, 6)


class ObjectiveHead(nn.Module):
    """Predict which team gets first objective (blue/red/none = 3 classes)."""
    def __init__(self, draft_dim: int = 256, hidden: int = 128, num_objectives: int = 4):
        super().__init__()
        # Separate classifier per objective type
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(draft_dim, hidden), nn.GELU(), nn.Dropout(0.1),
                nn.Linear(hidden, 3),  # blue, red, none
            )
            for _ in range(num_objectives)
        ])

    def forward(self, draft_embed: torch.Tensor) -> list[torch.Tensor]:
        return [head(draft_embed) for head in self.heads]  # list of (batch, 3)


# ── Multi-task model with uncertainty-weighted loss ───────────

class DraftModel(nn.Module):
    """
    Complete multi-task draft model.

    Combines encoder + 3 output heads with learned loss weights
    (Kendall et al. 2018 uncertainty weighting).
    """

    def __init__(
        self,
        num_champions: int,
        champion_dim: int = 64,
        draft_dim: int = 256,
        num_patches: int = 0,
        patch_dim: int = 16,
        team_hidden: int = 512,
        num_team_layers: int = 3,
        num_gold_points: int = 6,
        num_objectives: int = 4,
        dropout: float = 0.1,
        use_matchup_interactions: bool = True,
    ):
        super().__init__()

        self.encoder = DraftEncoder(
            num_champions=num_champions,
            champion_dim=champion_dim,
            draft_dim=draft_dim,
            num_patches=num_patches,
            patch_dim=patch_dim,
            team_hidden=team_hidden,
            num_team_layers=num_team_layers,
            dropout=dropout,
            use_matchup_interactions=use_matchup_interactions,
        )

        self.win_head = WinHead(draft_dim)
        self.gold_head = GoldCurveHead(draft_dim, num_points=num_gold_points)
        self.obj_head = ObjectiveHead(draft_dim, num_objectives=num_objectives)

        # Learned log-variance for uncertainty weighting (Kendall et al. 2018)
        # Initialize to 0 → weight = exp(0) = 1 for each task
        self.log_var_win = nn.Parameter(torch.zeros(1))
        self.log_var_gold = nn.Parameter(torch.zeros(1))
        self.log_var_obj = nn.Parameter(torch.zeros(1))

    def load_state_dict(self, state_dict, strict: bool = True):
        """
        Backward-compatible checkpoint loading: if loading a legacy checkpoint created
        before MatchupSynergyBlock was added, disable the interaction gate cleanly.
        """
        has_interaction_keys = any(k.startswith("encoder.matchup_synergy") for k in state_dict)
        if not has_interaction_keys and self.encoder.matchup_synergy is not None:
            result = super().load_state_dict(state_dict, strict=False)
            with torch.no_grad():
                self.encoder.interaction_gate.zero_()
            return result
        return super().load_state_dict(state_dict, strict=strict)

    def forward(self, blue_champs, red_champs, patch_ids=None):
        draft_embed = self.encoder(blue_champs, red_champs, patch_ids)
        win_logit = self.win_head(draft_embed)
        gold_pred = self.gold_head(draft_embed)
        obj_preds = self.obj_head(draft_embed)
        return draft_embed, win_logit, gold_pred, obj_preds

    def compute_loss(
        self,
        win_logit: torch.Tensor,
        win_target: torch.Tensor,
        gold_pred: torch.Tensor,
        gold_target: torch.Tensor,
        obj_preds: list[torch.Tensor],
        obj_targets: list[torch.Tensor],
        gold_mask: torch.Tensor = None,
        obj_mask: torch.Tensor = None,
        task_weights: dict[str, float] | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Compute uncertainty-weighted multi-task loss.

        Returns dict with 'total', 'win', 'gold', 'obj', and per-task weights.
        """
        # Win loss (BCE)
        loss_win = F.binary_cross_entropy_with_logits(win_logit, win_target.float())

        # Gold curve loss (MSE, optionally masked for missing timesteps)
        if gold_mask is not None:
            gold_diff = (gold_pred - gold_target) ** 2
            gold_count = gold_mask.sum()
            loss_gold = (gold_diff * gold_mask).sum() / gold_count.clamp(min=1)
        else:
            gold_count = gold_pred.new_tensor(gold_pred.numel())
            loss_gold = F.mse_loss(gold_pred, gold_target)

        # Objective loss (CE per objective type, averaged)
        obj_losses = []
        for index, (pred, target) in enumerate(zip(obj_preds, obj_targets)):
            if obj_mask is None:
                valid = torch.ones_like(target, dtype=torch.bool)
            else:
                valid = obj_mask[:, index] > 0
            if valid.any():
                obj_losses.append(F.cross_entropy(pred[valid], target[valid]))
        loss_obj = (
            torch.stack(obj_losses).mean()
            if obj_losses
            else win_logit.new_zeros(())
        )
        objective_count = (
            obj_mask.sum() if obj_mask is not None else win_logit.new_tensor(1.0)
        )

        if task_weights is not None:
            total = (
                float(task_weights.get("win", 0.0)) * loss_win
                + float(task_weights.get("gold", 0.0)) * loss_gold
                + float(task_weights.get("obj", 0.0)) * loss_obj
            )
            return {
                "total": total,
                "win": loss_win,
                "gold": loss_gold,
                "obj": loss_obj,
                "weight_win": loss_win.new_tensor(task_weights.get("win", 0.0)),
                "weight_gold": loss_gold.new_tensor(task_weights.get("gold", 0.0)),
                "weight_obj": loss_obj.new_tensor(task_weights.get("obj", 0.0)),
            }

        # Uncertainty weighting: L_total = Σ (1/(2σ²)) * L_i + log(σ)
        # Using log_var = log(σ²), so 1/(2σ²) = 0.5 * exp(-log_var)
        w_win = 0.5 * torch.exp(-self.log_var_win)
        w_gold = 0.5 * torch.exp(-self.log_var_gold)
        w_obj = 0.5 * torch.exp(-self.log_var_obj)

        total = w_win * loss_win + 0.5 * self.log_var_win
        if gold_count > 0:
            total = total + w_gold * loss_gold + 0.5 * self.log_var_gold
        if objective_count > 0:
            total = total + w_obj * loss_obj + 0.5 * self.log_var_obj

        return {
            "total": total,
            "win": loss_win,
            "gold": loss_gold,
            "obj": loss_obj,
            "weight_win": w_win.detach(),
            "weight_gold": w_gold.detach(),
            "weight_obj": w_obj.detach(),
        }

    def get_embedding(self, blue_champs, red_champs, patch_ids=None) -> torch.Tensor:
        """Get just the draft embedding (for evaluation/inference)."""
        return self.encoder(blue_champs, red_champs, patch_ids)