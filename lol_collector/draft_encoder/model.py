"""
Draft encoder model.

Architecture:
  Champion ID → 64-dim embedding (role-aware via positional encoding)
  5 champion embeds per team → concat (320) → 3-layer MLP (512, residual) → 256-dim team vector
  Blue team vec + Red team vec → concat (512) → project → 256-dim draft embedding
  + Optional patch embedding (16-dim) concatenated before output heads

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
    ):
        super().__init__()
        self.draft_dim = draft_dim

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

    def encode_team(self, champion_ids: torch.Tensor) -> torch.Tensor:
        """
        Encode one team's 5 champions.
        champion_ids: (batch, 5) — integer champion IDs ordered by role
        """
        batch_size = champion_ids.shape[0]
        # (batch, 5, champion_dim)
        champ_embeds = self.champion_embed(champion_ids)

        # Add role positional encoding
        role_indices = torch.arange(5, device=champion_ids.device).unsqueeze(0).expand(batch_size, -1)
        role_embeds = self.role_embed(role_indices)
        combined = champ_embeds + role_embeds

        # Flatten: (batch, 5 * champion_dim)
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
        blue_vec = self.encode_team(blue_champs)  # (batch, draft_dim)
        red_vec = self.encode_team(red_champs)     # (batch, draft_dim)

        combined = torch.cat([blue_vec, red_vec], dim=-1)  # (batch, draft_dim * 2)

        if self.patch_embed is not None and patch_ids is not None:
            patch_vec = self.patch_embed(patch_ids)  # (batch, patch_dim)
            combined = torch.cat([combined, patch_vec], dim=-1)

        return self.draft_proj(combined)  # (batch, draft_dim)


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
        )

        self.win_head = WinHead(draft_dim)
        self.gold_head = GoldCurveHead(draft_dim, num_points=num_gold_points)
        self.obj_head = ObjectiveHead(draft_dim, num_objectives=num_objectives)

        # Learned log-variance for uncertainty weighting (Kendall et al. 2018)
        # Initialize to 0 → weight = exp(0) = 1 for each task
        self.log_var_win = nn.Parameter(torch.zeros(1))
        self.log_var_gold = nn.Parameter(torch.zeros(1))
        self.log_var_obj = nn.Parameter(torch.zeros(1))

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
            loss_gold = (gold_diff * gold_mask).sum() / gold_mask.sum().clamp(min=1)
        else:
            loss_gold = F.mse_loss(gold_pred, gold_target)

        # Objective loss (CE per objective type, averaged)
        obj_losses = []
        for pred, target in zip(obj_preds, obj_targets):
            if target.numel() > 0:
                obj_losses.append(F.cross_entropy(pred, target))
        loss_obj = torch.stack(obj_losses).mean() if obj_losses else torch.tensor(0.0, device=win_logit.device)

        # Uncertainty weighting: L_total = Σ (1/(2σ²)) * L_i + log(σ)
        # Using log_var = log(σ²), so 1/(2σ²) = 0.5 * exp(-log_var)
        w_win = 0.5 * torch.exp(-self.log_var_win)
        w_gold = 0.5 * torch.exp(-self.log_var_gold)
        w_obj = 0.5 * torch.exp(-self.log_var_obj)

        total = (
            w_win * loss_win + 0.5 * self.log_var_win
            + w_gold * loss_gold + 0.5 * self.log_var_gold
            + w_obj * loss_obj + 0.5 * self.log_var_obj
        )

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
