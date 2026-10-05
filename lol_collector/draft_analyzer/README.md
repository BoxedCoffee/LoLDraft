# League Draft Analyzer

This is the safe pivot application for the senior project. It is separate from
`draft_encoder` and does not modify or depend on its checkpoints. It analyzes
how drafts relate to gold trajectories, objective control, comebacks, and wins.

## Build analysis artifacts

From this directory:

```powershell
python analysis.py `
  --data-dir ..\draft_encoder\data\processed_riot `
  --output-dir analysis_output
```

The analysis deduplicates augmented side-swapped rows by `match_id` before
calculating descriptive statistics. It produces a match-level Parquet table,
champion-role statistics, patch summaries, and `summary.json`.

## Run the analyzer

```powershell
streamlit run app.py
```

The UI is intentionally descriptive: it reports historical associations for
similar drafts and labels the result as low-confidence analysis rather than
claiming causal or guaranteed predictions.

The interface is organized into four views:

- **Pathway to victory**: win rate, gold trajectory, comeback rate, and lead conversion.
- **Objective control**: first Dragon, Herald, Baron, and Tower rates with observed sample counts.
- **Draft evidence**: role-specific champion history and sample sizes.
- **Patch context**: broader meta movement across the 49 available patches.

Selections match the same team side and role in historical matches. The app
shows the evidence size explicitly because a highly specific ten-champion draft
can have very few comparable matches.

The sidebar provides three comparison breadths:

- **Role-aware (either side)** (default): selected champions must appear in the
  selected roles, but may be on either team.
- **Exact roles & sides**: the strictest comparison; useful for studying side
  advantage but often sparse for complete drafts.
- **Champion presence**: the broadest comparison; selected champions may appear
  in any role or side.

Broader modes are intentional: they provide enough observations for descriptive
analysis while the UI clearly labels that they are not exact-draft predictions.

The UI also includes an evidence ladder showing the available sample size under
each comparison mode, Wilson 95% intervals for win and conversion rates, deltas
against the full-dataset baseline, and a lightly smoothed lead-conversion
estimate. These guard against overinterpreting rare drafts or noisy percentages.

Champion names are normalized from the processed dataset vocabulary through
`draft_encoder/data/champion_data.json`; this is distinct from Riot's external
champion IDs used by `full_champion_mapping.json`.

Additional analysis controls and views include:

- **Auto comparison mode**, which progressively broadens from exact sides to
  role-aware to champion presence until the configured evidence threshold is met.
  If a complete partial draft is still too rare, it uses the largest supported
  subset of the selected picks and labels that requirement explicitly.
- **Patch scope**, so outdated metas can be separated from aggregate trends.
- **Objective value by game state**, comparing objective associations when Blue is
  ahead or behind at 10 minutes.
- **Exact lane matchup evidence** for selected Blue/Red role pairs.
- **Transparent composition identity indicators** for engage, poke, scaling, and
  split-push champion pools. These are exploratory labels, not causal claims.

## Research direction

The central question is:

> How do League draft compositions relate to early economic advantage, objective
> control, and the ability to convert or recover from leads?

Useful follow-up analyses include composition archetype clustering, patch trends,
objective conversion rates conditioned on 10-minute gold, and bootstrap
confidence intervals.
