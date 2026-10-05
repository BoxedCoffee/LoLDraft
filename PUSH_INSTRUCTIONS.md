# Publishing checklist

Run these checks from the repository root before publishing:

```powershell
git lfs install
git status
git diff --cached --stat
git diff --cached --check
git ls-files | Select-String -Pattern '(^|/)(config.yaml|.*\.parquet|.*\.pt|.*\.db)$'
```

The last command should not report local credentials or generated data. Use
`config.example.yaml` as the shareable collector configuration template.

Review the staged diff, then create a scoped commit:

```powershell
git add .
git commit -m "Prepare draft analyzer for research release"
git push origin main
```

The first push includes Git LFS objects for `5mLoLGames/` and the canonical
processed Riot Parquet files. Confirm the repository's GitHub LFS quota is
available before pushing the large dataset objects.

Do not force-push unless the repository owner explicitly decides to rewrite
history. If a credential was ever committed, rotate it with Riot immediately;
removing it from the current tree does not remove it from existing Git history.