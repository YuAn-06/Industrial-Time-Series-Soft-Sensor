# Claude Code Instructions

Use the repository-local Codex skills as the operating guide for InduTS-SS benchmark work.

## Entry Point

Before handling benchmark tasks, read:

```text
.codex/skills/induts-benchmark-workflow/SKILL.md
```

Use that global workflow to classify the request and route to the smallest relevant skill.

## Skill Routing

| Task | Read |
| --- | --- |
| Environment setup, PyTorch/CUDA, conda, dependency troubleshooting | `.codex/skills/induts-create-env/SKILL.md` |
| Dataset stationarity, autocorrelation, missing values, outliers, split drift, correlations, visual reports | `.codex/skills/induts-characteristics/SKILL.md` |
| New dataset inspection, YAML scaffolding, dataset smoke checks | `.codex/skills/induts-add-dataset/SKILL.md` |
| New or edited model/YAML smoke tests | `.codex/skills/induts-smoke/SKILL.md` |
| Best-result analysis across models, grouped Excel reports, checkpoint/prediction artifact collection | `.codex/skills/induts-results-best-export/SKILL.md` |

## Repository Rules

- Work from the repository root.
- Prefer existing scripts and helpers from `.codex/skills/` instead of rewriting one-off logic.
- Keep benchmark fairness visible: do not silently change split rules, scaler behavior, seeds, task definitions, metrics, or training budgets.
- Keep Windows users in mind. Prefer Python/YAML commands unless the task explicitly uses Bash scripts.
- Do not commit generated experiment outputs, reports, or checkpoints unless the user explicitly asks.
