<!--
The commit message(s) above this template ARE the summary — gt pastes them
into the body first, and they were generated under gitlint/commitizen rules.
Do not re-summarize here. This template carries only what the commits cannot:
the gate evidence a reviewer needs to trust the PR. See CLAUDE.md House rules.
-->

## Gate evidence

- [ ] Published with `just ship-it` (live MLflow gate ran) — not a bare `gt submit`
- [ ] New/changed guards show RED teeth: the guarded thing was broken and the test failed
- [ ] Measured claims in comments carry their measurement (and retract, not overwrite, wrong ones)
- [ ] `data|models|results` touched → `dvc push` ran and `*.dvc` pointers are in this diff

## Not covered by the gates

<!-- Anything a reviewer must know that no gate checks: rollout order, manual
     steps, follow-up work deferred, DECISION-NEEDED items. Delete if none. -->
