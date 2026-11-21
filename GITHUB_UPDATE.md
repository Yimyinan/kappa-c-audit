# GitHub replacement instructions

Prepared payload: V49, 2026-11-21.

1. Keep the existing public repository name
   `https://github.com/Yimyinan/kappa-c-audit`.
2. Replace the repository working tree with this payload.
3. Confirm that `README.md`, `CITATION.cff`, and `docs/V49_MANUSCRIPT_METADATA.json`
   name the V49 manuscript.
4. Commit and tag `paper1-v49`.
5. Push the branch and tag.
6. Copy the resulting commit hash into the Figshare deposit description.

This payload intentionally omits `.git`, expression matrices, raw checkpoints and
large derived arrays. The small-result GitHub tree is self-contained for code
inspection and JSON-table verification.
