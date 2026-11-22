# scripts/

- **Bulk path-reference rewriter:** `scripts/rewrite_path_refs.py` rewrites stale path references after files move. Point it at a mapping source (`--git-renames BASE HEAD` or `--mapping-json FILE`) and the trees to sweep; it is a dry run unless `--apply`, `--exclude GLOB` leaves sealed files stale, and `--check` exits 1 while stale references remain. Pinned by `tests/repo/test_rewrite_path_refs.py`.
