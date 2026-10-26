# How versions work in this repository

Each major version has its own folder (`v1/`, `v2/`, `v3/`). The newest folder is the one being developed; older folders are frozen and only get documentation fixes. Every release also gets a git tag (`v3.0.0`) and a GitHub Release, so any version can be downloaded exactly as it was.

## Version numbers

`MAJOR.MINOR.PATCH`, for example `3.1.0`:

| Change | Example | New number | New folder? |
|---|---|---|---|
| New model or a different pipeline | replace MoViNet-A0 with another backbone | 4.0.0 | yes, `v4/` |
| New feature, same model | mobile alert upload, real night clips, a new script | 3.1.0 | no, stays in `v3/` |
| Bug fix or documentation fix | fix a crash, correct a README number | 3.0.1 | no |

## Day-to-day changes

1. Create a branch: `git checkout -b feature/short-name` (or `fix/short-name`).
2. Make the change and commit with a clear message, for example `v3: add real NoIR night clips to the demo`.
3. Push and open a Pull Request into `main`. Merge it when it looks right.
4. Add a line under `## [Unreleased]` at the top of `CHANGELOG.md` describing the change.

## Making a release

1. In `CHANGELOG.md`, rename `## [Unreleased]` to `## [3.1.0] - YYYY-MM-DD`.
2. Update the version table in the main `README.md` if anything there changed.
3. Merge to `main`, then tag the merge commit:
   ```bash
   git checkout main && git pull
   git tag -a v3.1.0 -m "PiLENS v3.1.0"
   git push origin v3.1.0
   ```
4. On GitHub: **Releases → Draft a new release → choose the tag**, paste the changelog entry as the description, attach large files (for example `export_clip_b16.zip`) and click **Publish release**.

## Starting a new major version

1. Copy the current folder: `cp -r v3 v4`, then change `v4/`.
2. Add `v4` to the version table in the main `README.md` and mark the old one as replaced.
3. Add a short note at the top of `v3/README.md` pointing to `v4/`.
4. Release it as `v4.0.0`.

## Rules

- Never commit `.env`, passwords, API keys or the real datasets.
- Files over 50 MB (for example the CLIP model) go into a GitHub Release, not into git.
- Every number in a README must come from a file in that version's `results/` folder.
