# Contributing to PiLENS

Thank you for helping. PiLENS is developed at The Islamia University of Bahawalpur and by PiVisen AI.

## Reporting a problem

Open an issue with the **Bug report** template. Include the version (`v3`), the command you ran, the full console output up to `END OF REPORT`, and your hardware (Pi model, OS, camera).

## Making a change

1. Fork the repository and create a branch: `feature/short-name` or `fix/short-name`.
2. Work in the current version folder (`v3/`). Older folders (`v1/`, `v2/`) only take documentation fixes.
3. Keep scripts runnable on a Raspberry Pi 5 CPU with `numpy`, `opencv-python` and `onnxruntime`.
4. If you change a number in a README, the source file must be in `v3/results/`.
5. Run the checks the CI runs:
   ```bash
   python -m py_compile v3/*.py v3/training/*.py v3/evaluation/*.py
   python v3/evaluation/reduced_classes_eval.py && git diff --exit-code v3/results/paper_final/reduced_classes.md
   ```
6. Add a line under `## [Unreleased]` in `CHANGELOG.md` and open a pull request.

## Rules

- Never commit `.env` files, passwords, API keys, datasets or videos of people.
- Files over 50 MB go into a GitHub Release, not into git.
- Report results honestly: thresholds from validation data, the test set used once, no "zero false alarms" claims from quiet runs.
