# Guy × Claude — progress & todo

Project: Brain Tumor Segmentation + Tumor-Type Classification (Multi-task Learning, 4-point difficulty level).
Team plan: [TASKS.md](TASKS.md) · Results: [results/RESULTS.md](results/RESULTS.md) · How to run: [README.md](README.md)

## Status: original experiments verified — team review, slides and rehearsal remain (2026-10-10)

**Headline:** multi-task had higher classification scores in the original runs (Macro-F1 0.916 → 0.938,
higher for 3/3 matched seed labels), with similar mean Dice (0.774 vs 0.775). This is not an equivalence test.
Current fixes and checks: [codextodo.md](codextodo.md). The original results predate augmentation seeding.

## Done

- [x] **2026-10-09** Reviewed the friend's proposal against the rubric → topic accepted
- [x] **2026-10-09** Created GitHub repo `NDN-P/DLMED` (public; username was G9YzNDN until 2026-10) and the project code
  (data prep, shared-encoder model, training `--tasks cls/seg/both`, fixed/uncertainty loss weighting, evaluation, Colab notebook)
- [x] **2026-10-09** Smoke-tested the pipeline on fake data; set up the local GPU environment (RTX 5070 Laptop, PyTorch 2.11 + CUDA 12.8)
- [x] **2026-10-09** Downloaded the Figshare dataset (checksums verified) → `DLMED/data/` (git-ignored); 3,064 images, masks checked visually
- [x] **2026-10-09** Grouped suffixed glioma IDs conservatively to reduce possible leakage → 209 groups, split 151 / 29 / 29; shared identity is unverified
- [x] **2026-10-09** Learning rate chosen on validation only (1e-4 / **3e-4** / 1e-3) → `results/tuning_lr_validation.csv`
- [x] **2026-10-09** Final experiments A–D × 3 seeds (12 runs, ≈ 7 min each) + test evaluation
- [x] **2026-10-09** Analysis: summary table, per-seed comparison, per-class / per-type / per-size results,
  error analysis by patient ID (one ID accounts for most classification errors), learned weights; clinical causes are hypotheses
- [x] **2026-10-09** 10 slide-ready figures in `results/figures/` (architecture, dataset, main result, curves, confusion matrices, Dice by type/size, weights, examples)
- [x] **2026-10-09** Wrote `results/RESULTS.md` (findings, limitations, Q&A answers) and rewrote `TASKS.md` as a review + slides plan
- [x] **2026-10-10** Reviewed raw data, splits, saved predictions and all 12 checkpoints; full test inference matches the original reported outputs
- [x] **2026-10-10** Corrected grouping/hospital/statistical/clinical claims and separated original results from the new training protocol; implementation checks are recorded in `codextodo.md`
- [x] **2026-10-10** Fixed run-folder protection, augmentation/worker seeding and Windows instructions; three regression tests and a real-data two-epoch GPU/validation check passed. All 104 archived files checked by hash are unchanged.

## Todo (me)

- [ ] Invite the 5 teammates as collaborators (repo → Settings → Collaborators) — only needed if they will push
- [ ] Send the team the repo link and say: read `results/RESULTS.md`, then follow `TASKS.md`
- [ ] Send the Thai topic message; each person picks 1 of the 6 topics and writes their name in TASKS.md (my topic: ______)
- [ ] Optional: share `data/raw/zips/` on Google Drive if a teammate wants to re-run on Colab

## Todo (team — see TASKS.md)

- [ ] Everyone reads RESULTS.md and reviews their part
- [ ] Meet as a group and make the slides together (slide order in TASKS.md)
- [ ] Rehearse twice with a timer, practise the Q&A list

## Notes

- Python env: `C:/Users/guymy/.venvs/dlmed/Scripts/python`
- Data: `DLMED/data/` · trained runs + checkpoints: `DLMED/runs/` (both git-ignored, stay on this laptop)
- Rebuild results after any change: `python src/analyze.py --runs_dir runs --out_dir results`
- For Windows use the environment-specific Python commands in README; new experiments need a new output folder and a separate report.
