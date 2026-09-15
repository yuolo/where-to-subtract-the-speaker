# Where to Subtract the Speaker: Normalization Placement in Speech Emotion Recognition

Code and per-run results for the paper by Vladimir Gurlev, Sheng Li,
Björn W. Schuller and Kun Qian, submitted to ICASSP 2027.

## Check the reported numbers

No data or GPU is needed:

```bash
pip install numpy pandas scipy
python paper_tables.py               # Tables 1-3 and the statistics of Sec. 2.1-3.5
python centered_confirm_analysis.py  # Sec. 3.5, closed form on the first retraining (seeds 45-47)
python centered_tuned_analysis.py    # Sec. 3.5, closed form on the second retraining (seeds 48-50)
python fixed_enroll_analysis.py      # Sec. 3.5, fixed enrollment (seeds 51-53)
```

`outputs/` holds the metrics of every reported run, one CSV row per (seed, fold).
Leakage columns are `<representation>_<probe>_mi_lb_bits` in bits, with
matching `_control_bits` (speaker-label permutation) and `_entropy_bits` (H(S)).
No audio, features or checkpoints are included.

## Data

The corpora are not redistributed. Place them at the repository root, or set
`SER_CREMA_D_DIR`, `SER_RAVDESS_DIR` and `SER_IEMOCAP_DIR`.

| Corpus | Source and license | Location |
|---|---|---|
| CREMA-D | [GitHub](https://github.com/CheyneyComputerScience/CREMA-D), ODbL | `AudioWAV/*.wav` |
| RAVDESS speech | [Zenodo](https://zenodo.org/records/1188976), CC BY-NC-SA 4.0 | `data/raw/ravdess/` (zip or extracted) |
| IEMOCAP | [USC SAIL](https://sail.usc.edu/iemocap/), license request | `iemocap/Session{1..5}/` |

## Retraining

```bash
pip install -r requirements.txt
```

Run each command for every corpus `D` in `cremad ravdess iemocap`. Each runner
accepts `--jobs N` for parallel runs on one GPU.

```bash
# Frozen wav2vec 2.0 embeddings, cached once per corpus
python wav2vec2_frozen_baseline.py --dataset D

# Trained from scratch (CRNN): Table 1, pressure sweep
python operator_runner.py --dataset D --seeds 42,43,44 [--affine]
python floor_sweep.py --dataset D --seeds 42,43,44 --pressures 0 --input-cmn --no-style-branch
python floor_sweep.py --dataset D --seeds 42,43,44 --pressures 0,0.5,1,2,4,8,16,32,64

# Frozen wav2vec 2.0: Table 1, adversary and orthogonality sweeps, post-hoc erasure
python ssl_sweep.py --dataset D --mode operator            # also --mode affine
python ssl_sweep.py --dataset D --mode cmn --no-style-branch
python ssl_sweep.py --dataset D --mode floor --pressures 0,0.5,1,2,4,8,16   # also --arm adv|orth --pressures 1,4,16
python posthoc_baselines.py --device cuda

# Fine-tuned wav2vec 2.0 (folds 1, 3, 5): Table 1, learning-rate sweep on CREMA-D
python finetune_w2v2.py --dataset D --mode operator --tag gpu2
python finetune_w2v2.py --dataset D --mode cmn --no-style-branch --tag gpu2
python finetune_w2v2.py --dataset cremad --mode plain                # also --mode factor
for lr in 0 1e-6 1e-5; do python finetune_w2v2.py --dataset cremad --mode cmn --no-style-branch --encoder-lr $lr --tag lr$lr; done

# Subtraction depth (position 0 = input-CMN, 4 = shift operator) and enrollment
python position_runner.py --dataset D --seeds 42,43,44 --fixed-position L   # L = 0..3
python position_runner.py --dataset D --seeds 42,43,44 --fixed-position 4 --enroll-grid
python position_runner.py --dataset D --seeds 45,46,47 --fixed-position 4 --enroll-grid --out-root outputs/position_centered_confirm
python position_runner.py --dataset D --seeds 48,49,50 --fixed-position 4 --enroll-grid --centered-grid --out-root outputs/position_centered_tuned
python position_runner.py --dataset D --seeds 51,52,53 --fixed-position L --fixed-enroll --out-root outputs/position_fixed_enroll   # L = 0, 4
```

The frozen-feature arms of Table 1 come from a separate run of the first two
`ssl_sweep.py` commands, stored as `ssl_local_*`. Retrained numbers agree with
`outputs/` up to run-to-run GPU variation.

The concept targets, folds and CRNN encoder are vendored from our NCMMSC 2026
code ([ser-affective-style-cbm](https://github.com/yuolo/ser-affective-style-cbm))
in `main_egemaps_baseline_deviation_cbm.py` and `wav2vec2_frozen_baseline.py`,
with comments removed and the default data paths changed.

## Citation

```bibtex
@misc{gurlev2026where,
  author = {Gurlev, Vladimir and Li, Sheng and Schuller, Bj{\"o}rn W. and Qian, Kun},
  title  = {Where to Subtract the Speaker: Normalization Placement in Speech Emotion Recognition},
  note   = {Submitted to ICASSP 2027},
  year   = {2026}
}
```

Code under the MIT License. The corpora keep their own licenses.
