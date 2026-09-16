# Where to Subtract the Speaker: Normalization Placement in Speech Emotion Recognition

Code and per-run results for the paper by Vladimir Gurlev, Sheng Li,
Björn W. Schuller and Kun Qian, submitted to ICASSP 2027.

## Check the reported numbers

No data or GPU is needed:

```bash
python -m pip install -r requirements-analysis.txt
python paper_tables.py               # Tables 1-3 and the statistics of Sec. 2.1-3.5
python centered_confirm_analysis.py  # Sec. 3.5, closed form on the first retraining (seeds 45-47)
python centered_tuned_analysis.py    # Sec. 3.5, closed form on the second retraining (seeds 48-50)
python fixed_enroll_analysis.py      # Sec. 3.5, fixed enrollment (seeds 51-53)
```

The 387 CSV files in `outputs/` contain the released per-run metrics for all
three corpora. Each trained run is identified by `(seed, fold)` within a corpus
and model configuration. Sweep and enrollment files also index pressure,
enrollment size, shrinkage, and evaluation protocol. Fixed-enrollment files
contain two enrollment draws per run; analyses average them within each run
before paired comparisons. These draws are not independent training runs.
Leakage columns are `<representation>_<probe>_mi_lb_bits` in bits, with
matching `_control_bits` (speaker-label permutation) and `_entropy_bits` (H(S)).
No audio, features or checkpoints are included. The scripts report unsuccessful
checks too: the E5 leakage-ordering criterion is not met on RAVDESS.

The CSV analyses were checked with Python 3.9.6 and the
versions in `requirements-analysis.txt`. The complete historical training
environment was not archived in this package; `requirements.txt` lists training
dependencies, not a verified training lockfile. Training on other library
versions or hardware need not reproduce the reference numbers exactly.
The final code audit also identified two issues in secondary comparisons:
fine-tuned input-CMN bypasses SpecAugment, whereas the operator uses the standard
wav2vec forward path; and the INLP implementation uses QR without numerical
rank truncation. The supplied code and CSVs preserve those historical
experiments. Matched fine-tuning and rank-aware INLP reruns are needed before
treating these comparisons as fully controlled. The CRNN placement, depth,
and fixed-enrollment experiments do not use these two paths.

## Data

The corpora are not redistributed. Place them at the repository root, or set
`SER_CREMA_D_DIR`, `SER_RAVDESS_DIR` and `SER_IEMOCAP_DIR`. For a RAVDESS
archive outside the default directory, also set `SER_RAVDESS_ZIP` to its path.
The wav2vec extractor additionally accepts `--data-dir`. CREMA-D audio must be
downloaded through Git LFS or as actual WAV files, not LFS pointer files.

| Corpus | Source and license | Location |
|---|---|---|
| CREMA-D | [GitHub](https://github.com/CheyneyComputerScience/CREMA-D), ODbL | `AudioWAV/*.wav` |
| RAVDESS speech | [Zenodo](https://zenodo.org/records/1188976), CC BY-NC-SA 4.0 | `data/raw/ravdess/` (zip or extracted) |
| IEMOCAP | [USC SAIL](https://sail.usc.edu/iemocap/), license request | `iemocap/Session{1..5}/` |

## Retraining

Install the training dependencies in a separate environment with PyTorch for
your hardware. The model weights `facebook/wav2vec2-base` are downloaded on
first use. Run the following commands from the repository root. The loops
specify every corpus, position, seed set and pressure used in the analyses.
`--jobs 1` is a conservative default; increase it if device memory permits.
The feature extractor has its own batching options and does not accept `--jobs`.

```bash
python -m pip install -r requirements.txt

for corpus in cremad ravdess iemocap; do
    # Cache frozen wav2vec 2.0 embeddings at the default local cache path.
    python wav2vec2_frozen_baseline.py --dataset "$corpus"

    # CRNN placement and pressure. Store new metrics separately from outputs/.
    python operator_runner.py --dataset "$corpus" --seeds 42,43,44 --jobs 1 --results-root reruns
    python operator_runner.py --dataset "$corpus" --seeds 42,43,44 --affine --jobs 1 --results-root reruns
    python floor_sweep.py --dataset "$corpus" --seeds 42,43,44 --pressures 0 --input-cmn --no-style-branch --jobs 1 --results-root reruns
    python floor_sweep.py --dataset "$corpus" --seeds 42,43,44 --pressures 0,0.5,1,2,4,8,16,32,64 --jobs 1 --results-root reruns

    # Frozen Table 1 arms are the separate local repeat (Apple hardware, CPU).
    python ssl_sweep.py --dataset "$corpus" --mode operator --tag local --device cpu --jobs 1 --results-root reruns
    python ssl_sweep.py --dataset "$corpus" --mode cmn --no-style-branch --tag local --device cpu --jobs 1 --results-root reruns
    # Original frozen sweeps also enter the probe-agreement and concept analyses.
    for mode in operator affine; do
        python ssl_sweep.py --dataset "$corpus" --mode "$mode" --jobs 1 --results-root reruns
    done
    python ssl_sweep.py --dataset "$corpus" --mode floor --pressures 0,0.5,1,2,4,8,16 --jobs 1 --results-root reruns
    for arm in adv orth; do
        python ssl_sweep.py --dataset "$corpus" --mode floor --arm "$arm" --pressures 1,4,16 --jobs 1 --results-root reruns
    done

    # Fine-tuned Table 1, folds 1,3,5, seeds 42,43,44.
    python finetune_w2v2.py --dataset "$corpus" --mode operator --tag gpu2 --jobs 1 --results-root reruns
    python finetune_w2v2.py --dataset "$corpus" --mode cmn --no-style-branch --tag gpu2 --jobs 1 --results-root reruns

    # Depth sweep behind Fig. 2, and the original redrawn enrollment grid.
    for position in 0 1 2 3; do
        python position_runner.py --dataset "$corpus" --seeds 42,43,44 --fixed-position "$position" --jobs 1 --out-root reruns/position
    done
    python position_runner.py --dataset "$corpus" --seeds 42,43,44 --fixed-position 4 --enroll-grid --jobs 1 --out-root reruns/position
    python position_runner.py --dataset "$corpus" --seeds 45,46,47 --fixed-position 4 --enroll-grid --jobs 1 --out-root reruns/position_centered_confirm
    python position_runner.py --dataset "$corpus" --seeds 48,49,50 --fixed-position 4 --enroll-grid --centered-grid --jobs 1 --out-root reruns/position_centered_tuned
    for position in 0 4; do
        python position_runner.py --dataset "$corpus" --seeds 51,52,53 --fixed-position "$position" --fixed-enroll --jobs 1 --out-root reruns/position_fixed_enroll
    done
done

# Post-hoc erasure uses all three corpora (use --device cpu if needed).
python posthoc_baselines.py --device cuda --jobs 1 --results-root reruns

# CREMA-D representation audit and learning-rate sweep.
for mode in plain factor; do
    python finetune_w2v2.py --dataset cremad --mode "$mode" --jobs 1 --results-root reruns
done
for lr in 0 1e-6 1e-5; do
    python finetune_w2v2.py --dataset cremad --mode cmn --no-style-branch --encoder-lr "$lr" --tag "lr$lr" --jobs 1 --results-root reruns
done

# Audit new runs after every required experiment has completed.
python paper_tables.py --results-root reruns
python centered_confirm_analysis.py --root reruns/position_centered_confirm --reference reruns/position
python centered_tuned_analysis.py --root reruns/position_centered_tuned
python fixed_enroll_analysis.py --root reruns/position_fixed_enroll
```

`outputs/` remains the published reference. Training examples write new metrics
to `reruns/`; feature caches are generated locally and ignored by Git.
The `local` and `gpu2` tags preserve historical filenames, not guarantees of
identical results. The position runner shares its default CRNN data cache under
`outputs/floor_sweep/`, even when `--out-root` points elsewhere.

Speaker-disjoint folds are generated deterministically by `GroupKFold` on the
sorted corpus inventory; inner validation uses `GroupShuffleSplit` with
`seed + fold`. Thus splits can be regenerated from the licensed corpora without
publishing IEMOCAP utterance identifiers. The model, fold and target code in
`main_egemaps_baseline_deviation_cbm.py` and `wav2vec2_frozen_baseline.py` was
adapted from the codebase accompanying our NCMMSC 2026 paper
([ser-affective-style-cbm](https://github.com/yuolo/ser-affective-style-cbm)).
This repository includes the three-corpus implementation used here.

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
