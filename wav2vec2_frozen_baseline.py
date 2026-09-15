"""Frozen wav2vec 2.0 embedding cache and linear baseline"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
import os
from contextlib import nullcontext
from typing import Dict, List, Optional, Tuple

import librosa
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from main_egemaps_baseline_deviation_cbm import (
    Config,
    dataset_display_name,
    discover_dataset,
    emotion_names_for_dataset,
    normalize_dataset_name,
    speaker_leakage_audit,
    write_dataset_statistics,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "wav2vec2_frozen_outputs"


@dataclass
class Wav2Vec2BaselineConfig:
    dataset: str = "cremad"
    crema_d_dir: str = os.environ.get("SER_CREMA_D_DIR", str(PROJECT_ROOT / "AudioWAV"))
    ravdess_dir: str = os.environ.get("SER_RAVDESS_DIR", str(PROJECT_ROOT / "data" / "raw" / "ravdess"))
    iemocap_dir: str = os.environ.get("SER_IEMOCAP_DIR", str(PROJECT_ROOT / "iemocap"))
    output_dir: str = str(DEFAULT_OUTPUT_DIR)
    model_name: str = "facebook/wav2vec2-base"
    seed: int = 42
    n_splits: int = 5
    inner_val_size: float = 0.15
    sr: int = 16000

    embedding_batch_size: int = 8
    num_workers: int = 0
    force_recompute_embeddings: bool = False

    train_batch_size: int = 256
    epochs: int = 50
    patience: int = 8
    lr: float = 1e-3
    weight_decay: float = 1e-4

    speaker_probe_repeats: int = 5
    speaker_probe_test_size: float = 0.30

    device: str = "auto"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: str) -> torch.device:
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def configure_runtime(device: torch.device) -> None:
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass


def inference_autocast(device: torch.device) -> object:
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def auto_num_workers(requested: int, device: torch.device) -> int:
    if device.type == "mps":
        return 0
    if requested and int(requested) > 0:
        return int(requested)
    cpu = os.cpu_count() or 1
    return int(min(8, max(2, cpu // 2)))


class AudioPathDataset(Dataset):
    def __init__(self, df: pd.DataFrame, sr: int) -> None:
        self.df = df.reset_index(drop=True)
        self.sr = int(sr)

    def __len__(self) -> int:
        return int(len(self.df))

    def __getitem__(self, idx: int) -> Dict[str, object]:
        row = self.df.iloc[idx]
        wav, _ = librosa.load(str(row["path"]), sr=self.sr, mono=True)
        return {
            "idx": int(idx),
            "audio": wav.astype(np.float32),
        }


def audio_collate(batch: List[Dict[str, object]]) -> Dict[str, object]:
    return {
        "idx": np.array([int(item["idx"]) for item in batch], dtype=np.int64),
        "audio": [item["audio"] for item in batch],
    }


def require_transformers() -> Tuple[object, object]:
    try:
        from transformers import AutoFeatureExtractor, Wav2Vec2Model
    except Exception as exc:
        raise RuntimeError(
            "Missing dependency: transformers. Install it before extracting wav2vec2 embeddings, "
            "for example: python3 -m pip install transformers"
        ) from exc
    return AutoFeatureExtractor, Wav2Vec2Model


def masked_mean_pool(
    last_hidden_state: torch.Tensor,
    input_attention_mask: Optional[torch.Tensor],
    model: torch.nn.Module,
) -> torch.Tensor:
    if input_attention_mask is None:
        return last_hidden_state.mean(dim=1)

    if hasattr(model, "_get_feature_vector_attention_mask"):
        feature_mask = model._get_feature_vector_attention_mask(
            last_hidden_state.shape[1],
            input_attention_mask,
        )
    else:
        feature_mask = torch.ones(
            last_hidden_state.shape[:2],
            dtype=torch.bool,
            device=last_hidden_state.device,
        )

    mask = feature_mask.to(dtype=last_hidden_state.dtype, device=last_hidden_state.device).unsqueeze(-1)
    summed = (last_hidden_state * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp_min(1.0)
    return summed / denom


def extract_wav2vec2_embeddings(df: pd.DataFrame, cfg: Wav2Vec2BaselineConfig) -> np.ndarray:
    AutoFeatureExtractor, Wav2Vec2Model = require_transformers()
    device = resolve_device(cfg.device)
    configure_runtime(device)

    feature_extractor = AutoFeatureExtractor.from_pretrained(cfg.model_name)
    model = Wav2Vec2Model.from_pretrained(cfg.model_name)
    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    loader = DataLoader(
        AudioPathDataset(df, cfg.sr),
        batch_size=cfg.embedding_batch_size,
        shuffle=False,
        num_workers=auto_num_workers(cfg.num_workers, device),
        pin_memory=device.type == "cuda",
        collate_fn=audio_collate,
    )

    embeddings: Optional[np.ndarray] = None
    for batch in tqdm(loader, desc="Extracting frozen wav2vec2 embeddings"):
        inputs = feature_extractor(
            batch["audio"],
            sampling_rate=cfg.sr,
            padding=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.no_grad(), inference_autocast(device):
            out = model(**inputs)
            pooled = masked_mean_pool(
                out.last_hidden_state,
                inputs.get("attention_mask"),
                model,
            )

        pooled_np = pooled.detach().float().cpu().numpy().astype(np.float32)
        if embeddings is None:
            embeddings = np.zeros((len(df), pooled_np.shape[1]), dtype=np.float32)
        embeddings[np.asarray(batch["idx"], dtype=np.int64)] = pooled_np

    if embeddings is None:
        raise RuntimeError("No embeddings were extracted; dataset is empty.")
    return embeddings


def load_or_build_embeddings(
    df: pd.DataFrame,
    cfg: Wav2Vec2BaselineConfig,
) -> Tuple[np.ndarray, Path, Path]:
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    emb_path = out_dir / "wav2vec2_embeddings.npy"
    meta_path = out_dir / "wav2vec2_metadata.csv"

    if emb_path.exists() and meta_path.exists() and not cfg.force_recompute_embeddings:
        embeddings = np.load(emb_path)
        meta = pd.read_csv(meta_path)
        same_len = len(meta) == len(df) and embeddings.shape[0] == len(df)
        same_files = (
            same_len
            and "path" in meta.columns
            and meta["path"].astype(str).tolist() == df["path"].astype(str).tolist()
        )
        same_model = (
            same_len
            and "model_name" in meta.columns
            and str(meta["model_name"].iloc[0]) == cfg.model_name
        )
        if same_len and same_files and same_model:
            return embeddings.astype(np.float32), emb_path, meta_path
        print("Cached wav2vec2 embeddings do not match current dataset; recomputing.")

    embeddings = extract_wav2vec2_embeddings(df, cfg)
    np.save(emb_path, embeddings)

    meta_cols = [c for c in ["filename", "path", "speaker", "emotion_code", "emotion", "intensity"] if c in df.columns]
    meta = df[meta_cols].copy()
    meta.insert(0, "sample_id", np.arange(len(meta), dtype=np.int64))
    meta["model_name"] = cfg.model_name
    meta["embedding_dim"] = int(embeddings.shape[1])
    meta.to_csv(meta_path, index=False)
    return embeddings, emb_path, meta_path


class EmbeddingDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return int(len(self.y))

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


class LinearEmotionHead(nn.Module):
    def __init__(self, in_dim: int, n_classes: int) -> None:
        super().__init__()
        self.classifier = nn.Linear(in_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


def metric_row(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "test_acc": float(accuracy_score(y_true, y_pred)),
        "test_uar": float(balanced_accuracy_score(y_true, y_pred)),
        "test_macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def predict_linear_head(
    model: nn.Module,
    X: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits_all: List[np.ndarray] = []
    loader = DataLoader(
        torch.tensor(X, dtype=torch.float32),
        batch_size=batch_size,
        shuffle=False,
    )
    with torch.no_grad():
        for xb in loader:
            logits = model(xb.to(device)).detach().cpu().numpy()
            logits_all.append(logits)
    logits_np = np.concatenate(logits_all, axis=0)
    pred = logits_np.argmax(axis=1).astype(np.int64)
    return pred, logits_np


def train_linear_head_for_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    cfg: Wav2Vec2BaselineConfig,
    fold: int,
    n_classes: int,
) -> nn.Module:
    device = resolve_device(cfg.device)
    set_seed(cfg.seed + fold)

    model = LinearEmotionHead(X_train.shape[1], n_classes).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()
    train_loader = DataLoader(
        EmbeddingDataset(X_train, y_train),
        batch_size=cfg.train_batch_size,
        shuffle=True,
    )

    best_score = -math.inf
    best_state: Optional[Dict[str, torch.Tensor]] = None
    bad_epochs = 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        val_pred, _ = predict_linear_head(model, X_val, cfg.train_batch_size, device)
        val_uar = float(balanced_accuracy_score(y_val, val_pred))
        val_f1 = float(f1_score(y_val, val_pred, average="macro", zero_division=0))
        score = val_uar + 0.5 * val_f1

        if score > best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1

        if bad_epochs >= cfg.patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def mean_std(values: pd.Series) -> Tuple[float, float]:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return float("nan"), float("nan")
    return float(numeric.mean()), float(numeric.std(ddof=1)) if len(numeric) > 1 else 0.0


def format_mean_std(mean: float, std: float) -> str:
    if not np.isfinite(mean):
        return "NA"
    if not np.isfinite(std):
        return f"{mean:.4f}"
    return f"{mean:.4f} +/- {std:.4f}"


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    headers = [str(c) for c in df.columns]
    body = [[str(row[c]) for c in df.columns] for _, row in df.iterrows()]
    widths = [max(len(headers[i]), *(len(row[i]) for row in body)) for i in range(len(headers))]

    def fmt(values: List[str]) -> str:
        return "| " + " | ".join(values[i].ljust(widths[i]) for i in range(len(values))) + " |"

    sep = "| " + " | ".join("-" * w for w in widths) + " |"
    return "\n".join([fmt(headers), sep] + [fmt(row) for row in body]) + "\n"


def write_summary(out_dir: Path, metrics: pd.DataFrame, probes: pd.DataFrame) -> None:
    rows = []
    for label, col in [
        ("UAR", "test_uar"),
        ("Macro-F1", "test_macro_f1"),
        ("Accuracy", "test_acc"),
    ]:
        mean, std = mean_std(metrics[col])
        rows.append({"metric": label, "value": format_mean_std(mean, std), "mean": mean, "std": std})

    for label, col in [
        ("Speaker probe accuracy", "speaker_probe_acc"),
        ("Speaker probe uniform chance", "speaker_probe_chance_uniform"),
        ("Speaker probe majority chance", "speaker_probe_chance_majority"),
        ("Speaker probe leakage index", "speaker_probe_leakage_index"),
    ]:
        mean, std = mean_std(probes[col])
        rows.append({"metric": label, "value": format_mean_std(mean, std), "mean": mean, "std": std})

    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "wav2vec2_summary.csv", index=False)
    with (out_dir / "wav2vec2_summary.md").open("w", encoding="utf-8") as f:
        f.write("# Frozen wav2vec2-base Baseline Summary\n\n")
        f.write(dataframe_to_markdown(summary[["metric", "value"]]))


def run_wav2vec2_baseline(cfg: Wav2Vec2BaselineConfig) -> None:
    cfg.dataset = normalize_dataset_name(cfg.dataset)
    emotion_names = emotion_names_for_dataset(cfg.dataset)
    n_emotions = len(emotion_names)

    set_seed(cfg.seed)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with (out_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2, ensure_ascii=False)

    stats_cfg = Config()
    stats_cfg.DATASET = cfg.dataset
    stats_cfg.CREMA_D_DIR = cfg.crema_d_dir
    stats_cfg.RAVDESS_DIR = cfg.ravdess_dir
    stats_cfg.IEMOCAP_DIR = cfg.iemocap_dir
    stats_cfg.OUT_DIR = str(out_dir)
    stats_cfg.N_SPLITS = cfg.n_splits
    df = discover_dataset(stats_cfg)
    write_dataset_statistics(df, stats_cfg)
    print(f"Dataset: {dataset_display_name(cfg.dataset)} | classes={n_emotions}")

    X, emb_path, meta_path = load_or_build_embeddings(df, cfg)
    if X.shape[1] != 768:
        print(f"Warning: expected 768-d wav2vec2 embeddings, got {X.shape[1]}.")
    print(f"Embeddings: {X.shape} saved at {emb_path}")
    print(f"Metadata: {meta_path}")

    y = df["emotion"].to_numpy(dtype=np.int64)
    groups = df["speaker"].astype(str).to_numpy(dtype=str)
    idx_all = np.arange(len(df), dtype=np.int64)
    outer = GroupKFold(n_splits=cfg.n_splits)
    device = resolve_device(cfg.device)

    metric_rows: List[Dict[str, object]] = []
    probe_rows: List[Dict[str, object]] = []
    pred_frames: List[pd.DataFrame] = []

    for fold, (trainval_idx, test_idx) in enumerate(outer.split(idx_all, y, groups), start=1):
        trainval_groups = groups[trainval_idx]
        gss = GroupShuffleSplit(
            n_splits=1,
            test_size=cfg.inner_val_size,
            random_state=cfg.seed + fold,
        )
        inner_train_rel, val_rel = next(gss.split(trainval_idx, y[trainval_idx], trainval_groups))
        train_idx = trainval_idx[inner_train_rel]
        val_idx = trainval_idx[val_rel]

        print(
            f"Fold {fold}/{cfg.n_splits}: "
            f"train={len(train_idx)} val={len(val_idx)} test={len(test_idx)} "
            f"test_speakers={len(set(groups[test_idx]))}"
        )

        model = train_linear_head_for_fold(
            X[train_idx],
            y[train_idx],
            X[val_idx],
            y[val_idx],
            cfg,
            fold,
            n_emotions,
        )
        test_pred, test_logits = predict_linear_head(model, X[test_idx], cfg.train_batch_size, device)
        metrics = metric_row(y[test_idx], test_pred)

        train_speakers = set(groups[train_idx].tolist())
        val_speakers = set(groups[val_idx].tolist())
        test_speakers = set(groups[test_idx].tolist())
        metric_rows.append({
            "fold": fold,
            **metrics,
            "n_train": int(len(train_idx)),
            "n_val": int(len(val_idx)),
            "n_test": int(len(test_idx)),
            "n_train_speakers": int(len(train_speakers)),
            "n_val_speakers": int(len(val_speakers)),
            "n_test_speakers": int(len(test_speakers)),
            "n_train_val_speaker_overlap": int(len(train_speakers & val_speakers)),
            "n_train_test_speaker_overlap": int(len(train_speakers & test_speakers)),
            "n_val_test_speaker_overlap": int(len(val_speakers & test_speakers)),
        })

        probe = speaker_leakage_audit(
            X[test_idx],
            groups[test_idx],
            seed=cfg.seed + fold,
            test_size=cfg.speaker_probe_test_size,
            n_repeats=cfg.speaker_probe_repeats,
        )
        probe_rows.append({
            "fold": fold,
            "speaker_probe_acc": probe["probe_acc_mean"],
            "speaker_probe_acc_std": probe["probe_acc_std"],
            "speaker_probe_chance_uniform": probe["probe_chance_uniform"],
            "speaker_probe_chance_majority": probe["probe_chance_majority"],
            "speaker_probe_leakage_index": probe["probe_leakage_index"],
            "speaker_probe_n_speakers": probe["probe_n_speakers"],
            "speaker_probe_n_samples": probe["probe_n_samples"],
        })

        fold_pred = df.iloc[test_idx].copy()
        fold_pred.insert(0, "fold", fold)
        fold_pred["y_true"] = y[test_idx]
        fold_pred["y_pred"] = test_pred
        for j, name in enumerate(emotion_names):
            fold_pred[f"logit_{name}"] = test_logits[:, j]
        pred_frames.append(fold_pred)

        print(
            f"  test_UAR={metrics['test_uar']:.4f} "
            f"test_F1={metrics['test_macro_f1']:.4f} "
            f"speaker_probe={probe['probe_acc_mean']:.4f}"
        )

    probes_df = pd.DataFrame(probe_rows)
    metrics_df = pd.DataFrame(metric_rows).merge(probes_df, on="fold", how="left")
    preds_df = pd.concat(pred_frames, axis=0).reset_index(drop=True)

    metrics_df.to_csv(out_dir / "fold_metrics.csv", index=False)
    probes_df.to_csv(out_dir / "speaker_probe_by_fold.csv", index=False)
    preds_df.to_csv(out_dir / "test_predictions.csv", index=False)
    write_summary(out_dir, metrics_df, probes_df)

    print(f"Wrote {out_dir / 'fold_metrics.csv'}")
    print(f"Wrote {out_dir / 'speaker_probe_by_fold.csv'}")
    print(f"Wrote {out_dir / 'wav2vec2_summary.md'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Frozen wav2vec2-base + linear SER baseline.")
    parser.add_argument("--dataset", default="cremad", choices=["cremad", "ravdess", "iemocap"])
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Dataset directory, overriding SER_*_DIR and the corpus-specific directory option.",
    )
    parser.add_argument("--crema-d-dir", default=os.environ.get("SER_CREMA_D_DIR", str(PROJECT_ROOT / "AudioWAV")))
    parser.add_argument("--ravdess-dir", default=os.environ.get("SER_RAVDESS_DIR", str(PROJECT_ROOT / "data" / "raw" / "ravdess")))
    parser.add_argument("--iemocap-dir", default=os.environ.get("SER_IEMOCAP_DIR", str(PROJECT_ROOT / "iemocap")))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--model-name", default="facebook/wav2vec2-base")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--embedding-batch-size", type=int, default=8)
    parser.add_argument("--train-batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--speaker-probe-repeats", type=int, default=5)
    parser.add_argument("--force-recompute-embeddings", action="store_true")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Smoke-test settings: 2 folds, 3 epochs, smaller batches.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_name = normalize_dataset_name(args.dataset)
    crema_d_dir = args.crema_d_dir
    ravdess_dir = args.ravdess_dir
    iemocap_dir = args.iemocap_dir
    if args.data_dir:
        if dataset_name == "cremad":
            crema_d_dir = args.data_dir
        elif dataset_name == "ravdess":
            ravdess_dir = args.data_dir
        elif dataset_name == "iemocap":
            iemocap_dir = args.data_dir

    if args.output_dir == str(DEFAULT_OUTPUT_DIR) and dataset_name != "cremad":
        output_dir = str(PROJECT_ROOT / f"{dataset_name}_wav2vec2_frozen_outputs")
    else:
        output_dir = args.output_dir

    cfg = Wav2Vec2BaselineConfig(
        dataset=dataset_name,
        crema_d_dir=crema_d_dir,
        ravdess_dir=ravdess_dir,
        iemocap_dir=iemocap_dir,
        output_dir=output_dir,
        model_name=args.model_name,
        seed=args.seed,
        n_splits=args.n_splits,
        embedding_batch_size=args.embedding_batch_size,
        train_batch_size=args.train_batch_size,
        epochs=args.epochs,
        patience=args.patience,
        lr=args.lr,
        weight_decay=args.weight_decay,
        speaker_probe_repeats=args.speaker_probe_repeats,
        force_recompute_embeddings=args.force_recompute_embeddings,
        device=args.device,
    )
    if args.quick:
        cfg.n_splits = 2
        cfg.epochs = 3
        cfg.patience = 2
        cfg.speaker_probe_repeats = 2
        cfg.embedding_batch_size = min(cfg.embedding_batch_size, 4)
    run_wav2vec2_baseline(cfg)


if __name__ == "__main__":
    main()
