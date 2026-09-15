"""Data loading, eGeMAPS concept targets, folds and CRNN encoder from the NCMMSC 2026 code"""

import os
import tempfile
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("NUMBA_CACHE_DIR", os.path.join(tempfile.gettempdir(), "numba_cache"))
os.makedirs(os.environ["NUMBA_CACHE_DIR"], exist_ok=True)

import glob
import math
import random
import re
import zipfile
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
from tqdm import tqdm

import librosa

try:
    import opensmile
except Exception:
    opensmile = None

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import GroupKFold, GroupShuffleSplit, StratifiedShuffleSplit
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
)
from sklearn.linear_model import LogisticRegression


_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))


@dataclass
class Config:
    DATASET: str = "cremad"
    CREMA_D_DIR: str = os.environ.get("SER_CREMA_D_DIR", os.path.join(_REPO_ROOT, "AudioWAV"))
    RAVDESS_DIR: str = os.environ.get("SER_RAVDESS_DIR", os.path.join(_REPO_ROOT, "data", "raw", "ravdess"))
    RAVDESS_ZIP: str = os.environ.get("SER_RAVDESS_ZIP", os.path.join(_REPO_ROOT, "data", "raw", "ravdess", "Audio_Speech_Actors_01-24.zip"))
    IEMOCAP_DIR: str = os.environ.get("SER_IEMOCAP_DIR", os.path.join(_REPO_ROOT, "iemocap"))
    OUT_DIR: str = os.path.join(_REPO_ROOT, "outputs", "paper1")

    FEATURE_BACKEND: str = "opensmile"
    OPENSMILE_FEATURE_SET: str = "eGeMAPSv02"
    CACHE_CONCEPT_FEATURES_TO_CSV: bool = True
    FORCE_REBUILD_CONCEPT_CACHE: bool = False

    DIAGNOSTIC_LOCAL_BASELINES_FOR_VAL_TEST: bool = True

    SELECTION_CONCEPT_PENALTY: float = 0.0

    SR: int = 16000
    MAX_SECONDS: float = 3.5
    N_FFT: int = 400
    HOP_LENGTH: int = 160
    WIN_LENGTH: int = 400
    N_MELS: int = 64
    FMIN: int = 50
    FMAX: int = 7600

    YIN_FRAME_LENGTH: int = 1024
    YIN_FMIN: int = 50
    YIN_FMAX: int = 500

    N_SPLITS: int = 5
    INNER_VAL_SIZE: float = 0.15
    SEED: int = 42
    BATCH_SIZE: int = 32
    NUM_EPOCHS: int = 40
    LR: float = 1e-3
    WEIGHT_DECAY: float = 1e-4
    NUM_WORKERS: int = -1
    PATIENCE: int = 10

    FEATURE_EXTRACTION_JOBS: int = -1
    USE_AMP: bool = True
    AMP_DTYPE: str = "bf16"
    NUM_THREADS: int = -1

    H_DIM: int = 192
    N_AFF_CONCEPTS: int = 6
    N_STYLE_CONCEPTS: int = 6
    DROPOUT: float = 0.25

    USE_CONCEPT_EMBEDDINGS: bool = False
    CONCEPT_EMB_DIM: int = 16

    LAMBDA_AFF_CONCEPT: float = 1.50
    LAMBDA_STYLE_CONCEPT: float = 0.50
    LAMBDA_STYLE_SPEAKER: float = 0.50
    LAMBDA_AFF_SPK_ADV: float = 0.35
    LAMBDA_STYLE_EMO_ADV: float = 0.10
    LAMBDA_ORTH: float = 0.05

    GRL_MAX_LAMBDA: float = 1.0
    ADV_WARMUP_EPOCHS: int = 0

    EMOTION_HEAD_INPUT: str = "aff"
    USE_AFF_CONCEPT_BRANCH: bool = True
    USE_AFF_CONCEPT_SUPERVISION: bool = True
    USE_AFF_SPEAKER_ADVERSARY: bool = True
    USE_STYLE_EMOTION_ADVERSARY: bool = True
    USE_ORTHOGONALITY: bool = True
    USE_STYLE_BRANCH: bool = True

    SPEAKER_PROBE_REPEATS: int = 5
    SPEAKER_PROBE_TEST_SIZE: float = 0.30

    DEVICE: str = "auto"


CFG = Config()


EMOTION_MAP = {
    "ANG": 0,
    "DIS": 1,
    "FEA": 2,
    "HAP": 3,
    "NEU": 4,
    "SAD": 5,
}

EMOTION_NAMES = ["angry", "disgust", "fear", "happy", "neutral", "sad"]

RAVDESS_EMOTION_MAP = {
    "01": 0,
    "02": 1,
    "03": 2,
    "04": 3,
    "05": 4,
    "06": 5,
    "07": 6,
    "08": 7,
}

RAVDESS_EMOTION_NAMES = [
    "neutral",
    "calm",
    "happy",
    "sad",
    "angry",
    "fearful",
    "disgust",
    "surprised",
]

IEMOCAP_EMOTION_MAP = {
    "ang": 0,
    "hap": 1,
    "exc": 1,
    "neu": 2,
    "sad": 3,
}

IEMOCAP_EMOTION_NAMES = ["angry", "happy", "neutral", "sad"]

DATASET_DISPLAY_NAMES = {
    "cremad": "CREMA-D",
    "ravdess": "RAVDESS",
    "iemocap": "IEMOCAP",
}

DATASET_ALIASES = {
    "cremad": "cremad",
    "crema-d": "cremad",
    "crema_d": "cremad",
    "crema": "cremad",
    "ravdess": "ravdess",
    "ravdess-speech": "ravdess",
    "ravdess_speech": "ravdess",
    "iemocap": "iemocap",
    "iemo-cap": "iemocap",
    "iemo_cap": "iemocap",
}

AFF_CONCEPT_NAMES = [
    "vocal_arousal",
    "pitch_instability",
    "energy_variability",
    "pause_hesitation",
    "voice_tension",
    "rhythm_irregularity",
]

STYLE_CONCEPT_NAMES = [
    "baseline_pitch_level",
    "habitual_loudness_level",
    "timbre_brightness",
    "spectral_breadth",
    "articulation_sharpness",
    "tempo_tendency",
]

LIBROSA_PRIMITIVE_NAMES = [
    "f0_mean",
    "f0_std",
    "f0_range",
    "rms_mean",
    "rms_std",
    "pause_ratio",
    "centroid_mean",
    "bandwidth_mean",
    "flatness_mean",
    "zcr_mean",
    "onset_rate",
    "onset_interval_cv",
]


def is_mps_available() -> bool:
    try:
        mps_backend = getattr(torch.backends, "mps", None)
        is_available_fn = getattr(mps_backend, "is_available", None)
        return bool(callable(is_available_fn) and is_available_fn())
    except Exception:
        return False


def empty_mps_cache_safely() -> None:
    try:
        mps_mod = getattr(torch, "mps", None)
        empty_cache_fn = getattr(mps_mod, "empty_cache", None)
        if callable(empty_cache_fn):
            empty_cache_fn()
    except Exception:
        pass


def seed_mps_safely(seed: int) -> None:
    try:
        mps_mod = getattr(torch, "mps", None)
        manual_seed_fn = getattr(mps_mod, "manual_seed", None)
        if callable(manual_seed_fn):
            manual_seed_fn(seed)
    except Exception:
        pass


def select_device(device_choice: str = "auto") -> torch.device:
    choice = str(device_choice).lower().strip()
    if choice not in {"", "auto"}:
        return torch.device(choice)
    if is_mps_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def configure_torch_runtime(device: torch.device) -> None:
    if device.type == "cuda":
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
    elif device.type == "mps":
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass


MAX_FEATURE_JOBS = 32


def _cgroup_cpu_limit() -> Optional[float]:
    try:
        with open("/sys/fs/cgroup/cpu.max") as fh:
            parts = fh.read().split()
        if parts and parts[0] != "max":
            quota = float(parts[0])
            period = float(parts[1]) if len(parts) > 1 else 100000.0
            if quota > 0 and period > 0:
                return quota / period
    except (OSError, ValueError):
        pass
    try:
        with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us") as fh:
            quota = float(fh.read().strip())
        with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us") as fh:
            period = float(fh.read().strip())
        if quota > 0 and period > 0:
            return quota / period
    except (OSError, ValueError):
        pass
    return None


def available_cpu_count() -> int:
    candidates: List[int] = []
    try:
        candidates.append(len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        pass
    cg = _cgroup_cpu_limit()
    if cg is not None and cg >= 1:
        candidates.append(int(cg))
    candidates.append(os.cpu_count() or 1)
    return max(1, min(candidates))


def resolve_num_threads(num_threads: int) -> int:
    if num_threads is not None and int(num_threads) > 0:
        return int(num_threads)
    return max(1, available_cpu_count())


def resolve_n_jobs(requested: int, n_items: Optional[int] = None) -> int:
    cpu = available_cpu_count()
    if requested is None or int(requested) <= 0:
        jobs = max(1, cpu - 1) if cpu > 2 else 1
        jobs = min(jobs, MAX_FEATURE_JOBS)
    else:
        jobs = min(int(requested), cpu)
    if n_items is not None:
        jobs = max(1, min(jobs, int(n_items)))
    return max(1, jobs)


def resolve_num_workers(cfg: "Config", device: torch.device) -> int:
    requested = getattr(cfg, "NUM_WORKERS", -1)
    if device.type == "mps":
        return 0
    if requested is not None and int(requested) >= 0:
        return int(requested)
    cpu = available_cpu_count()
    return int(min(8, max(2, cpu // 2)))


def clear_device_cache(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        empty_mps_cache_safely()


def _amp_dtype(cfg: "Config") -> "torch.dtype":
    return torch.float16 if str(getattr(cfg, "AMP_DTYPE", "bf16")).lower() in {"fp16", "float16", "half"} else torch.bfloat16


def amp_autocast(cfg: "Config", device) -> Any:
    dev = device if isinstance(device, torch.device) else torch.device(str(device))
    if not getattr(cfg, "USE_AMP", False) or dev.type != "cuda":
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=_amp_dtype(cfg))


def make_grad_scaler(cfg: "Config", device):
    dev = device if isinstance(device, torch.device) else torch.device(str(device))
    enabled = (
        getattr(cfg, "USE_AMP", False)
        and dev.type == "cuda"
        and _amp_dtype(cfg) == torch.float16
    )
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    seed_mps_safely(seed)


def normalize_dataset_name(dataset: str) -> str:
    key = str(dataset).strip().lower()
    if key not in DATASET_ALIASES:
        known = ", ".join(sorted(set(DATASET_ALIASES.values())))
        raise ValueError(f"Unknown dataset {dataset!r}. Expected one of: {known}")
    return DATASET_ALIASES[key]


def dataset_display_name(dataset: str) -> str:
    return DATASET_DISPLAY_NAMES[normalize_dataset_name(dataset)]


def emotion_map_for_dataset(dataset: str) -> Dict[str, int]:
    dataset_name = normalize_dataset_name(dataset)
    if dataset_name == "cremad":
        return EMOTION_MAP
    if dataset_name == "ravdess":
        return RAVDESS_EMOTION_MAP
    if dataset_name == "iemocap":
        return IEMOCAP_EMOTION_MAP
    raise ValueError(f"Unsupported dataset: {dataset}")


def emotion_names_for_dataset(dataset: str) -> List[str]:
    dataset_name = normalize_dataset_name(dataset)
    if dataset_name == "cremad":
        return list(EMOTION_NAMES)
    if dataset_name == "ravdess":
        return list(RAVDESS_EMOTION_NAMES)
    if dataset_name == "iemocap":
        return list(IEMOCAP_EMOTION_NAMES)
    raise ValueError(f"Unsupported dataset: {dataset}")


def dataset_dir_for_config(cfg: Config) -> str:
    dataset_name = normalize_dataset_name(cfg.DATASET)
    if dataset_name == "cremad":
        return str(cfg.CREMA_D_DIR)
    if dataset_name == "ravdess":
        return str(cfg.RAVDESS_DIR)
    if dataset_name == "iemocap":
        return str(cfg.IEMOCAP_DIR)
    raise ValueError(f"Unsupported dataset: {cfg.DATASET}")


def parse_cremad_filename(path: str) -> Optional[Dict[str, Any]]:
    base = os.path.basename(path)
    stem = os.path.splitext(base)[0]
    parts = stem.split("_")
    if len(parts) < 4:
        return None

    speaker = parts[0]
    emotion_code = parts[2]
    intensity = parts[3]

    if emotion_code not in EMOTION_MAP:
        return None

    return {
        "path": path,
        "speaker": speaker,
        "emotion_code": emotion_code,
        "emotion": EMOTION_MAP[emotion_code],
        "intensity": intensity,
        "filename": base,
        "dataset": "cremad",
    }


def discover_cremad(cremad_dir: str) -> pd.DataFrame:
    wavs = sorted(glob.glob(os.path.join(cremad_dir, "*.wav")))
    rows = []
    for p in wavs:
        item = parse_cremad_filename(p)
        if item is not None:
            rows.append(item)
    df = pd.DataFrame(rows)
    if len(df) == 0:
        raise FileNotFoundError(
            f"No valid CREMA-D wav files found in: {cremad_dir}\n"
            "Expected filenames like 1001_DFA_ANG_XX.wav"
        )
    return df.reset_index(drop=True)


def maybe_extract_ravdess(ravdess_dir: str, ravdess_zip: Optional[str]) -> None:
    wavs = glob.glob(os.path.join(ravdess_dir, "**", "*.wav"), recursive=True)
    if wavs:
        return
    if ravdess_zip is None or not os.path.exists(ravdess_zip):
        return
    os.makedirs(ravdess_dir, exist_ok=True)
    print(f"No RAVDESS wav files found under {ravdess_dir}; extracting {ravdess_zip}")
    with zipfile.ZipFile(ravdess_zip, "r") as zf:
        zf.extractall(ravdess_dir)


def parse_ravdess_filename(path: str) -> Optional[Dict[str, Any]]:
    base = os.path.basename(path)
    stem = os.path.splitext(base)[0]
    parts = stem.split("-")
    if len(parts) != 7:
        return None

    modality, vocal_channel, emotion_code, intensity, statement, repetition, actor = parts
    if emotion_code not in RAVDESS_EMOTION_MAP:
        return None

    return {
        "path": path,
        "speaker": f"Actor_{actor}",
        "actor": actor,
        "emotion_code": emotion_code,
        "emotion": RAVDESS_EMOTION_MAP[emotion_code],
        "intensity": intensity,
        "statement": statement,
        "repetition": repetition,
        "modality": modality,
        "vocal_channel": vocal_channel,
        "filename": base,
        "dataset": "ravdess",
    }


def discover_ravdess(ravdess_dir: str, ravdess_zip: Optional[str] = None) -> pd.DataFrame:
    maybe_extract_ravdess(ravdess_dir, ravdess_zip)
    wavs = sorted(glob.glob(os.path.join(ravdess_dir, "**", "*.wav"), recursive=True))
    rows = []
    for p in wavs:
        item = parse_ravdess_filename(p)
        if item is not None:
            rows.append(item)
    df = pd.DataFrame(rows)
    if len(df) == 0:
        raise FileNotFoundError(
            f"No valid RAVDESS wav files found in: {ravdess_dir}\n"
            "Expected filenames like 03-01-05-01-02-01-16.wav"
        )
    return df.reset_index(drop=True)


IEMOCAP_LABEL_LINE_RE = re.compile(
    r"^\[\d+\.\d+\s*-\s*\d+\.\d+\]\s+(\S+)\s+(\w+)\s+\["
)


def parse_iemocap_eval_file(eval_path: str, session_dir: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(eval_path, encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            match = IEMOCAP_LABEL_LINE_RE.match(line)
            if match is None:
                continue
            turn_id, emotion_code = match.group(1), match.group(2)
            if emotion_code not in IEMOCAP_EMOTION_MAP:
                continue

            dialog = turn_id.rsplit("_", 1)[0]
            gender = turn_id.rsplit("_", 1)[1][0]
            wav_path = os.path.join(
                session_dir, "sentences", "wav", dialog, turn_id + ".wav"
            )
            if not os.path.exists(wav_path):
                continue

            rows.append(
                {
                    "path": wav_path,
                    "speaker": f"{turn_id[:5]}_{gender}",
                    "emotion_code": emotion_code,
                    "emotion": IEMOCAP_EMOTION_MAP[emotion_code],
                    "session": turn_id[:5],
                    "dialog": dialog,
                    "filename": turn_id + ".wav",
                    "dataset": "iemocap",
                }
            )
    return rows


def discover_iemocap(iemocap_dir: str) -> pd.DataFrame:
    eval_files = sorted(
        glob.glob(
            os.path.join(iemocap_dir, "Session*", "dialog", "EmoEvaluation", "*.txt")
        )
    )
    eval_files = [f for f in eval_files if not os.path.basename(f).startswith(".")]
    rows: List[Dict[str, Any]] = []
    for eval_path in eval_files:
        session_dir = eval_path.split(os.sep + "dialog" + os.sep)[0]
        rows.extend(parse_iemocap_eval_file(eval_path, session_dir))
    df = pd.DataFrame(rows)
    if len(df) == 0:
        raise FileNotFoundError(
            f"No valid IEMOCAP utterances found under: {iemocap_dir}\n"
            "Expected Session*/dialog/EmoEvaluation/*.txt label files and "
            "matching Session*/sentences/wav/<dialog>/<turn>.wav audio."
        )
    return df.reset_index(drop=True)


def discover_dataset(cfg: Config) -> pd.DataFrame:
    dataset_name = normalize_dataset_name(cfg.DATASET)
    if dataset_name == "cremad":
        return discover_cremad(cfg.CREMA_D_DIR)
    if dataset_name == "ravdess":
        return discover_ravdess(cfg.RAVDESS_DIR, cfg.RAVDESS_ZIP)
    if dataset_name == "iemocap":
        return discover_iemocap(cfg.IEMOCAP_DIR)
    raise ValueError(f"Unsupported dataset: {cfg.DATASET}")


def load_audio_fixed(
    path: str,
    sr: int,
    max_seconds: float,
    normalize_peak: bool = False,
) -> np.ndarray:
    y, _ = librosa.load(path, sr=sr, mono=True)
    max_len = int(sr * max_seconds)
    if len(y) > max_len:
        y = y[:max_len]
    elif len(y) < max_len:
        y = np.pad(y, (0, max_len - len(y)), mode="constant")
    y = y.astype(np.float32)
    if normalize_peak:
        peak = np.max(np.abs(y)) + 1e-8
        y = y / peak
    return y


def waveform_to_logmel(y: np.ndarray, cfg: Config) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=cfg.SR,
        n_fft=cfg.N_FFT,
        hop_length=cfg.HOP_LENGTH,
        win_length=cfg.WIN_LENGTH,
        n_mels=cfg.N_MELS,
        fmin=cfg.FMIN,
        fmax=cfg.FMAX,
        power=2.0,
    )
    logmel = librosa.power_to_db(mel, ref=np.max).astype(np.float32)
    logmel = (logmel - logmel.mean()) / (logmel.std() + 1e-6)
    return logmel


def safe_nan_to_num(x: np.ndarray, value: float = 0.0) -> np.ndarray:
    return np.nan_to_num(x, nan=value, posinf=value, neginf=value)


def make_opensmile_extractor(cfg: Config):
    if opensmile is None:
        raise ImportError(
            "opensmile is not installed. Install it with:\n"
            "    python -m pip install opensmile\n"
            "or set CFG.FEATURE_BACKEND = 'librosa' for the fallback ablation."
        )
    try:
        feature_set = getattr(opensmile.FeatureSet, cfg.OPENSMILE_FEATURE_SET)
    except Exception as exc:
        available = [x for x in dir(opensmile.FeatureSet) if not x.startswith("_")]
        raise ValueError(
            f"Unknown openSMILE FeatureSet: {cfg.OPENSMILE_FEATURE_SET}.\n"
            f"Available FeatureSet names include: {available}"
        ) from exc
    return opensmile.Smile(
        feature_set=feature_set,
        feature_level=opensmile.FeatureLevel.Functionals,
    )


def extract_egemaps_features_from_signal(y: np.ndarray, sr: int, smile) -> Tuple[np.ndarray, List[str]]:
    try:
        df_feat = smile.process_signal(y.astype(np.float32), sr)
    except Exception as exc:
        raise RuntimeError(f"openSMILE failed on an audio signal: {exc}") from exc

    if len(df_feat) == 0:
        raise RuntimeError("openSMILE returned an empty feature frame.")

    df_num = df_feat.apply(pd.to_numeric, errors="coerce")
    arr = df_num.to_numpy(dtype=np.float32)
    if arr.ndim == 2 and arr.shape[0] > 1:
        vec = np.nanmean(arr, axis=0)
    else:
        vec = arr.reshape(-1)
    names = [str(c) for c in df_num.columns.tolist()]
    return safe_nan_to_num(vec.astype(np.float32)), names


def extract_librosa_primitives(y: np.ndarray, cfg: Config) -> np.ndarray:
    hop = cfg.HOP_LENGTH
    frame_length = cfg.WIN_LENGTH

    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop)[0]
    rms = safe_nan_to_num(rms)
    rms_mean = float(np.mean(rms))
    rms_std = float(np.std(rms))
    if np.max(rms) > 1e-8:
        pause_threshold = max(1e-4, 0.10 * float(np.max(rms)))
        pause_ratio = float(np.mean(rms < pause_threshold))
    else:
        pause_ratio = 1.0

    try:
        f0 = librosa.yin(
            y,
            fmin=cfg.YIN_FMIN,
            fmax=cfg.YIN_FMAX,
            sr=cfg.SR,
            frame_length=cfg.YIN_FRAME_LENGTH,
            hop_length=hop,
        )
        f0 = safe_nan_to_num(f0)
        if len(rms) == len(f0) and np.max(rms) > 1e-8:
            voiced = rms > np.percentile(rms, 35)
            f0_voiced = f0[voiced]
        else:
            f0_voiced = f0
        f0_voiced = f0_voiced[(f0_voiced > cfg.YIN_FMIN) & (f0_voiced < cfg.YIN_FMAX)]
        if len(f0_voiced) < 3:
            f0_mean, f0_std, f0_range = 0.0, 0.0, 0.0
        else:
            f0_mean = float(np.mean(f0_voiced))
            f0_std = float(np.std(f0_voiced))
            f0_range = float(np.percentile(f0_voiced, 95) - np.percentile(f0_voiced, 5))
    except Exception:
        f0_mean, f0_std, f0_range = 0.0, 0.0, 0.0

    centroid = librosa.feature.spectral_centroid(y=y, sr=cfg.SR, n_fft=cfg.N_FFT, hop_length=hop)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=cfg.SR, n_fft=cfg.N_FFT, hop_length=hop)[0]
    flatness = librosa.feature.spectral_flatness(y=y, n_fft=cfg.N_FFT, hop_length=hop)[0]
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=frame_length, hop_length=hop)[0]

    centroid_mean = float(np.mean(safe_nan_to_num(centroid)))
    bandwidth_mean = float(np.mean(safe_nan_to_num(bandwidth)))
    flatness_mean = float(np.mean(safe_nan_to_num(flatness)))
    zcr_mean = float(np.mean(safe_nan_to_num(zcr)))

    try:
        onset_env = librosa.onset.onset_strength(y=y, sr=cfg.SR, hop_length=hop)
        onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=cfg.SR, hop_length=hop)
        duration = len(y) / cfg.SR
        onset_rate = float(len(onset_frames) / max(duration, 1e-6))
        if len(onset_frames) >= 3:
            onset_times = librosa.frames_to_time(onset_frames, sr=cfg.SR, hop_length=hop)
            intervals = np.diff(onset_times)
            onset_interval_cv = float(np.std(intervals) / (np.mean(intervals) + 1e-6))
        else:
            onset_interval_cv = 0.0
    except Exception:
        onset_rate = 0.0
        onset_interval_cv = 0.0

    feats = np.array(
        [
            f0_mean,
            f0_std,
            f0_range,
            rms_mean,
            rms_std,
            pause_ratio,
            centroid_mean,
            bandwidth_mean,
            flatness_mean,
            zcr_mean,
            onset_rate,
            onset_interval_cv,
        ],
        dtype=np.float32,
    )
    return safe_nan_to_num(feats)


def concept_cache_csv_path(cfg: Config) -> str:
    safe_backend = str(cfg.FEATURE_BACKEND).lower().strip()
    safe_set = str(cfg.OPENSMILE_FEATURE_SET).strip()
    return os.path.join(cfg.OUT_DIR, f"concept_feature_cache_{safe_backend}_{safe_set}.csv")


def try_load_concept_feature_cache(
    df: pd.DataFrame,
    cfg: Config
) -> Optional[Tuple[Dict[str, np.ndarray], List[str]]]:
    if not cfg.CACHE_CONCEPT_FEATURES_TO_CSV or cfg.FORCE_REBUILD_CONCEPT_CACHE:
        return None

    path = concept_cache_csv_path(cfg)
    if not os.path.exists(path):
        return None

    try:
        feat_df = pd.read_csv(path)

        if "path" not in feat_df.columns:
            return None

        feat_df["path"] = feat_df["path"].astype(str)
        feat_df = feat_df.drop_duplicates(subset=["path"], keep="last")

        needed_paths = set(df["path"].astype(str).tolist())
        got_paths = set(feat_df["path"].astype(str).tolist())

        if not needed_paths.issubset(got_paths):
            return None

        feature_names = [c for c in feat_df.columns if c not in {"path", "filename"}]

        feat_df = feat_df.set_index("path")

        cache: Dict[str, np.ndarray] = {}

        for p in df["path"].astype(str).tolist():
            vec = (
                feat_df
                .loc[[p], feature_names]
                .to_numpy(dtype=np.float32)
                .reshape(-1)
            )
            cache[p] = safe_nan_to_num(vec.astype(np.float32))

        print(f"Loaded concept feature cache: {path}")
        return cache, feature_names

    except Exception as exc:
        print(f"Could not load concept feature cache; rebuilding. Reason: {exc}")
        return None


def save_concept_feature_cache(
    feature_cache: Dict[str, np.ndarray],
    feature_names: List[str],
    cfg: Config
) -> None:
    if not cfg.CACHE_CONCEPT_FEATURES_TO_CSV:
        return

    path = concept_cache_csv_path(cfg)

    try:
        rows: List[Dict[str, Any]] = []

        for p, vec in feature_cache.items():
            row: Dict[str, Any] = {
                "path": str(p),
                "filename": os.path.basename(str(p)),
            }

            for name, val in zip(feature_names, vec):
                row[str(name)] = float(val)

            rows.append(row)

        pd.DataFrame(rows).to_csv(path, index=False)
        print(f"Saved concept feature cache: {path}")

    except Exception as exc:
        print(f"Could not save concept feature cache: {exc}")


def _extract_file_features(
    path: str,
    cfg: "Config",
    backend: str,
    smile: Any,
    need_concepts: bool,
) -> Tuple[str, np.ndarray, Optional[np.ndarray], Optional[List[str]]]:
    y_model = load_audio_fixed(path, cfg.SR, cfg.MAX_SECONDS, normalize_peak=True)
    logmel = waveform_to_logmel(y_model, cfg)

    concept_vec: Optional[np.ndarray] = None
    names: Optional[List[str]] = None
    if need_concepts:
        y_raw = load_audio_fixed(path, cfg.SR, cfg.MAX_SECONDS, normalize_peak=False)
        if backend == "opensmile":
            concept_vec, names = extract_egemaps_features_from_signal(y_raw, cfg.SR, smile)
        else:
            concept_vec = extract_librosa_primitives(y_raw, cfg)
            names = list(LIBROSA_PRIMITIVE_NAMES)
    return path, logmel, concept_vec, names


_FEATURE_WORKER_STATE: Dict[str, Any] = {}


def _init_feature_worker(cfg: "Config", backend: str, need_concepts: bool) -> None:
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(var, "1")
    try:
        torch.set_num_threads(1)
    except Exception:
        pass
    smile = None
    if need_concepts and backend == "opensmile":
        smile = make_opensmile_extractor(cfg)
    _FEATURE_WORKER_STATE.update(
        {"cfg": cfg, "backend": backend, "need_concepts": need_concepts, "smile": smile}
    )


def _feature_worker_extract(path: str):
    s = _FEATURE_WORKER_STATE
    return _extract_file_features(path, s["cfg"], s["backend"], s["smile"], s["need_concepts"])


def build_feature_cache(df: pd.DataFrame, cfg: Config) -> Dict[str, Dict[str, np.ndarray]]:
    backend = str(cfg.FEATURE_BACKEND).lower().strip()
    if backend not in {"opensmile", "librosa"}:
        raise ValueError("CFG.FEATURE_BACKEND must be 'opensmile' or 'librosa'.")

    os.makedirs(cfg.OUT_DIR, exist_ok=True)

    concept_feature_cache: Optional[Dict[str, np.ndarray]] = None
    concept_feature_names: Optional[List[str]] = None

    cached = try_load_concept_feature_cache(df, cfg)
    if cached is not None:
        concept_feature_cache, concept_feature_names = cached

    need_concepts = concept_feature_cache is None
    if need_concepts:
        if backend == "opensmile":
            print(f"Using openSMILE feature set: {cfg.OPENSMILE_FEATURE_SET}")
        else:
            print("Using librosa fallback primitive features for concepts.")

    paths = df["path"].astype(str).tolist()
    n_jobs = resolve_n_jobs(getattr(cfg, "FEATURE_EXTRACTION_JOBS", -1), len(paths))
    desc = "Extracting logmel/concept features"

    results: Dict[str, Tuple[np.ndarray, Optional[np.ndarray], Optional[List[str]]]] = {}

    if n_jobs <= 1:
        smile = make_opensmile_extractor(cfg) if (need_concepts and backend == "opensmile") else None
        for path in tqdm(paths, desc=desc):
            _, logmel, cvec, names = _extract_file_features(path, cfg, backend, smile, need_concepts)
            results[path] = (logmel, cvec, names)
    else:
        print(f"Extracting features in parallel across {n_jobs} processes")
        chunksize = max(1, len(paths) // (n_jobs * 8))
        with ProcessPoolExecutor(
            max_workers=n_jobs,
            initializer=_init_feature_worker,
            initargs=(cfg, backend, need_concepts),
        ) as executor:
            for path, logmel, cvec, names in tqdm(
                executor.map(_feature_worker_extract, paths, chunksize=chunksize),
                total=len(paths),
                desc=desc,
            ):
                results[path] = (logmel, cvec, names)

    cache: Dict[str, Dict[str, np.ndarray]] = {}
    new_concept_feature_cache: Dict[str, np.ndarray] = {}
    final_feature_names: Optional[List[str]] = concept_feature_names

    for path in paths:
        logmel, cvec, names = results[path]
        if need_concepts:
            if final_feature_names is None:
                final_feature_names = names
            elif names is not None and len(names) != len(final_feature_names):
                raise RuntimeError("Inconsistent concept feature dimensionality across files.")
            concept_vec = cvec
            new_concept_feature_cache[path] = cvec
        else:
            concept_vec = concept_feature_cache[path]

        cache[path] = {
            "logmel": logmel,
            "concept_features": concept_vec.astype(np.float32),
        }

    if need_concepts and final_feature_names is not None:
        save_concept_feature_cache(new_concept_feature_cache, final_feature_names, cfg)

    if final_feature_names is None:
        raise RuntimeError("No concept feature names were extracted.")

    cache["__concept_feature_names__"] = {"names": np.array(final_feature_names, dtype=object)}
    return cache


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _lower_names(feature_names: List[str]) -> List[str]:
    return [str(n).lower() for n in feature_names]


def _match_indices(
    feature_names: List[str],
    any_of: Tuple[str, ...],
    all_of: Tuple[str, ...] = (),
    none_of: Tuple[str, ...] = (),
) -> List[int]:
    names_l = _lower_names(feature_names)
    idxs = []
    any_l = tuple(s.lower() for s in any_of)
    all_l = tuple(s.lower() for s in all_of)
    none_l = tuple(s.lower() for s in none_of)
    for i, name in enumerate(names_l):
        if any_l and not any(s in name for s in any_l):
            continue
        if all_l and not all(s in name for s in all_l):
            continue
        if none_l and any(s in name for s in none_l):
            continue
        idxs.append(i)
    return idxs


def _group_value(z: np.ndarray, idxs: List[int]) -> np.ndarray:
    if len(idxs) == 0:
        return np.zeros(z.shape[0], dtype=np.float32)
    return np.mean(z[:, idxs], axis=1).astype(np.float32)


def _feature_groups(feature_names: List[str]) -> Dict[str, List[int]]:
    groups: Dict[str, List[int]] = {}

    groups["f0_mean"] = _match_indices(
        feature_names,
        any_of=("f0", "pitch"),
        none_of=("stddev", "std", "range", "pctl", "percentile"),
    )
    groups["f0_var"] = _match_indices(
        feature_names,
        any_of=("f0", "pitch"),
        none_of=(),
    )
    groups["f0_var"] = [
        i for i in groups["f0_var"]
        if any(tok in feature_names[i].lower() for tok in ("std", "range", "pctl", "percentile"))
    ]

    groups["loud_mean"] = _match_indices(
        feature_names,
        any_of=("loudness", "rms", "energy"),
        none_of=("stddev", "std", "range", "pctl", "percentile"),
    )
    groups["loud_var"] = _match_indices(
        feature_names,
        any_of=("loudness", "rms", "energy"),
    )
    groups["loud_var"] = [
        i for i in groups["loud_var"]
        if any(tok in feature_names[i].lower() for tok in ("std", "range", "pctl", "percentile"))
    ]

    groups["spectral_flux"] = _match_indices(feature_names, any_of=("spectralflux", "spectral_flux"))
    groups["brightness"] = _match_indices(
        feature_names,
        any_of=("alpharatio", "hammarberg", "centroid", "brightness"),
    )
    groups["spectral_breadth"] = _match_indices(
        feature_names,
        any_of=("mfcc", "slope", "bandwidth", "lspfrequency", "spectral"),
    )
    groups["flatness_zcr"] = _match_indices(feature_names, any_of=("flatness", "zcr", "zero_crossing"))

    groups["jitter"] = _match_indices(feature_names, any_of=("jitter",))
    groups["shimmer"] = _match_indices(feature_names, any_of=("shimmer",))
    groups["hnr"] = _match_indices(feature_names, any_of=("hnr", "harmonic"))
    groups["voice_quality"] = sorted(set(groups["jitter"] + groups["shimmer"] + groups["hnr"]))

    groups["voiced_rate"] = _match_indices(feature_names, any_of=("voicedsegmentspersec", "voicedsegment", "onset_rate"))
    groups["segment_lengths"] = _match_indices(
        feature_names,
        any_of=("voicedsegmentlength", "unvoicedsegmentlength", "segmentlength", "onset_interval"),
    )
    groups["pause"] = _match_indices(feature_names, any_of=("unvoiced", "pause", "silence"))

    direct = {name: i for i, name in enumerate(feature_names)}
    for k in ["f0_mean", "f0_std", "f0_range", "rms_mean", "rms_std", "pause_ratio",
              "centroid_mean", "bandwidth_mean", "flatness_mean", "zcr_mean",
              "onset_rate", "onset_interval_cv"]:
        if k in direct:
            idx = direct[k]
            if k == "f0_mean":
                groups["f0_mean"] = sorted(set(groups["f0_mean"] + [idx]))
            elif k in {"f0_std", "f0_range"}:
                groups["f0_var"] = sorted(set(groups["f0_var"] + [idx]))
            elif k == "rms_mean":
                groups["loud_mean"] = sorted(set(groups["loud_mean"] + [idx]))
            elif k == "rms_std":
                groups["loud_var"] = sorted(set(groups["loud_var"] + [idx]))
            elif k == "pause_ratio":
                groups["pause"] = sorted(set(groups["pause"] + [idx]))
            elif k in {"centroid_mean", "bandwidth_mean"}:
                groups["brightness"] = sorted(set(groups["brightness"] + [idx]))
                groups["spectral_breadth"] = sorted(set(groups["spectral_breadth"] + [idx]))
            elif k in {"flatness_mean", "zcr_mean"}:
                groups["flatness_zcr"] = sorted(set(groups["flatness_zcr"] + [idx]))
            elif k == "onset_rate":
                groups["voiced_rate"] = sorted(set(groups["voiced_rate"] + [idx]))
            elif k == "onset_interval_cv":
                groups["segment_lengths"] = sorted(set(groups["segment_lengths"] + [idx]))

    return groups


def write_feature_group_report(feature_names: List[str], cfg: Config) -> None:
    groups = _feature_groups(feature_names)
    rows = []
    for group, idxs in groups.items():
        rows.append({
            "group": group,
            "n_features": len(idxs),
            "features": "; ".join([feature_names[i] for i in idxs[:25]]),
        })
    path = os.path.join(cfg.OUT_DIR, "concept_feature_group_report.csv")
    pd.DataFrame(rows).to_csv(path, index=False)
    print("Concept feature group report:", path)


def compute_speaker_baselines(z: np.ndarray, speakers: np.ndarray) -> Dict[str, np.ndarray]:
    speakers = np.asarray(speakers, dtype=str)
    baselines: Dict[str, np.ndarray] = {}
    for spk in sorted(np.unique(speakers).tolist()):
        mask = speakers == spk
        if np.any(mask):
            baselines[str(spk)] = np.mean(z[mask], axis=0).astype(np.float32)
    return baselines


def baseline_matrix_for_rows(
    z: np.ndarray,
    speakers: np.ndarray,
    baseline_by_speaker: Dict[str, np.ndarray],
    fallback_global: np.ndarray,
) -> np.ndarray:
    speakers = np.asarray(speakers, dtype=str)
    rows = []
    for spk in speakers:
        rows.append(baseline_by_speaker.get(str(spk), fallback_global))
    return np.stack(rows, axis=0).astype(np.float32)


def build_concepts_from_baseline_and_deviation(
    z: np.ndarray,
    baseline_z: np.ndarray,
    feature_names: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    dev_z = z - baseline_z
    groups = _feature_groups(feature_names)

    f0_mean_d = _group_value(dev_z, groups["f0_mean"])
    f0_var_d = _group_value(dev_z, groups["f0_var"])
    loud_mean_d = _group_value(dev_z, groups["loud_mean"])
    loud_var_d = _group_value(dev_z, groups["loud_var"])
    spectral_flux_d = _group_value(dev_z, groups["spectral_flux"])
    brightness_d = _group_value(dev_z, groups["brightness"])
    flatness_zcr_d = _group_value(dev_z, groups["flatness_zcr"])
    voice_quality_d = _group_value(dev_z, groups["voice_quality"])
    hnr_d = _group_value(dev_z, groups["hnr"])
    pause_d = _group_value(dev_z, groups["pause"])
    voiced_rate_d = _group_value(dev_z, groups["voiced_rate"])
    segment_len_d = _group_value(dev_z, groups["segment_lengths"])

    vocal_arousal = 0.50 * loud_mean_d + 0.30 * f0_mean_d + 0.20 * spectral_flux_d
    pitch_instability = 0.70 * f0_var_d + 0.30 * voice_quality_d
    energy_variability = 0.80 * loud_var_d + 0.20 * spectral_flux_d
    pause_hesitation = 0.70 * pause_d - 0.30 * voiced_rate_d
    voice_tension = 0.35 * voice_quality_d + 0.25 * brightness_d + 0.20 * flatness_zcr_d - 0.20 * hnr_d
    rhythm_irregularity = 0.60 * segment_len_d + 0.25 * np.abs(voiced_rate_d) + 0.15 * np.abs(loud_var_d)

    aff_raw = np.stack(
        [
            vocal_arousal,
            pitch_instability,
            energy_variability,
            pause_hesitation,
            voice_tension,
            rhythm_irregularity,
        ],
        axis=1,
    )

    f0_mean_b = _group_value(baseline_z, groups["f0_mean"])
    loud_mean_b = _group_value(baseline_z, groups["loud_mean"])
    brightness_b = _group_value(baseline_z, groups["brightness"])
    spectral_breadth_b = _group_value(baseline_z, groups["spectral_breadth"])
    flatness_zcr_b = _group_value(baseline_z, groups["flatness_zcr"])
    spectral_flux_b = _group_value(baseline_z, groups["spectral_flux"])
    voice_quality_b = _group_value(baseline_z, groups["voice_quality"])
    voiced_rate_b = _group_value(baseline_z, groups["voiced_rate"])

    baseline_pitch_level = f0_mean_b
    habitual_loudness_level = loud_mean_b
    timbre_brightness = 0.60 * brightness_b + 0.20 * spectral_flux_b + 0.20 * flatness_zcr_b
    spectral_breadth = spectral_breadth_b
    articulation_sharpness = 0.45 * flatness_zcr_b + 0.35 * spectral_flux_b + 0.20 * voice_quality_b
    tempo_tendency = voiced_rate_b

    style_raw = np.stack(
        [
            baseline_pitch_level,
            habitual_loudness_level,
            timbre_brightness,
            spectral_breadth,
            articulation_sharpness,
            tempo_tendency,
        ],
        axis=1,
    )

    aff_targets = sigmoid_np(np.clip(aff_raw, -4.0, 4.0)).astype(np.float32)
    style_targets = sigmoid_np(np.clip(style_raw, -4.0, 4.0)).astype(np.float32)
    return aff_targets, style_targets


class CREMADConceptDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        cache: Dict[str, Dict[str, np.ndarray]],
        aff_targets: np.ndarray,
        style_targets: np.ndarray,
        speaker_to_local: Optional[Dict[str, int]] = None,
    ):
        self.df = df.reset_index(drop=True)
        self.cache = cache
        self.aff_targets = aff_targets.astype(np.float32)
        self.style_targets = style_targets.astype(np.float32)
        self.speaker_to_local = speaker_to_local or {}

        assert len(self.df) == len(self.aff_targets) == len(self.style_targets)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        path = str(row["path"])
        logmel = self.cache[path]["logmel"]
        x = torch.tensor(logmel[None, :, :], dtype=torch.float32)
        y = torch.tensor(int(row["emotion"]), dtype=torch.long)

        speaker_str = str(row["speaker"])
        speaker_local = self.speaker_to_local.get(speaker_str, -1)
        speaker_local = torch.tensor(speaker_local, dtype=torch.long)

        aff = torch.tensor(self.aff_targets[idx], dtype=torch.float32)
        style = torch.tensor(self.style_targets[idx], dtype=torch.float32)

        return {
            "x": x,
            "y": y,
            "speaker_local": speaker_local,
            "aff_targets": aff,
            "style_targets": style,
        }


class GradReverseFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambd: float) -> torch.Tensor:
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.lambd * grad_output, None


def grad_reverse(x: torch.Tensor, lambd: float = 1.0) -> torch.Tensor:
    return GradReverseFn.apply(x, lambd)


class CRNNEncoder(nn.Module):
    def __init__(self, n_mels: int, h_dim: int, dropout: float):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),

            nn.Conv2d(64, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1)),
        )
        self.gru = nn.GRU(
            input_size=96,
            hidden_size=h_dim // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.proj = nn.Sequential(
            nn.LayerNorm(h_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.conv(x)
        z = z.mean(dim=2)
        z = z.transpose(1, 2)
        out, _ = self.gru(z)
        h = out.mean(dim=1)
        h = self.proj(h)
        return h


class ConceptEmbeddingBottleneck(nn.Module):
    def __init__(self, h_dim: int, n_concepts: int, emb_dim: int, dropout: float):
        super().__init__()
        self.n_concepts = n_concepts
        self.emb_dim = emb_dim
        self.dropout = nn.Dropout(dropout)
        self.pos = nn.Linear(h_dim, n_concepts * emb_dim)
        self.neg = nn.Linear(h_dim, n_concepts * emb_dim)
        self.scorer = nn.Linear(2 * emb_dim, 1)

    @property
    def rep_dim(self) -> int:
        return self.n_concepts * self.emb_dim

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b = h.size(0)
        hd = self.dropout(h)
        pos = self.pos(hd).view(b, self.n_concepts, self.emb_dim)
        neg = self.neg(hd).view(b, self.n_concepts, self.emb_dim)
        score = self.scorer(torch.cat([pos, neg], dim=-1)).squeeze(-1)
        act = torch.sigmoid(score)
        mixed = act.unsqueeze(-1) * pos + (1.0 - act).unsqueeze(-1) * neg
        emb = mixed.reshape(b, self.n_concepts * self.emb_dim)
        return act, emb


class DisentangledAffectiveStyleCBM(nn.Module):
    def __init__(
        self,
        n_mels: int,
        h_dim: int,
        n_aff: int,
        n_style: int,
        n_emotions: int,
        n_train_speakers: int,
        dropout: float,
        emotion_head_input: str = "aff",
        use_aff_concept_branch: bool = True,
        use_style_branch: bool = True,
        use_concept_embeddings: bool = False,
        concept_emb_dim: int = 16,
    ):
        super().__init__()
        if emotion_head_input not in {"aff", "encoder"}:
            raise ValueError("emotion_head_input must be 'aff' or 'encoder'.")
        if emotion_head_input == "aff" and not use_aff_concept_branch:
            raise ValueError("Affective concept branch is required when emotion_head_input='aff'.")

        self.n_aff = n_aff
        self.n_style = n_style
        self.emotion_head_input = emotion_head_input
        self.use_aff_concept_branch = use_aff_concept_branch
        self.use_style_branch = use_style_branch
        self.use_concept_embeddings = use_concept_embeddings
        self.encoder = CRNNEncoder(n_mels=n_mels, h_dim=h_dim, dropout=dropout)

        if use_concept_embeddings:
            self.aff_bottleneck = ConceptEmbeddingBottleneck(h_dim, n_aff, concept_emb_dim, dropout)
            self.style_bottleneck = ConceptEmbeddingBottleneck(h_dim, n_style, concept_emb_dim, dropout)
            self.aff_rep_dim = self.aff_bottleneck.rep_dim
            self.style_rep_dim = self.style_bottleneck.rep_dim
        else:
            self.aff_head = nn.Sequential(
                nn.Linear(h_dim, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(128, n_aff),
                nn.Sigmoid(),
            )
            self.style_head = nn.Sequential(
                nn.Linear(h_dim, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(128, n_style),
                nn.Sigmoid(),
            )
            self.aff_rep_dim = n_aff
            self.style_rep_dim = n_style

        self.emotion_head = nn.Sequential(
            nn.Linear(self.aff_rep_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, n_emotions),
        )

        self.encoder_emotion_head = nn.Sequential(
            nn.Linear(h_dim, 96),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(96, n_emotions),
        )

        self.style_speaker_head = nn.Sequential(
            nn.Linear(self.style_rep_dim, 96),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(96, n_train_speakers),
        )

        self.aff_speaker_adv_head = nn.Sequential(
            nn.Linear(self.aff_rep_dim, 96),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(96, n_train_speakers),
        )

        self.style_emotion_adv_head = nn.Sequential(
            nn.Linear(self.style_rep_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, n_emotions),
        )

    def forward(self, x: torch.Tensor, grl_lambda: float = 1.0) -> Dict[str, torch.Tensor]:
        h = self.encoder(x)
        b = h.size(0)

        if self.use_aff_concept_branch:
            if self.use_concept_embeddings:
                c_aff, c_aff_rep = self.aff_bottleneck(h)
            else:
                c_aff = self.aff_head(h)
                c_aff_rep = c_aff
        else:
            c_aff = torch.zeros(b, self.n_aff, dtype=h.dtype, device=h.device)
            c_aff_rep = torch.zeros(b, self.aff_rep_dim, dtype=h.dtype, device=h.device)

        if self.use_style_branch:
            if self.use_concept_embeddings:
                c_style, c_style_rep = self.style_bottleneck(h)
            else:
                c_style = self.style_head(h)
                c_style_rep = c_style
        else:
            c_style = torch.zeros(b, self.n_style, dtype=h.dtype, device=h.device)
            c_style_rep = torch.zeros(b, self.style_rep_dim, dtype=h.dtype, device=h.device)

        if self.emotion_head_input == "encoder":
            emotion_logits = self.encoder_emotion_head(h)
        else:
            emotion_logits = self.emotion_head(c_aff_rep)
        style_speaker_logits = self.style_speaker_head(c_style_rep)
        aff_speaker_adv_logits = self.aff_speaker_adv_head(grad_reverse(c_aff_rep, grl_lambda))
        style_emotion_adv_logits = self.style_emotion_adv_head(grad_reverse(c_style_rep, grl_lambda))

        return {
            "h": h,
            "c_aff": c_aff,
            "c_style": c_style,
            "c_aff_rep": c_aff_rep,
            "c_style_rep": c_style_rep,
            "emotion_logits": emotion_logits,
            "style_speaker_logits": style_speaker_logits,
            "aff_speaker_adv_logits": aff_speaker_adv_logits,
            "style_emotion_adv_logits": style_emotion_adv_logits,
        }


def batch_correlation_penalty(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if a.size(0) < 2:
        return torch.tensor(0.0, device=a.device)
    a0 = a - a.mean(dim=0, keepdim=True)
    b0 = b - b.mean(dim=0, keepdim=True)
    a0 = a0 / (a0.std(dim=0, keepdim=True) + 1e-6)
    b0 = b0 / (b0.std(dim=0, keepdim=True) + 1e-6)
    corr = (a0.T @ b0) / max(a.size(0) - 1, 1)
    return (corr ** 2).mean()


def ece_score(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 15) -> float:
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == y_true).astype(np.float32)

    ece = 0.0
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i < n_bins - 1:
            mask = (confidences >= lo) & (confidences < hi)
        else:
            mask = (confidences >= lo) & (confidences <= hi)
        if np.any(mask):
            bin_conf = confidences[mask].mean()
            bin_acc = accuracies[mask].mean()
            ece += np.mean(mask) * abs(bin_acc - bin_conf)
    return float(ece)


def compute_metrics(y_true: np.ndarray, logits: np.ndarray) -> Dict[str, float]:
    probs = torch.softmax(torch.tensor(logits), dim=1).numpy()
    y_pred = probs.argmax(axis=1)
    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "uar": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "ece": float(ece_score(probs, y_true)),
    }


def grl_schedule(epoch: int, num_epochs: int, max_lambda: float, warmup_epochs: int = 0) -> float:
    if warmup_epochs > 0:
        if epoch < warmup_epochs:
            return 0.0
        epoch = epoch - warmup_epochs
        num_epochs = max(num_epochs - warmup_epochs, 1)
    p = epoch / max(num_epochs - 1, 1)
    return float(max_lambda * (2.0 / (1.0 + math.exp(-10 * p)) - 1.0))


def class_weights_from_labels(labels: np.ndarray, n_classes: int) -> torch.Tensor:
    counts = np.bincount(labels.astype(np.int64), minlength=n_classes).astype(np.float32)
    weights = counts.sum() / (n_classes * np.maximum(counts, 1.0))
    return torch.tensor(weights, dtype=torch.float32)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    cfg: Config,
    emotion_weights: torch.Tensor,
    grl_lambda: float,
    scaler: Optional["torch.cuda.amp.GradScaler"] = None,
) -> Dict[str, float]:
    model.train()
    totals = {
        "loss": 0.0,
        "emo": 0.0,
        "aff": 0.0,
        "style": 0.0,
        "style_spk": 0.0,
        "aff_spk_adv": 0.0,
        "style_emo_adv": 0.0,
        "orth": 0.0,
    }
    n = 0

    emotion_weights = emotion_weights.to(device)

    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        speaker_local = batch["speaker_local"].to(device)
        aff_targets = batch["aff_targets"].to(device)
        style_targets = batch["style_targets"].to(device)

        valid_spk = speaker_local >= 0
        if not valid_spk.all():
            raise RuntimeError("Training batch contains speakers missing from local speaker mapping.")

        with amp_autocast(cfg, device):
            out = model(x, grl_lambda=grl_lambda)

            loss_emo = F.cross_entropy(out["emotion_logits"], y, weight=emotion_weights)
            loss_aff = F.smooth_l1_loss(out["c_aff"], aff_targets)
            loss_style = F.smooth_l1_loss(out["c_style"], style_targets)
            loss_style_spk = F.cross_entropy(out["style_speaker_logits"], speaker_local)
            loss_aff_spk_adv = F.cross_entropy(out["aff_speaker_adv_logits"], speaker_local)
            loss_style_emo_adv = F.cross_entropy(out["style_emotion_adv_logits"], y, weight=emotion_weights)
            loss_orth = batch_correlation_penalty(out["c_aff_rep"], out["c_style_rep"])

            loss = loss_emo

            if cfg.USE_AFF_CONCEPT_BRANCH and cfg.USE_AFF_CONCEPT_SUPERVISION:
                loss = loss + cfg.LAMBDA_AFF_CONCEPT * loss_aff

            if cfg.USE_STYLE_BRANCH:
                loss = loss + cfg.LAMBDA_STYLE_CONCEPT * loss_style
                loss = loss + cfg.LAMBDA_STYLE_SPEAKER * loss_style_spk

            if cfg.USE_AFF_CONCEPT_BRANCH and cfg.USE_AFF_SPEAKER_ADVERSARY:
                loss = loss + cfg.LAMBDA_AFF_SPK_ADV * loss_aff_spk_adv

            if cfg.USE_STYLE_EMOTION_ADVERSARY and cfg.USE_STYLE_BRANCH:
                loss = loss + cfg.LAMBDA_STYLE_EMO_ADV * loss_style_emo_adv

            if cfg.USE_ORTHOGONALITY and cfg.USE_STYLE_BRANCH:
                loss = loss + cfg.LAMBDA_ORTH * loss_orth

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        bs = x.size(0)
        n += bs
        totals["loss"] += float(loss.item()) * bs
        totals["emo"] += float(loss_emo.item()) * bs
        totals["aff"] += float(loss_aff.item()) * bs
        totals["style"] += float(loss_style.item()) * bs
        totals["style_spk"] += float(loss_style_spk.item()) * bs
        totals["aff_spk_adv"] += float(loss_aff_spk_adv.item()) * bs
        totals["style_emo_adv"] += float(loss_style_emo_adv.item()) * bs
        totals["orth"] += float(loss_orth.item()) * bs

    return {k: v / max(n, 1) for k, v in totals.items()}


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    cfg: Optional[Config] = None,
) -> Dict[str, Any]:
    model.eval()
    ys = []
    logits = []
    c_affs = []
    c_styles = []
    c_aff_reps = []
    c_style_reps = []
    aff_targets = []
    style_targets = []

    for batch in loader:
        x = batch["x"].to(device)
        out = model(x, grl_lambda=0.0)
        ys.append(batch["y"].detach().cpu().numpy())
        logits.append(out["emotion_logits"].cpu().numpy())
        c_affs.append(out["c_aff"].cpu().numpy())
        c_styles.append(out["c_style"].cpu().numpy())
        c_aff_reps.append(out["c_aff_rep"].cpu().numpy())
        c_style_reps.append(out["c_style_rep"].cpu().numpy())
        aff_targets.append(batch["aff_targets"].detach().cpu().numpy())
        style_targets.append(batch["style_targets"].detach().cpu().numpy())

    y_true = np.concatenate(ys)
    logit_arr = np.concatenate(logits)
    c_aff = np.concatenate(c_affs)
    c_style = np.concatenate(c_styles)
    c_aff_rep = np.concatenate(c_aff_reps)
    c_style_rep = np.concatenate(c_style_reps)
    aff_t = np.concatenate(aff_targets)
    style_t = np.concatenate(style_targets)

    metrics = compute_metrics(y_true, logit_arr)
    metrics["aff_mae"] = float(np.mean(np.abs(c_aff - aff_t)))
    metrics["style_mae"] = float(np.mean(np.abs(c_style - style_t)))
    if cfg is not None:
        if not (cfg.USE_AFF_CONCEPT_BRANCH and cfg.USE_AFF_CONCEPT_SUPERVISION):
            metrics["aff_mae"] = float("nan")
        if not cfg.USE_STYLE_BRANCH:
            metrics["style_mae"] = float("nan")

    return {
        "metrics": metrics,
        "y_true": y_true,
        "logits": logit_arr,
        "c_aff": c_aff,
        "c_style": c_style,
        "c_aff_rep": c_aff_rep,
        "c_style_rep": c_style_rep,
        "aff_targets": aff_t,
        "style_targets": style_t,
    }


def _feature_matrix_for_df(df: pd.DataFrame, cache: Dict[str, Dict[str, np.ndarray]]) -> np.ndarray:
    return np.stack([cache[str(p)]["concept_features"] for p in df["path"].astype(str).tolist()], axis=0).astype(np.float32)


def _feature_names_from_cache(cache: Dict[str, Dict[str, np.ndarray]]) -> List[str]:
    arr = cache["__concept_feature_names__"]["names"]
    return [str(x) for x in arr.tolist()]


def make_loaders_for_fold(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cache: Dict[str, Dict[str, np.ndarray]],
    cfg: Config,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, int], Dict[str, np.ndarray]]:
    feature_names = _feature_names_from_cache(cache)

    train_feats = _feature_matrix_for_df(train_df, cache)
    val_feats = _feature_matrix_for_df(val_df, cache)
    test_feats = _feature_matrix_for_df(test_df, cache)

    scaler = RobustScaler()
    scaler.fit(train_feats)

    train_z = np.clip(scaler.transform(train_feats), -5.0, 5.0).astype(np.float32)
    val_z = np.clip(scaler.transform(val_feats), -5.0, 5.0).astype(np.float32)
    test_z = np.clip(scaler.transform(test_feats), -5.0, 5.0).astype(np.float32)

    train_speakers_arr = train_df["speaker"].astype(str).to_numpy(dtype=str)
    val_speakers_arr = val_df["speaker"].astype(str).to_numpy(dtype=str)
    test_speakers_arr = test_df["speaker"].astype(str).to_numpy(dtype=str)

    train_baselines = compute_speaker_baselines(train_z, train_speakers_arr)
    train_global_baseline = np.mean(train_z, axis=0).astype(np.float32)
    train_baseline_mat = baseline_matrix_for_rows(
        train_z,
        train_speakers_arr,
        baseline_by_speaker=train_baselines,
        fallback_global=train_global_baseline,
    )

    if cfg.DIAGNOSTIC_LOCAL_BASELINES_FOR_VAL_TEST:
        val_baselines = compute_speaker_baselines(val_z, val_speakers_arr)
        test_baselines = compute_speaker_baselines(test_z, test_speakers_arr)
    else:
        val_baselines = train_baselines
        test_baselines = train_baselines

    val_baseline_mat = baseline_matrix_for_rows(
        val_z,
        val_speakers_arr,
        baseline_by_speaker=val_baselines,
        fallback_global=train_global_baseline,
    )
    test_baseline_mat = baseline_matrix_for_rows(
        test_z,
        test_speakers_arr,
        baseline_by_speaker=test_baselines,
        fallback_global=train_global_baseline,
    )

    train_aff, train_style = build_concepts_from_baseline_and_deviation(train_z, train_baseline_mat, feature_names)
    val_aff, val_style = build_concepts_from_baseline_and_deviation(val_z, val_baseline_mat, feature_names)
    test_aff, test_style = build_concepts_from_baseline_and_deviation(test_z, test_baseline_mat, feature_names)

    train_speakers = sorted(train_df["speaker"].astype(str).unique().tolist())
    speaker_to_local = {spk: i for i, spk in enumerate(train_speakers)}

    train_ds = CREMADConceptDataset(train_df, cache, train_aff, train_style, speaker_to_local=speaker_to_local)
    val_ds = CREMADConceptDataset(val_df, cache, val_aff, val_style, speaker_to_local=speaker_to_local)
    test_ds = CREMADConceptDataset(test_df, cache, test_aff, test_style, speaker_to_local=speaker_to_local)

    device_type = str(cfg.DEVICE).lower()
    pin_memory = device_type.startswith("cuda")
    persistent_workers = cfg.NUM_WORKERS > 0

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    concept_arrays = {
        "train_aff": train_aff,
        "train_style": train_style,
        "val_aff": val_aff,
        "val_style": val_style,
        "test_aff": test_aff,
        "test_style": test_style,
        "train_z": train_z,
        "val_z": val_z,
        "test_z": test_z,
    }

    return train_loader, val_loader, test_loader, speaker_to_local, concept_arrays


def fit_logistic_regression_probe(X: np.ndarray, y: np.ndarray, seed: int) -> Optional[LogisticRegression]:
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y)

    if X.ndim != 2 or len(X) != len(y) or len(np.unique(y)) < 2:
        return None

    try:
        clf = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            solver="lbfgs",
            random_state=seed,
        )
        clf.fit(X, y)
        return clf
    except Exception:
        return None


def speaker_probe_accuracy(
    concepts: np.ndarray,
    speaker_labels: np.ndarray,
    seed: int,
    test_size: float = 0.30,
) -> float:
    audit = speaker_leakage_audit(
        concepts=concepts,
        speaker_labels=speaker_labels,
        seed=seed,
        test_size=test_size,
        n_repeats=1,
    )
    return audit["probe_acc_mean"]


def speaker_leakage_audit(
    concepts: np.ndarray,
    speaker_labels: np.ndarray,
    seed: int,
    test_size: float = 0.30,
    n_repeats: int = 5,
) -> Dict[str, float]:
    X_all = np.asarray(concepts, dtype=np.float32)
    speaker_labels = np.asarray(speaker_labels, dtype=str)

    unique, counts = np.unique(speaker_labels, return_counts=True)
    keep_speakers = unique[counts >= 3]
    keep = np.isin(speaker_labels, keep_speakers)
    X = X_all[keep]
    y = speaker_labels[keep]

    unique_kept, kept_counts = np.unique(y, return_counts=True)
    n_classes = int(len(unique_kept))
    n_samples = int(len(y))
    chance_uniform = float(1.0 / n_classes) if n_classes > 0 else float("nan")
    chance_majority = float(np.max(kept_counts) / n_samples) if n_samples > 0 else float("nan")

    empty = {
        "probe_acc_mean": float("nan"),
        "probe_acc_std": float("nan"),
        "probe_chance_uniform": chance_uniform,
        "probe_chance_majority": chance_majority,
        "probe_leakage_index": float("nan"),
        "probe_n_speakers": float(n_classes),
        "probe_n_samples": float(n_samples),
    }

    if n_classes < 2 or n_samples < 20:
        return empty

    splitter = StratifiedShuffleSplit(
        n_splits=max(int(n_repeats), 1),
        test_size=test_size,
        random_state=seed,
    )
    accs = []
    try:
        splits = list(splitter.split(X, y))
    except ValueError:
        return empty

    for split_idx, (tr, te) in enumerate(splits):
        clf = fit_logistic_regression_probe(X[tr], y[tr], seed=seed + split_idx)
        if clf is None:
            continue
        pred = clf.predict(X[te])
        accs.append(float(accuracy_score(y[te], pred)))

    if not accs:
        return empty

    acc_mean = float(np.mean(accs))
    acc_std = float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0
    denom = max(1.0 - chance_uniform, 1e-8) if np.isfinite(chance_uniform) else float("nan")
    leakage_index = float((acc_mean - chance_uniform) / denom) if np.isfinite(denom) else float("nan")
    return {
        "probe_acc_mean": acc_mean,
        "probe_acc_std": acc_std,
        "probe_chance_uniform": chance_uniform,
        "probe_chance_majority": chance_majority,
        "probe_leakage_index": leakage_index,
        "probe_n_speakers": float(n_classes),
        "probe_n_samples": float(n_samples),
    }


def prefix_metrics(prefix: str, metrics: Dict[str, float]) -> Dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def _mean_metric(df: pd.DataFrame, col: str) -> float:
    if col not in df.columns:
        return float("nan")
    return float(pd.to_numeric(df[col], errors="coerce").mean(skipna=True))


def _format_metric(value: float, digits: int = 4) -> str:
    if not np.isfinite(value):
        return "nan"
    return f"{value:.{digits}f}"


def dataframe_to_markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._\n"

    headers = [str(c) for c in df.columns]
    body = []
    for _, row in df.iterrows():
        vals = []
        for c in df.columns:
            val = row[c]
            if isinstance(val, (float, np.floating)):
                vals.append(_format_metric(float(val)))
            elif pd.isna(val):
                vals.append("NA")
            else:
                vals.append(str(val))
        body.append(vals)

    widths = [max(len(headers[i]), *(len(row[i]) for row in body)) for i in range(len(headers))]

    def fmt(values: List[str]) -> str:
        return "| " + " | ".join(values[i].ljust(widths[i]) for i in range(len(values))) + " |"

    sep = "| " + " | ".join("-" * w for w in widths) + " |"
    return "\n".join([fmt(headers), sep] + [fmt(row) for row in body]) + "\n"


def write_dataset_statistics(df: pd.DataFrame, cfg: Config) -> Tuple[str, str]:
    dataset_name = normalize_dataset_name(cfg.DATASET)
    emotion_names = emotion_names_for_dataset(dataset_name)
    class_counts = (
        df["emotion"]
        .astype(int)
        .value_counts()
        .reindex(range(len(emotion_names)), fill_value=0)
    )

    row: Dict[str, Any] = {
        "dataset": dataset_display_name(dataset_name),
        "utterances": int(len(df)),
        "speakers": int(df["speaker"].astype(str).nunique()),
    }
    for idx, name in enumerate(emotion_names):
        row[f"{name}_count"] = int(class_counts.loc[idx])

    stats = pd.DataFrame([row])
    csv_path = os.path.join(cfg.OUT_DIR, "dataset_statistics.csv")
    md_path = os.path.join(cfg.OUT_DIR, "dataset_statistics.md")
    stats.to_csv(csv_path, index=False)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Dataset Statistics\n\n")
        f.write(dataframe_to_markdown_table(stats))

    return csv_path, md_path


def _speaker_summary_value(
    speaker_summary: Optional[pd.DataFrame],
    representation: str,
    col: str,
) -> float:
    if speaker_summary is None or speaker_summary.empty or col not in speaker_summary.columns:
        return float("nan")
    mask = speaker_summary["representation"].astype(str) == representation
    if not mask.any():
        return float("nan")
    return float(pd.to_numeric(speaker_summary.loc[mask, col], errors="coerce").mean(skipna=True))


def write_main_claim_summary(
    results: pd.DataFrame,
    cfg: Config,
    speaker_summary: Optional[pd.DataFrame] = None,
) -> str:
    path = os.path.join(cfg.OUT_DIR, "main_claim_summary.txt")
    aff_rep = "Affective concepts c_aff"
    style_rep = "Style concepts c_style"
    aff_uniform = _mean_metric(results, "speaker_probe_aff_probe_chance_uniform")
    aff_majority = _mean_metric(results, "speaker_probe_aff_probe_chance_majority")
    style_uniform = _mean_metric(results, "speaker_probe_style_probe_chance_uniform")
    style_majority = _mean_metric(results, "speaker_probe_style_probe_chance_majority")

    if not np.isfinite(aff_uniform):
        aff_uniform = _speaker_summary_value(speaker_summary, aff_rep, "uniform_chance")
    if not np.isfinite(aff_majority):
        aff_majority = _speaker_summary_value(speaker_summary, aff_rep, "majority_chance")
    if not np.isfinite(style_uniform):
        style_uniform = _speaker_summary_value(speaker_summary, style_rep, "uniform_chance")
    if not np.isfinite(style_majority):
        style_majority = _speaker_summary_value(speaker_summary, style_rep, "majority_chance")

    lines = [
        f"UAR = {_format_metric(_mean_metric(results, 'test_uar'))}",
        f"Macro-F1 = {_format_metric(_mean_metric(results, 'test_macro_f1'))}",
        f"speaker_probe_aff = {_format_metric(_mean_metric(results, 'speaker_probe_aff_acc'))}",
        f"speaker_probe_aff_uniform_chance = {_format_metric(aff_uniform)}",
        f"speaker_probe_aff_majority_chance = {_format_metric(aff_majority)}",
        f"speaker_probe_style = {_format_metric(_mean_metric(results, 'speaker_probe_style_acc'))}",
        f"speaker_probe_style_uniform_chance = {_format_metric(style_uniform)}",
        f"speaker_probe_style_majority_chance = {_format_metric(style_majority)}",
        f"style_swap_consistency = {_format_metric(_mean_metric(results, 'style_swap_consistency'))}",
        f"aff_swap_sensitivity = {_format_metric(_mean_metric(results, 'aff_swap_sensitivity'))}",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return path


def _concept_cols(df: pd.DataFrame, prefix: str) -> List[str]:
    return [c for c in df.columns if c.startswith(prefix)]


def _speaker_probe_summary_from_predictions(pred_all: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    if "fold" not in pred_all.columns or "speaker" not in pred_all.columns:
        return pd.DataFrame()

    specs = [
        ("Affective concepts c_aff", _concept_cols(pred_all, "c_aff_")),
        ("Style concepts c_style", _concept_cols(pred_all, "c_style_")),
    ]
    rows = []
    for representation, cols in specs:
        if not cols:
            continue
        fold_audits = []
        for fold in sorted(pred_all["fold"].dropna().unique().tolist()):
            fold_df = pred_all[pred_all["fold"] == fold]
            audit = speaker_leakage_audit(
                fold_df[cols].to_numpy(dtype=np.float32),
                fold_df["speaker"].astype(str).to_numpy(dtype=str),
                seed=cfg.SEED + int(fold),
                test_size=cfg.SPEAKER_PROBE_TEST_SIZE,
                n_repeats=cfg.SPEAKER_PROBE_REPEATS,
            )
            fold_audits.append(audit)

        if not fold_audits:
            continue

        def mean_key(key: str) -> float:
            vals = np.array([a[key] for a in fold_audits], dtype=np.float32)
            return float(np.nanmean(vals))

        def std_key(key: str) -> float:
            vals = np.array([a[key] for a in fold_audits], dtype=np.float32)
            return float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else 0.0

        rows.append({
            "representation": representation,
            "speaker_probe_accuracy": mean_key("probe_acc_mean"),
            "speaker_probe_accuracy_std": std_key("probe_acc_mean"),
            "uniform_chance": mean_key("probe_chance_uniform"),
            "majority_chance": mean_key("probe_chance_majority"),
            "leakage_index": mean_key("probe_leakage_index"),
            "n_speakers": mean_key("probe_n_speakers"),
        })

    return pd.DataFrame(rows)


def build_speaker_probe_chance_summary(
    results: pd.DataFrame,
    cfg: Config,
    pred_all: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    rows = []
    specs = [
        ("Affective concepts c_aff", "speaker_probe_aff"),
        ("Style concepts c_style", "speaker_probe_style"),
    ]
    for representation, prefix in specs:
        rows.append({
            "representation": representation,
            "speaker_probe_accuracy": _mean_metric(results, f"{prefix}_acc"),
            "speaker_probe_accuracy_std": _mean_metric(results, f"{prefix}_probe_acc_std"),
            "uniform_chance": _mean_metric(results, f"{prefix}_probe_chance_uniform"),
            "majority_chance": _mean_metric(results, f"{prefix}_probe_chance_majority"),
            "leakage_index": _mean_metric(results, f"{prefix}_probe_leakage_index"),
            "n_speakers": _mean_metric(results, f"{prefix}_probe_n_speakers"),
        })

    summary = pd.DataFrame(rows)
    if pred_all is not None:
        needs_prediction_fallback = (
            summary.empty
            or "uniform_chance" not in summary.columns
            or not np.isfinite(pd.to_numeric(summary["uniform_chance"], errors="coerce")).any()
        )
        if needs_prediction_fallback:
            prediction_summary = _speaker_probe_summary_from_predictions(pred_all, cfg)
            if not prediction_summary.empty:
                summary = prediction_summary
                overrides = {
                    "Affective concepts c_aff": "speaker_probe_aff",
                    "Style concepts c_style": "speaker_probe_style",
                }
                for representation, prefix in overrides.items():
                    acc = _mean_metric(results, f"{prefix}_acc")
                    if not np.isfinite(acc) or "representation" not in summary.columns:
                        continue
                    mask = summary["representation"].astype(str) == representation
                    if mask.any():
                        summary.loc[mask, "speaker_probe_accuracy"] = acc
    return summary


def write_speaker_probe_chance_summary(
    results: pd.DataFrame,
    cfg: Config,
    pred_all: Optional[pd.DataFrame] = None,
) -> Tuple[str, str, pd.DataFrame]:
    summary = build_speaker_probe_chance_summary(results, cfg, pred_all=pred_all)
    csv_path = os.path.join(cfg.OUT_DIR, "speaker_probe_chance_summary.csv")
    md_path = os.path.join(cfg.OUT_DIR, "speaker_probe_chance_summary.md")
    summary.to_csv(csv_path, index=False)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Speaker Probe Chance Baselines\n\n")
        f.write(dataframe_to_markdown_table(summary))

    return csv_path, md_path, summary


def write_concept_mae_by_concept(pred_all: pd.DataFrame, cfg: Config) -> str:
    rows = []
    specs = [
        (
            "affective",
            bool(cfg.USE_AFF_CONCEPT_BRANCH and cfg.USE_AFF_CONCEPT_SUPERVISION),
            AFF_CONCEPT_NAMES,
            "c_aff_",
            "target_aff_",
        ),
        (
            "style",
            bool(cfg.USE_STYLE_BRANCH),
            STYLE_CONCEPT_NAMES,
            "c_style_",
            "target_style_",
        ),
    ]

    for branch, branch_active, names, pred_prefix, target_prefix in specs:
        for concept in names:
            pred_col = f"{pred_prefix}{concept}"
            target_col = f"{target_prefix}{concept}"
            if pred_col not in pred_all.columns or target_col not in pred_all.columns:
                continue

            pred = pd.to_numeric(pred_all[pred_col], errors="coerce")
            target = pd.to_numeric(pred_all[target_col], errors="coerce")
            err = (pred - target).abs()
            valid = err.notna()

            rows.append({
                "branch": branch,
                "concept": concept,
                "branch_active": branch_active,
                "mae": float(err[valid].mean()) if branch_active and valid.any() else float("nan"),
                "abs_error_std": float(err[valid].std()) if branch_active and valid.any() else float("nan"),
                "pred_mean": float(pred[valid].mean()) if valid.any() else float("nan"),
                "target_mean": float(target[valid].mean()) if valid.any() else float("nan"),
                "n": int(valid.sum()),
            })

    path = os.path.join(cfg.OUT_DIR, "concept_mae_by_concept.csv")
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _pdf_escape(text: str) -> str:
    return str(text).replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _pdf_add_text(ops: List[str], x: float, y: float, text: str, size: float = 10.0, bold: bool = False) -> None:
    font = "F2" if bold else "F1"
    ops.append(f"BT /{font} {size:.1f} Tf {x:.1f} {y:.1f} Td ({_pdf_escape(text)}) Tj ET")


def _pdf_add_rect(ops: List[str], x: float, y: float, w: float, h: float, rgb: Tuple[float, float, float]) -> None:
    r, g, b = rgb
    ops.append(f"q {r:.3f} {g:.3f} {b:.3f} rg {x:.1f} {y:.1f} {w:.1f} {h:.1f} re f Q")


def _pdf_add_line(ops: List[str], x1: float, y1: float, x2: float, y2: float, gray: float = 0.75, width: float = 0.8) -> None:
    ops.append(f"q {gray:.3f} {gray:.3f} {gray:.3f} RG {width:.2f} w {x1:.1f} {y1:.1f} m {x2:.1f} {y2:.1f} l S Q")


def _write_simple_pdf(path: str, width: int, height: int, ops: List[str]) -> None:
    content = ("\n".join(ops) + "\n").encode("latin-1", errors="replace")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {width} {height}] "
            f"/Resources << /Font << /F1 4 0 R /F2 5 0 R >> >> /Contents 6 0 R >>"
        ).encode("ascii"),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>",
        b"<< /Length " + str(len(content)).encode("ascii") + b" >>\nstream\n" + content + b"endstream",
    ]

    pdf = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for idx, body in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf.extend(f"{idx} 0 obj\n".encode("ascii"))
        pdf.extend(body)
        pdf.extend(b"\nendobj\n")

    xref_pos = len(pdf)
    pdf.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    pdf.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        pdf.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    pdf.extend(
        (
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_pos}\n%%EOF\n"
        ).encode("ascii")
    )

    with open(path, "wb") as f:
        f.write(pdf)


def write_factorization_audit_figure(results: pd.DataFrame, cfg: Config) -> str:
    path = os.path.join(cfg.OUT_DIR, "factorization_audit_figure.pdf")
    width, height = 612, 420
    ops: List[str] = []

    _pdf_add_text(ops, 44, 388, "Factorization audit", size=16, bold=True)
    _pdf_add_text(
        ops,
        44,
        369,
        "Speaker separability and concept intervention diagnostics averaged across folds.",
        size=9,
    )

    panels = [
        (
            54,
            "Speaker probe accuracy",
            [
                ("affective", _mean_metric(results, "speaker_probe_aff_acc"), (0.180, 0.360, 0.720)),
                ("style", _mean_metric(results, "speaker_probe_style_acc"), (0.760, 0.300, 0.240)),
            ],
        ),
        (
            330,
            "Swap diagnostics",
            [
                ("style-swap", _mean_metric(results, "style_swap_consistency"), (0.200, 0.560, 0.360)),
                ("aff-swap", _mean_metric(results, "aff_swap_sensitivity"), (0.860, 0.570, 0.170)),
            ],
        ),
    ]

    base_y = 112.0
    chart_h = 210.0
    chart_w = 220.0
    bar_w = 55.0

    for panel_x, title, bars in panels:
        _pdf_add_text(ops, panel_x, 340, title, size=11, bold=True)
        _pdf_add_line(ops, panel_x, base_y, panel_x + chart_w, base_y, gray=0.15, width=1.0)
        _pdf_add_line(ops, panel_x, base_y, panel_x, base_y + chart_h, gray=0.15, width=1.0)
        for tick in [0.0, 0.5, 1.0]:
            y = base_y + tick * chart_h
            _pdf_add_line(ops, panel_x, y, panel_x + chart_w, y, gray=0.88, width=0.5)
            _pdf_add_text(ops, panel_x - 23, y - 3, f"{tick:.1f}", size=8)

        for idx, (label, value, color) in enumerate(bars):
            value_for_bar = value if np.isfinite(value) else 0.0
            value_for_bar = float(np.clip(value_for_bar, 0.0, 1.0))
            bar_h = value_for_bar * chart_h
            x = panel_x + 45 + idx * 92
            _pdf_add_rect(ops, x, base_y, bar_w, bar_h, color)
            _pdf_add_text(ops, x + 5, base_y + bar_h + 8, _format_metric(value), size=9, bold=True)
            _pdf_add_text(ops, x - 2, 88, label, size=9)

    _pdf_add_text(ops, 44, 42, "Expected pattern: lower affective speaker probe, higher style speaker probe, stable style-swap,", size=8)
    _pdf_add_text(ops, 44, 29, "and high affect-swap sensitivity.", size=8)

    _write_simple_pdf(path, width, height, ops)
    return path


@torch.no_grad()
def collect_concepts_for_diagnostics(model: nn.Module, loader: DataLoader, device: str) -> Dict[str, np.ndarray]:
    model.eval()
    ys = []
    c_affs = []
    c_styles = []

    for batch in loader:
        x = batch["x"].to(device)
        out = model(x, grl_lambda=0.0)
        ys.append(batch["y"].detach().cpu().numpy())
        c_affs.append(out["c_aff_rep"].detach().cpu().numpy())
        c_styles.append(out["c_style_rep"].detach().cpu().numpy())

    return {
        "y": np.concatenate(ys).astype(np.int64),
        "c_aff": np.concatenate(c_affs).astype(np.float32),
        "c_style": np.concatenate(c_styles).astype(np.float32),
    }


def _safe_probe_metrics(clf: Optional[LogisticRegression], X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    if clf is None:
        return {"acc": float("nan"), "uar": float("nan"), "macro_f1": float("nan")}
    try:
        pred = clf.predict(np.asarray(X, dtype=np.float32))
        return {
            "acc": float(accuracy_score(y, pred)),
            "uar": float(balanced_accuracy_score(y, pred)),
            "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        }
    except Exception:
        return {"acc": float("nan"), "uar": float("nan"), "macro_f1": float("nan")}


def _nontrivial_permutation(n: int, rng: np.random.Generator) -> np.ndarray:
    if n <= 1:
        return np.arange(n)
    perm = rng.permutation(n)
    if np.all(perm == np.arange(n)):
        perm = np.roll(perm, 1)
    return perm


def concept_intervention_diagnostics(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: str,
    seed: int,
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)

    train = collect_concepts_for_diagnostics(model, train_loader, device)
    test = collect_concepts_for_diagnostics(model, test_loader, device)

    Xtr_aff = train["c_aff"]
    Xte_aff = test["c_aff"]
    Xtr_style = train["c_style"]
    Xte_style = test["c_style"]
    Xtr_both = np.concatenate([Xtr_aff, Xtr_style], axis=1)
    Xte_both = np.concatenate([Xte_aff, Xte_style], axis=1)
    ytr = train["y"]
    yte = test["y"]

    aff_probe = fit_logistic_regression_probe(Xtr_aff, ytr, seed=seed)
    style_probe = fit_logistic_regression_probe(Xtr_style, ytr, seed=seed + 17)
    both_probe = fit_logistic_regression_probe(Xtr_both, ytr, seed=seed + 31)

    aff_m = _safe_probe_metrics(aff_probe, Xte_aff, yte)
    style_m = _safe_probe_metrics(style_probe, Xte_style, yte)
    both_m = _safe_probe_metrics(both_probe, Xte_both, yte)

    out = {
        "emotion_probe_aff_acc": aff_m["acc"],
        "emotion_probe_aff_uar": aff_m["uar"],
        "emotion_probe_aff_macro_f1": aff_m["macro_f1"],
        "emotion_probe_style_acc": style_m["acc"],
        "emotion_probe_style_uar": style_m["uar"],
        "emotion_probe_style_macro_f1": style_m["macro_f1"],
        "emotion_probe_both_acc": both_m["acc"],
        "emotion_probe_both_uar": both_m["uar"],
        "emotion_probe_both_macro_f1": both_m["macro_f1"],
        "style_swap_consistency": float("nan"),
        "style_swap_prob_l1": float("nan"),
        "aff_swap_sensitivity": float("nan"),
        "aff_swap_prob_l1": float("nan"),
    }

    if both_probe is None or len(Xte_both) < 2:
        return out

    try:
        base_pred = both_probe.predict(Xte_both)
        base_prob = both_probe.predict_proba(Xte_both)

        perm_style = _nontrivial_permutation(len(Xte_style), rng)
        Xte_style_swapped = np.concatenate([Xte_aff, Xte_style[perm_style]], axis=1)
        style_swap_pred = both_probe.predict(Xte_style_swapped)
        style_swap_prob = both_probe.predict_proba(Xte_style_swapped)

        perm_aff = _nontrivial_permutation(len(Xte_aff), rng)
        Xte_aff_swapped = np.concatenate([Xte_aff[perm_aff], Xte_style], axis=1)
        aff_swap_pred = both_probe.predict(Xte_aff_swapped)
        aff_swap_prob = both_probe.predict_proba(Xte_aff_swapped)

        out["style_swap_consistency"] = float(np.mean(base_pred == style_swap_pred))
        out["style_swap_prob_l1"] = float(np.mean(np.sum(np.abs(base_prob - style_swap_prob), axis=1)))
        out["aff_swap_sensitivity"] = float(1.0 - np.mean(base_pred == aff_swap_pred))
        out["aff_swap_prob_l1"] = float(np.mean(np.sum(np.abs(base_prob - aff_swap_prob), axis=1)))
    except Exception:
        pass

    return out


def run_experiment(cfg: Config) -> None:
    dataset_name = normalize_dataset_name(cfg.DATASET)
    cfg.DATASET = dataset_name
    emotion_names = emotion_names_for_dataset(dataset_name)
    n_emotions = len(emotion_names)
    data_dir = dataset_dir_for_config(cfg)

    set_seed(cfg.SEED)
    device = select_device(cfg.DEVICE)
    configure_torch_runtime(device)
    cfg.DEVICE = str(device)

    cfg.NUM_WORKERS = resolve_num_workers(cfg, device)
    try:
        torch.set_num_threads(resolve_num_threads(getattr(cfg, "NUM_THREADS", -1)))
    except Exception:
        pass

    os.makedirs(cfg.OUT_DIR, exist_ok=True)

    amp_status = (
        f"on ({str(cfg.AMP_DTYPE)})" if (getattr(cfg, "USE_AMP", False) and device.type == "cuda") else "off"
    )

    print("=" * 100)
    print("Disentangled Affective--Style CBM with eGeMAPS baseline/deviation concepts")
    print("Device:", device)
    print("MPS available:", is_mps_available())
    print("CUDA available:", torch.cuda.is_available())
    if device.type == "cuda":
        try:
            print("CUDA device:", torch.cuda.get_device_name(0))
        except Exception:
            pass
    print("DataLoader workers:", cfg.NUM_WORKERS)
    print("Feature extraction jobs:", resolve_n_jobs(getattr(cfg, "FEATURE_EXTRACTION_JOBS", -1)))
    print("Mixed precision (AMP):", amp_status)
    print("Dataset:", dataset_display_name(dataset_name))
    print("Dataset dir:", data_dir)
    print("Feature backend:", cfg.FEATURE_BACKEND)
    if str(cfg.FEATURE_BACKEND).lower().strip() == "opensmile":
        print("openSMILE feature set:", cfg.OPENSMILE_FEATURE_SET)
    print("Emotion head input:", cfg.EMOTION_HEAD_INPUT)
    print("Affective concept branch:", cfg.USE_AFF_CONCEPT_BRANCH)
    print("Affective concept supervision:", cfg.USE_AFF_CONCEPT_SUPERVISION)
    print("Style branch:", cfg.USE_STYLE_BRANCH)
    print("Selection concept penalty:", cfg.SELECTION_CONCEPT_PENALTY)
    print("=" * 100)

    df = discover_dataset(cfg)
    print(f"Discovered {len(df)} clips from {df['speaker'].nunique()} speakers")
    print("Emotion counts:")
    print(df["emotion_code"].value_counts().sort_index())
    dataset_stats_csv, dataset_stats_md = write_dataset_statistics(df, cfg)

    cache = build_feature_cache(df, cfg)
    feature_names = _feature_names_from_cache(cache)
    write_feature_group_report(feature_names, cfg)

    groups = df["speaker"].astype(str).to_numpy(dtype=str)
    y_all = df["emotion"].to_numpy(dtype=np.int64)
    X_all = np.arange(len(df))

    outer = GroupKFold(n_splits=cfg.N_SPLITS)
    rows = []
    all_fold_predictions = []

    for fold, (trainval_idx, test_idx) in enumerate(outer.split(X_all, y_all, groups), start=1):
        print("\n" + "#" * 100)
        print(f"Fold {fold}/{cfg.N_SPLITS}")
        print("#" * 100)

        trainval_df = df.iloc[trainval_idx].reset_index(drop=True)
        test_df = df.iloc[test_idx].reset_index(drop=True)

        inner_groups = trainval_df["speaker"].astype(str).to_numpy(dtype=str)
        inner_y = trainval_df["emotion"].to_numpy(dtype=np.int64)
        inner_X = np.arange(len(trainval_df))
        gss = GroupShuffleSplit(n_splits=1, test_size=cfg.INNER_VAL_SIZE, random_state=cfg.SEED + fold)
        inner_train_idx, inner_val_idx = next(gss.split(inner_X, inner_y, inner_groups))

        train_df = trainval_df.iloc[inner_train_idx].reset_index(drop=True)
        val_df = trainval_df.iloc[inner_val_idx].reset_index(drop=True)

        print(f"Train clips={len(train_df)} | Val clips={len(val_df)} | Test clips={len(test_df)}")
        print(
            f"Train speakers={train_df['speaker'].nunique()} | "
            f"Val speakers={val_df['speaker'].nunique()} | Test speakers={test_df['speaker'].nunique()}"
        )

        train_loader, val_loader, test_loader, speaker_to_local, _ = make_loaders_for_fold(
            train_df, val_df, test_df, cache, cfg
        )
        n_train_speakers = len(speaker_to_local)

        model = DisentangledAffectiveStyleCBM(
            n_mels=cfg.N_MELS,
            h_dim=cfg.H_DIM,
            n_aff=cfg.N_AFF_CONCEPTS,
            n_style=cfg.N_STYLE_CONCEPTS,
            n_emotions=n_emotions,
            n_train_speakers=n_train_speakers,
            dropout=cfg.DROPOUT,
            emotion_head_input=cfg.EMOTION_HEAD_INPUT,
            use_aff_concept_branch=cfg.USE_AFF_CONCEPT_BRANCH,
            use_style_branch=cfg.USE_STYLE_BRANCH,
            use_concept_embeddings=cfg.USE_CONCEPT_EMBEDDINGS,
            concept_emb_dim=cfg.CONCEPT_EMB_DIM,
        ).to(cfg.DEVICE)

        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.NUM_EPOCHS)
        emotion_weights = class_weights_from_labels(train_df["emotion"].to_numpy(dtype=np.int64), n_emotions)
        scaler = make_grad_scaler(cfg, device)

        best_score = -1e9
        best_state = None
        best_epoch = 0
        bad_epochs = 0

        for epoch in range(1, cfg.NUM_EPOCHS + 1):
            grl_lambd = grl_schedule(epoch - 1, cfg.NUM_EPOCHS, cfg.GRL_MAX_LAMBDA, cfg.ADV_WARMUP_EPOCHS)
            tr_losses = train_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                device=cfg.DEVICE,
                cfg=cfg,
                emotion_weights=emotion_weights,
                grl_lambda=grl_lambd,
            )
            scheduler.step()

            val_out = evaluate_model(model, val_loader, cfg.DEVICE, cfg=cfg)
            val_metrics = val_out["metrics"]
            aff_penalty = val_metrics["aff_mae"] if np.isfinite(val_metrics["aff_mae"]) else 0.0

            score = (
                val_metrics["uar"]
                + 0.5 * val_metrics["macro_f1"]
                - float(cfg.SELECTION_CONCEPT_PENALTY) * aff_penalty
            )

            if score > best_score:
                best_score = score
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1

            if epoch == 1 or epoch % 5 == 0 or bad_epochs == 0:
                print(
                    f"[Fold {fold} | E{epoch:03d}] "
                    f"loss={tr_losses['loss']:.4f} "
                    f"val_UAR={val_metrics['uar']:.4f} "
                    f"val_F1={val_metrics['macro_f1']:.4f} "
                    f"val_ECE={val_metrics['ece']:.4f} "
                    f"aff_MAE={val_metrics['aff_mae']:.4f} "
                    f"style_MAE={val_metrics['style_mae']:.4f} "
                    f"grl={grl_lambd:.3f}"
                )

            if bad_epochs >= cfg.PATIENCE:
                print(f"Early stopping at epoch {epoch}; best epoch={best_epoch}")
                break

        if best_state is not None:
            model.load_state_dict(best_state)

        test_out = evaluate_model(model, test_loader, cfg.DEVICE, cfg=cfg)
        test_metrics = test_out["metrics"]

        test_speakers = test_df["speaker"].astype(str).to_numpy(dtype=str)
        audit_aff = speaker_leakage_audit(
            test_out["c_aff_rep"],
            test_speakers,
            seed=cfg.SEED + fold,
            test_size=cfg.SPEAKER_PROBE_TEST_SIZE,
            n_repeats=cfg.SPEAKER_PROBE_REPEATS,
        )
        audit_style = speaker_leakage_audit(
            test_out["c_style_rep"],
            test_speakers,
            seed=cfg.SEED + 100 + fold,
            test_size=cfg.SPEAKER_PROBE_TEST_SIZE,
            n_repeats=cfg.SPEAKER_PROBE_REPEATS,
        )
        probe_aff = audit_aff["probe_acc_mean"]
        probe_style = audit_style["probe_acc_mean"]

        intervention = concept_intervention_diagnostics(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            device=cfg.DEVICE,
            seed=cfg.SEED + fold,
        )

        row = {
            "fold": fold,
            "emotion_head_input": cfg.EMOTION_HEAD_INPUT,
            "use_aff_concept_branch": cfg.USE_AFF_CONCEPT_BRANCH,
            "use_aff_concept_supervision": cfg.USE_AFF_CONCEPT_SUPERVISION,
            "use_style_branch": cfg.USE_STYLE_BRANCH,
            "use_aff_speaker_adversary": cfg.USE_AFF_SPEAKER_ADVERSARY,
            "use_style_emotion_adversary": cfg.USE_STYLE_EMOTION_ADVERSARY,
            "use_orthogonality": cfg.USE_ORTHOGONALITY,
            "use_concept_embeddings": cfg.USE_CONCEPT_EMBEDDINGS,
            "concept_emb_dim": cfg.CONCEPT_EMB_DIM,
            "adv_warmup_epochs": cfg.ADV_WARMUP_EPOCHS,
            "best_epoch": best_epoch,
            "test_acc": test_metrics["acc"],
            "test_uar": test_metrics["uar"],
            "test_macro_f1": test_metrics["macro_f1"],
            "test_ece": test_metrics["ece"],
            "test_aff_mae": test_metrics["aff_mae"],
            "test_style_mae": test_metrics["style_mae"],
            "speaker_probe_aff_acc": probe_aff,
            "speaker_probe_style_acc": probe_style,
            **prefix_metrics("speaker_probe_aff", audit_aff),
            **prefix_metrics("speaker_probe_style", audit_style),
            "swap_consistency": intervention["style_swap_consistency"],
            "style_swap_consistency": intervention["style_swap_consistency"],
            "style_swap_prob_l1": intervention["style_swap_prob_l1"],
            "aff_swap_sensitivity": intervention["aff_swap_sensitivity"],
            "aff_swap_prob_l1": intervention["aff_swap_prob_l1"],
            "emotion_probe_aff_uar": intervention["emotion_probe_aff_uar"],
            "emotion_probe_style_uar": intervention["emotion_probe_style_uar"],
            "emotion_probe_both_uar": intervention["emotion_probe_both_uar"],
            "emotion_probe_aff_macro_f1": intervention["emotion_probe_aff_macro_f1"],
            "emotion_probe_style_macro_f1": intervention["emotion_probe_style_macro_f1"],
            "emotion_probe_both_macro_f1": intervention["emotion_probe_both_macro_f1"],
            "n_train": len(train_df),
            "n_val": len(val_df),
            "n_test": len(test_df),
            "n_train_speakers": train_df["speaker"].nunique(),
            "n_val_speakers": val_df["speaker"].nunique(),
            "n_test_speakers": test_df["speaker"].nunique(),
            "n_train_val_speaker_overlap": len(set(train_df["speaker"].astype(str)) & set(val_df["speaker"].astype(str))),
            "n_train_test_speaker_overlap": len(set(train_df["speaker"].astype(str)) & set(test_df["speaker"].astype(str))),
            "n_val_test_speaker_overlap": len(set(val_df["speaker"].astype(str)) & set(test_df["speaker"].astype(str))),
        }
        rows.append(row)

        print("\nFold test metrics:")
        for k, v in row.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")

        probs = torch.softmax(torch.tensor(test_out["logits"]), dim=1).numpy()
        pred = probs.argmax(axis=1)
        fold_pred_df = test_df.copy()
        fold_pred_df["fold"] = fold
        fold_pred_df["y_true"] = test_out["y_true"]
        fold_pred_df["y_pred"] = pred
        for j, name in enumerate(emotion_names):
            fold_pred_df[f"prob_{name}"] = probs[:, j]
        for j, name in enumerate(AFF_CONCEPT_NAMES):
            fold_pred_df[f"c_aff_{name}"] = test_out["c_aff"][:, j]
            fold_pred_df[f"target_aff_{name}"] = test_out["aff_targets"][:, j]
        for j, name in enumerate(STYLE_CONCEPT_NAMES):
            fold_pred_df[f"c_style_{name}"] = test_out["c_style"][:, j]
            fold_pred_df[f"target_style_{name}"] = test_out["style_targets"][:, j]
        all_fold_predictions.append(fold_pred_df)
        clear_device_cache(torch.device(cfg.DEVICE))

    results = pd.DataFrame(rows)
    pred_all = pd.concat(all_fold_predictions, axis=0).reset_index(drop=True)

    results_path = os.path.join(cfg.OUT_DIR, "fold_metrics.csv")
    pred_path = os.path.join(cfg.OUT_DIR, "test_predictions_and_concepts.csv")
    results.to_csv(results_path, index=False)
    pred_all.to_csv(pred_path, index=False)

    leakage_cols = [
        "fold",
        "emotion_head_input",
        "use_aff_concept_branch",
        "use_aff_concept_supervision",
        "use_style_branch",
        "use_aff_speaker_adversary",
        "use_style_emotion_adversary",
        "use_orthogonality",
        "n_train_val_speaker_overlap",
        "n_train_test_speaker_overlap",
        "n_val_test_speaker_overlap",
        "speaker_probe_aff_probe_acc_mean",
        "speaker_probe_aff_probe_acc_std",
        "speaker_probe_aff_probe_chance_uniform",
        "speaker_probe_aff_probe_chance_majority",
        "speaker_probe_aff_probe_leakage_index",
        "speaker_probe_aff_probe_n_speakers",
        "speaker_probe_aff_probe_n_samples",
        "speaker_probe_style_probe_acc_mean",
        "speaker_probe_style_probe_acc_std",
        "speaker_probe_style_probe_chance_uniform",
        "speaker_probe_style_probe_chance_majority",
        "speaker_probe_style_probe_leakage_index",
        "speaker_probe_style_probe_n_speakers",
        "speaker_probe_style_probe_n_samples",
    ]
    leakage_cols = [c for c in leakage_cols if c in results.columns]
    leakage_path = os.path.join(cfg.OUT_DIR, "speaker_leakage_audit.csv")
    results[leakage_cols].to_csv(leakage_path, index=False)
    speaker_chance_csv, speaker_chance_md, speaker_summary = write_speaker_probe_chance_summary(
        results,
        cfg,
        pred_all=pred_all,
    )
    claim_summary_path = write_main_claim_summary(results, cfg, speaker_summary=speaker_summary)
    concept_mae_path = write_concept_mae_by_concept(pred_all, cfg)
    factorization_figure_path = write_factorization_audit_figure(results, cfg)

    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    metric_cols = [
        "test_acc",
        "test_uar",
        "test_macro_f1",
        "test_ece",
        "test_aff_mae",
        "test_style_mae",
        "speaker_probe_aff_acc",
        "speaker_probe_style_acc",
        "speaker_probe_aff_probe_chance_uniform",
        "speaker_probe_aff_probe_chance_majority",
        "speaker_probe_aff_probe_leakage_index",
        "speaker_probe_style_probe_chance_uniform",
        "speaker_probe_style_probe_chance_majority",
        "speaker_probe_style_probe_leakage_index",
        "style_swap_consistency",
        "style_swap_prob_l1",
        "aff_swap_sensitivity",
        "aff_swap_prob_l1",
        "emotion_probe_aff_uar",
        "emotion_probe_style_uar",
        "emotion_probe_both_uar",
    ]
    for col in metric_cols:
        mean = results[col].mean(skipna=True)
        std = results[col].std(skipna=True)
        print(f"{col:28s}: {mean:.4f} +/- {std:.4f}")

    print("\nSaved:")
    print("  ", results_path)
    print("  ", pred_path)
    print("  ", dataset_stats_csv)
    print("  ", dataset_stats_md)
    print("  ", leakage_path)
    print("  ", claim_summary_path)
    print("  ", speaker_chance_csv)
    print("  ", speaker_chance_md)
    print("  ", concept_mae_path)
    print("  ", factorization_figure_path)

    cm = confusion_matrix(pred_all["y_true"], pred_all["y_pred"], labels=list(range(n_emotions)))
    cm_df = pd.DataFrame(cm, index=emotion_names, columns=emotion_names)
    cm_path = os.path.join(cfg.OUT_DIR, "confusion_matrix.csv")
    cm_df.to_csv(cm_path)
    print("  ", cm_path)


if __name__ == "__main__":
    run_experiment(CFG)
