"""Load the four precomputed DE-LDS datasets used in the paper."""

from __future__ import annotations

import os
import pickle
from pathlib import Path

import numpy as np
import scipy.io as sio

SEED_IV_LABELS = {
    1: [1, 2, 3, 0, 2, 0, 0, 1, 0, 1, 2, 1, 1, 1, 2, 3, 2, 2, 3, 3, 0, 3, 0, 3],
    2: [2, 1, 3, 0, 0, 2, 0, 2, 3, 3, 2, 3, 2, 0, 1, 1, 2, 1, 0, 3, 0, 1, 3, 1],
    3: [1, 2, 2, 1, 3, 3, 3, 1, 1, 2, 1, 0, 2, 3, 3, 0, 2, 3, 0, 0, 2, 0, 1, 0],
}


SEED_VII_CLASSES = ["Anger", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]


SEED_VII_LABELS = [
    "Happy",
    "Neutral",
    "Disgust",
    "Sad",
    "Anger",
    "Anger",
    "Sad",
    "Disgust",
    "Neutral",
    "Happy",
    "Happy",
    "Neutral",
    "Disgust",
    "Sad",
    "Anger",
    "Anger",
    "Sad",
    "Disgust",
    "Neutral",
    "Happy",
    "Anger",
    "Sad",
    "Fear",
    "Neutral",
    "Surprise",
    "Surprise",
    "Neutral",
    "Fear",
    "Sad",
    "Anger",
    "Anger",
    "Sad",
    "Fear",
    "Neutral",
    "Surprise",
    "Surprise",
    "Neutral",
    "Fear",
    "Sad",
    "Anger",
    "Happy",
    "Surprise",
    "Disgust",
    "Fear",
    "Anger",
    "Anger",
    "Fear",
    "Disgust",
    "Surprise",
    "Happy",
    "Happy",
    "Surprise",
    "Disgust",
    "Fear",
    "Anger",
    "Anger",
    "Fear",
    "Disgust",
    "Surprise",
    "Happy",
    "Disgust",
    "Sad",
    "Fear",
    "Surprise",
    "Happy",
    "Happy",
    "Surprise",
    "Fear",
    "Sad",
    "Disgust",
    "Disgust",
    "Sad",
    "Fear",
    "Surprise",
    "Happy",
    "Happy",
    "Surprise",
    "Fear",
    "Sad",
    "Disgust",
]


def _as_time_nodes_bands(data: np.ndarray) -> np.ndarray:
    if data.ndim != 3:
        raise ValueError(f"expected a 3-D DE-LDS trial, got {data.shape}")
    if data.shape[0] == 62 and data.shape[2] == 5:
        data = np.transpose(data, (1, 0, 2))
    if data.shape[1:] != (62, 5):
        raise ValueError(f"expected [T,62,5] or [62,T,5], got {data.shape}")
    return np.asarray(data, dtype=np.float32)


def _pad_trials(trials: list[np.ndarray]) -> np.ndarray:
    if not trials:
        raise ValueError("no DE-LDS trials were loaded")
    max_steps = max(trial.shape[0] for trial in trials)
    padded = np.zeros((len(trials), max_steps, 62, 5), dtype=np.float32)
    for index, trial in enumerate(trials):
        padded[index, : trial.shape[0]] = trial
    return padded


def _load_seediv_session(session_dir: str, session_id: int):
    trials, labels, subjects = [], [], []
    filenames = sorted(
        (name for name in os.listdir(session_dir) if name.endswith(".mat")),
        key=lambda name: int(Path(name).stem.split("_")[0]),
    )
    for filename in filenames:
        subject = int(Path(filename).stem.split("_")[0])
        contents = sio.loadmat(os.path.join(session_dir, filename))
        keys = sorted(
            (key for key in contents if key.startswith("de_LDS")),
            key=lambda key: int(key[len("de_LDS") :]),
        )
        if len(keys) != len(SEED_IV_LABELS[session_id]):
            raise ValueError(f"unexpected trial count in {filename}")
        for trial_index, key in enumerate(keys):
            trials.append(_as_time_nodes_bands(contents[key]))
            labels.append(SEED_IV_LABELS[session_id][trial_index])
            subjects.append(f"P{subject}")
    return trials, labels, subjects


def load_seediv(root: str, session_ids: tuple[int, ...] = (1, 2, 3)):
    trials, labels, subjects = [], [], []
    for session_id in session_ids:
        session_dir = os.path.join(root, str(session_id))
        if not os.path.isdir(session_dir):
            raise FileNotFoundError(session_dir)
        session_trials, session_labels, session_subjects = _load_seediv_session(
            session_dir, session_id
        )
        trials.extend(session_trials)
        labels.extend(session_labels)
        subjects.extend(session_subjects)
    return _pad_trials(trials), np.asarray(labels, dtype=np.int64), subjects, 4


def load_seed(root: str, session_ids: tuple[int, ...] = (1, 2, 3)):
    feature_dir = os.path.join(root, "ExtractedFeatures_1s")
    if not os.path.isdir(feature_dir):
        raise FileNotFoundError(feature_dir)
    filenames = [
        name
        for name in os.listdir(feature_dir)
        if name.endswith(".mat") and name != "label.mat"
    ]
    by_subject: dict[int, list[str]] = {}
    for filename in filenames:
        subject = int(filename.split("_")[0])
        by_subject.setdefault(subject, []).append(filename)
    raw_labels = (
        np.asarray(sio.loadmat(os.path.join(feature_dir, "label.mat"))["label"])
        .reshape(-1)
        .astype(int)
    )
    label_map = {-1: 0, 0: 1, 1: 2}
    labels_by_trial = np.asarray([label_map[int(value)] for value in raw_labels])
    trials, labels, subjects = [], [], []
    for subject, subject_files in sorted(by_subject.items()):
        subject_files.sort()
        for session_id in session_ids:
            if session_id < 1 or session_id > len(subject_files):
                raise ValueError(f"subject {subject} has no session {session_id}")
            filename = subject_files[session_id - 1]
            contents = sio.loadmat(os.path.join(feature_dir, filename))
            keys = sorted(
                (key for key in contents if key.startswith("de_LDS")),
                key=lambda key: int(key[len("de_LDS") :]),
            )
            if len(keys) != len(labels_by_trial):
                raise ValueError(f"unexpected trial count in {filename}")
            for trial_index, key in enumerate(keys):
                trials.append(_as_time_nodes_bands(contents[key]))
                labels.append(labels_by_trial[trial_index])
                subjects.append(f"P{subject}")
    return _pad_trials(trials), np.asarray(labels, dtype=np.int64), subjects, 3


def load_seedv(root: str):
    feature_dir = os.path.join(root, "EEG_DE_features")
    if not os.path.isdir(feature_dir):
        raise FileNotFoundError(feature_dir)
    filenames = sorted(
        (name for name in os.listdir(feature_dir) if name.endswith("_123.npz")),
        key=lambda name: int(name.split("_")[0]),
    )
    trials, labels, subjects = [], [], []
    for filename in filenames:
        subject = int(filename.split("_")[0])
        packed = np.load(os.path.join(feature_dir, filename), allow_pickle=False)
        data = pickle.loads(packed["data"].tobytes())
        label_data = pickle.loads(packed["label"].tobytes())
        for trial in sorted(data, key=int):
            values = np.asarray(data[trial], dtype=np.float32)
            if values.ndim != 2 or values.shape[1] != 62 * 5:
                raise ValueError(f"unexpected SEED-V trial shape {values.shape}")
            trials.append(values.reshape(values.shape[0], 62, 5))
            labels.append(int(np.asarray(label_data[trial]).reshape(-1)[0]))
            subjects.append(f"P{subject}")
    return _pad_trials(trials), np.asarray(labels, dtype=np.int64), subjects, 5


def load_seedvii(root: str):
    feature_dir = os.path.join(root, "EEG_features")
    if not os.path.isdir(feature_dir):
        raise FileNotFoundError(feature_dir)
    names = {name: index for index, name in enumerate(SEED_VII_CLASSES)}
    filenames = sorted(
        (name for name in os.listdir(feature_dir) if name.endswith(".mat")),
        key=lambda name: int(Path(name).stem),
    )
    trials, labels, subjects = [], [], []
    for filename in filenames:
        subject = int(Path(filename).stem)
        contents = sio.loadmat(os.path.join(feature_dir, filename))
        for trial_index, label in enumerate(SEED_VII_LABELS, start=1):
            key = f"de_LDS_{trial_index}"
            if key not in contents:
                raise ValueError(f"missing {key} in {filename}")
            data = np.asarray(contents[key], dtype=np.float32)
            if data.ndim != 3 or data.shape[1:] != (5, 62):
                raise ValueError(f"unexpected SEED-VII trial shape {data.shape}")
            trials.append(np.transpose(data, (0, 2, 1)))
            labels.append(names[label])
            subjects.append(f"P{subject}")
    return _pad_trials(trials), np.asarray(labels, dtype=np.int64), subjects, 7


def load_dataset(name: str, root: str, sessions: tuple[int, ...] = (1, 2, 3)):
    loaders = {
        "seed": lambda: load_seed(root, sessions),
        "seediv": lambda: load_seediv(root, sessions),
        "seedv": lambda: load_seedv(root),
        "seedvii": lambda: load_seedvii(root),
    }
    if name not in loaders:
        raise ValueError(f"unknown dataset {name!r}")
    return loaders[name]()
