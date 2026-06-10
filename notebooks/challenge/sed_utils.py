import os
import sys
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Callable
from sklearn.multioutput import MultiOutputClassifier

# make the official scoring helpers importable from the provided baseline folder
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "provided_baseline"))
from evaluate import (
    aggregate_ground_truth_annotations,
    build_segment_frame_from_intervals,
    calculate_f1_score,
)

# constants for feature extraction and class names
SEGMENT_LENGTH = 1.0  # each feature vector covers a 1-second window
HOP_SIZE       = 0.5  # segments are extracted with 50% overlap

FEATURE_NAMES = [
    "zcr_mean",        "zcr_std",        "zcr_min",        "zcr_max",
    "melspect_mean",   "melspect_std",   "melspect_min",   "melspect_max",
    "mfcc_mean",       "mfcc_std",       "mfcc_min",       "mfcc_max",
    "mfcc_d_mean",     "mfcc_d_std",     "mfcc_d_min",     "mfcc_d_max",
    "mfcc_d2_mean",    "mfcc_d2_std",    "mfcc_d2_min",    "mfcc_d2_max",
    "flux_mean",       "flux_std",       "flux_min",       "flux_max",
    "flatness_mean",   "flatness_std",   "flatness_min",   "flatness_max",
    "centroid_mean",   "centroid_std",   "centroid_min",   "centroid_max",
    "bandwidth_mean",  "bandwidth_std",  "bandwidth_min",  "bandwidth_max",
    "contrast_mean",   "contrast_std",   "contrast_min",   "contrast_max",
    "rolloff_low_mean",  "rolloff_low_std",  "rolloff_low_min",  "rolloff_low_max",
    "rolloff_high_mean", "rolloff_high_std", "rolloff_high_min", "rolloff_high_max",
    "energy_mean",     "energy_std",     "energy_min",     "energy_max",
    "power_mean",      "power_std",      "power_min",      "power_max",
]

CLASS_NAMES = [
    "bell_ringing",
    "coffee_machine",
    "cutlery_dishes",
    "door_open_close",
    "footsteps",
    "keyboard_typing",
    "keychain",
    "light_switch",
    "microwave",
    "phone_ringing",
    "running_water",
    "toilet_flushing",
    "vacuum_cleaner",
    "wardrobe_drawer_open_close",
    "window_open_close",
]

def build_feature_matrix(data: dict) -> np.ndarray:
    """Concatenate all named features from a loaded .npz dict into one matrix.

    Parameters
    ----------
    data : dict
        Contents of a .npz file (loaded with allow_pickle=True).

    Returns
    -------
    X : np.ndarray of shape (N_segments, D_features)
    """
    arrays = []
    for feat_name in FEATURE_NAMES:
        feat = data[feat_name]
        if feat.ndim == 1:
            feat = feat[:, np.newaxis]
        arrays.append(feat.astype(np.float32))
    return np.concatenate(arrays, axis=1)


def get_segment_labels(data: dict) -> np.ndarray:
    """Extract binary multilabel targets from the annotations field of a .npz file.

    The annotations array has shape (N_segments, N_classes, N_annotators).
    Each value is the fraction of the 1-second segment during which a given
    annotator marked the class as active.

    We binarize by checking for any overlap (> 0) and then apply majority vote
    across annotators to obtain the final binary label.

    Parameters
    ----------
    data : dict
        Contents of a .npz file (loaded with allow_pickle=True).
        Must contain the 'annotations' key.

    Returns
    -------
    Y : np.ndarray of shape (N_segments, N_classes), dtype int
    """
    annotations = data["annotations"]           # shape: (N, C, A)
    # Binarize: 1 if annotator marked ANY part of this segment as active.
    # (This threshold can be improved, e.g. > 0.5 for majority of the segment.)
    binary = (annotations > 0).astype(int)      # (N, C, A)
    # Majority vote: class is active if more than half the annotators agree.
    votes = binary.sum(axis=2)                  # (N, C)
    n_annotators = binary.shape[2]
    y = (votes > (n_annotators // 2)).astype(int)
    return y

def load_all_segments(
    file_list: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """Load features and labels from a list of .npz files and stack them.

    Parameters
    ----------
    file_list : list of str
        Paths to .npz audio feature files.

    Returns
    -------
    X : np.ndarray of shape (N_total_segments, D_features)
    Y : np.ndarray of shape (N_total_segments, N_classes)
    """
    X_list, Y_list = [], []
    for filepath in file_list:
        data = dict(np.load(filepath, allow_pickle=True))
        X_list.append(build_feature_matrix(data))
        Y_list.append(get_segment_labels(data))
    return np.vstack(X_list), np.vstack(Y_list)

def run_sed_inference(
    filepath: str,
    classifier: MultiOutputClassifier,
) -> Tuple[np.ndarray, np.ndarray, str]:
    """Run SED inference on a single recording.

    Loads all overlapping segments, applies the classifier to each of them,
    and returns predictions only for segments starting at whole-second timestamps.

    Parameters
    ----------
    filepath : str
        Path to the .npz feature file for one recording.
    classifier : MultiOutputClassifier
        A fitted multi-label classifier.

    Returns
    -------
    predictions : np.ndarray of shape (N_whole_second_segments, N_classes)
        Binary predictions (0 or 1) for each class at each whole-second segment.
    start_times : np.ndarray of shape (N_whole_second_segments,)
        Start times (in seconds) of the retained segments.
    filename : str
        Audio filename for this recording (e.g. '000001.wav').
    """
    data = dict(np.load(filepath, allow_pickle=True))
    start_times_all = data["start_time"]   # [0.0, 0.5, 1.0, 1.5, ...]

    # Build feature matrix for ALL segments (both whole-second and half-second)
    X_all = build_feature_matrix(data)     # shape: (N_all_segments, D_features)

    # Apply the classifier to every segment
    # MODIFIED FROM BASE VERSION - was: pred_all = classifier.predict(X_all)
    pred_all = classifier.predict(X_all) if hasattr(classifier, "predict") else classifier(X_all) # shape: (N_all_segments, N_classes)

    # Identify segments that start at a whole-second timestamp (0.0, 1.0, 2.0, ...)
    whole_second_mask = np.isclose(start_times_all % 1.0, 0.0)
    # The half-second segments (0.5, 1.5, ...) are dropped here.
    # Future improvement: use them to smooth predictions or refine temporal boundaries.

    predictions = pred_all[whole_second_mask]
    start_times = start_times_all[whole_second_mask]

    filename = os.path.basename(filepath).replace(".npz", ".wav")
    return predictions, start_times, filename

def predictions_to_intervals(
    predictions: np.ndarray,
    start_times: np.ndarray,
    filename: str,
) -> List[Dict]:
    """Convert per-second binary predictions to onset/offset interval annotations.

    Consecutive active 1-second segments are merged into a single interval.
    For example, predictions active at t = 2 s and t = 3 s are merged into
    the interval [2.0, 4.0] (both 1-second windows combined).

    Parameters
    ----------
    predictions : np.ndarray of shape (N_segments, N_classes)
        Binary predictions for each class at each whole-second segment.
    start_times : np.ndarray of shape (N_segments,)
        Start time (in seconds) of each segment.
    filename : str
        Audio filename (e.g. '000001.wav').

    Returns
    -------
    rows : list of dict with keys: 'filename', 'annotation', 'onset', 'offset'
    """
    rows = []

    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        cls_preds = predictions[:, cls_idx]

        in_event = False
        onset    = None

        for t, pred in zip(start_times, cls_preds):
            if pred == 1 and not in_event:
                onset    = float(t)
                in_event = True
            elif pred == 0 and in_event:
                # The event ends at the start of the first inactive segment,
                # which equals the start time of the previous active segment + 1 s.
                offset = float(t)
                rows.append({"filename": filename, "annotation": cls_name,
                             "onset": onset, "offset": offset})
                in_event = False

        # Handle an event that is still active at the very end of the recording
        if in_event:
            offset = float(start_times[-1]) + SEGMENT_LENGTH
            rows.append({"filename": filename, "annotation": cls_name,
                         "onset": onset, "offset": offset})

    return rows

def generate_predictions(
    file_list: List[str],
    classifier: MultiOutputClassifier,
) -> pd.DataFrame:
    """Run SED inference on a list of recordings and return predictions as a DataFrame.

    Parameters
    ----------
    file_list : list of str
        Paths to .npz feature files.
    classifier : MultiOutputClassifier
        A fitted multi-label classifier.

    Returns
    -------
    pred_df : pd.DataFrame
        Columns: filename, annotation, onset, offset.
        One row per detected event interval.
        Recordings with no detected events are not represented.
    """
    all_rows = []
    for filepath in file_list:
        preds, times, fname = run_sed_inference(filepath, classifier)
        all_rows.extend(predictions_to_intervals(preds, times, fname))

    if not all_rows:
        return pd.DataFrame(columns=["filename", "annotation", "onset", "offset"])
    return pd.DataFrame(all_rows)

def evaluate_split(
    pred_df: pd.DataFrame,
    file_list: List[str],
    ann_df: pd.DataFrame,
) -> Tuple[float, pd.DataFrame]:
    """Evaluate SED predictions against ground-truth annotations for a given split.

    Parameters
    ----------
    pred_df : pd.DataFrame
        Prediction DataFrame with columns: filename, annotation, onset, offset.
    file_list : list of str
        Paths to .npz files for the split to evaluate.
    ann_df : pd.DataFrame
        Full annotation DataFrame (loaded from annotations.csv); will be filtered.

    Returns
    -------
    macro_f1 : float
    results : pd.DataFrame
        Per-class precision, recall, and F1 score.
    """
    split_filenames = {os.path.basename(f).replace(".npz", ".wav") for f in file_list}

    # Filter annotations to only include files in this split
    ann_split = ann_df[ann_df["filename"].isin(split_filenames)].copy()

    # Aggregate per-annotator annotations via majority vote
    gt = aggregate_ground_truth_annotations(ann_split)

    # Build segment-level indicator matrices (indexed by filename + second)
    gt_segments   = build_segment_frame_from_intervals(gt,      name="ground_truth")
    pred_segments = build_segment_frame_from_intervals(pred_df, name="predictions")

    # Restrict predictions to files belonging to this split
    if len(pred_segments) > 0:
        pred_filenames = pred_segments.index.get_level_values("filename")
        pred_segments  = pred_segments[pred_filenames.isin(split_filenames)]

    return calculate_f1_score(gt_segments, pred_segments)