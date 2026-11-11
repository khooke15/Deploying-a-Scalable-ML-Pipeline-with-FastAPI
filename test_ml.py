import numpy as np
from sklearn.ensemble import RandomForestClassifier

from ml.model import (
    compute_model_metrics,
    inference,
    train_model,
)


def test_train_model():
    """Test that train_model returns a RandomForestClassifier."""
    X = np.array([
        [1, 2],
        [2, 3],
        [3, 4],
        [4, 5],
    ])
    y = np.array([0, 0, 1, 1])

    model = train_model(X, y)

    assert isinstance(model, RandomForestClassifier)


def test_inference():
    """Test that inference returns one prediction per input row."""
    X = np.array([
        [1, 2],
        [2, 3],
        [3, 4],
        [4, 5],
    ])
    y = np.array([0, 0, 1, 1])

    model = train_model(X, y)
    preds = inference(model, X)

    assert isinstance(preds, np.ndarray)
    assert len(preds) == len(X)


def test_compute_model_metrics():
    """Test metric calculations with perfect predictions."""
    y = np.array([0, 1, 0, 1])
    preds = np.array([0, 1, 0, 1])

    precision, recall, fbeta = compute_model_metrics(y, preds)

    assert precision == 1.0
    assert recall == 1.0
    assert fbeta == 1.0
