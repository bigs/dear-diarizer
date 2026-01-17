"""Linear probe training and evaluation."""

import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, top_k_accuracy_score


def train_linear_probe(
    embeddings: np.ndarray,
    labels: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
    max_iter: int = 1000,
) -> dict:
    """Train and evaluate a linear probe on speaker embeddings.

    Args:
        embeddings: Utterance embeddings [N, 768]
        labels: Speaker IDs [N]
        test_size: Fraction of data to use for testing
        random_state: Random seed for reproducibility
        max_iter: Maximum iterations for logistic regression

    Returns:
        Dictionary with evaluation metrics:
        - top1_accuracy: Top-1 accuracy on test set
        - top5_accuracy: Top-5 accuracy on test set
        - num_train: Number of training samples
        - num_test: Number of test samples
        - num_classes: Number of speaker classes
    """
    # Split data (stratified to maintain class balance)
    X_train, X_test, y_train, y_test = train_test_split(
        embeddings,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )

    # Train logistic regression (multinomial)
    clf = LogisticRegression(
        max_iter=max_iter,
        random_state=random_state,
        multi_class="multinomial",
        solver="lbfgs",
        verbose=1,
    )

    print(f"\nTraining linear probe on {len(X_train)} samples...")
    clf.fit(X_train, y_train)

    # Evaluate on test set
    print(f"Evaluating on {len(X_test)} test samples...")
    y_pred = clf.predict(X_test)
    top1_acc = accuracy_score(y_test, y_pred)

    # Top-5 accuracy (if we have >= 5 classes)
    num_classes = len(np.unique(labels))
    if num_classes >= 5:
        # Get predicted probabilities for top-k
        y_proba = clf.predict_proba(X_test)
        top5_acc = top_k_accuracy_score(y_test, y_proba, k=5, labels=clf.classes_)
    else:
        top5_acc = None

    results = {
        "top1_accuracy": float(top1_acc),
        "top5_accuracy": float(top5_acc) if top5_acc is not None else None,
        "num_train": int(len(X_train)),
        "num_test": int(len(X_test)),
        "num_classes": int(num_classes),
    }

    return results


def cross_validate_probe(
    embeddings: np.ndarray,
    labels: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42,
    max_iter: int = 1000,
) -> dict:
    """Cross-validate linear probe.

    Args:
        embeddings: Utterance embeddings [N, 768]
        labels: Speaker IDs [N]
        n_splits: Number of CV folds
        random_state: Random seed
        max_iter: Maximum iterations for logistic regression

    Returns:
        Dictionary with cross-validation results
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    top1_scores = []
    top5_scores = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(embeddings, labels)):
        print(f"\n=== Fold {fold + 1}/{n_splits} ===")
        X_train, X_test = embeddings[train_idx], embeddings[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]

        # Train
        clf = LogisticRegression(
            max_iter=max_iter,
            random_state=random_state,
            multi_class="multinomial",
            solver="lbfgs",
        )
        clf.fit(X_train, y_train)

        # Evaluate
        y_pred = clf.predict(X_test)
        top1_acc = accuracy_score(y_test, y_pred)
        top1_scores.append(top1_acc)

        # Top-5
        num_classes = len(np.unique(labels))
        if num_classes >= 5:
            y_proba = clf.predict_proba(X_test)
            top5_acc = top_k_accuracy_score(y_test, y_proba, k=5, labels=clf.classes_)
            top5_scores.append(top5_acc)

        print(f"Top-1: {top1_acc:.4f}")
        if num_classes >= 5:
            print(f"Top-5: {top5_acc:.4f}")

    results = {
        "top1_mean": float(np.mean(top1_scores)),
        "top1_std": float(np.std(top1_scores)),
        "top5_mean": float(np.mean(top5_scores)) if top5_scores else None,
        "top5_std": float(np.std(top5_scores)) if top5_scores else None,
        "n_splits": n_splits,
    }

    return results
