
"""
Dataset splitting utilities for controlled helpdesk experiments.

Additional examples are included only in the training split.
Validation and test sets always come from the original dataset.
"""

import numpy as np
from sklearn.model_selection import train_test_split


def split_helpdesk_data(original_data, additional_data, label_to_id, seed=42):
    """Create reproducible training, validation, and test splits."""

    labels = [label_to_id[item["label"]] for item in original_data]

    # Split original dataset only
    train_indices, holdout_indices = train_test_split(
        np.arange(len(original_data)),
        test_size=0.30,
        random_state=seed,
        stratify=labels
    )

    val_indices, test_indices = train_test_split(
        holdout_indices,
        test_size=0.50,
        random_state=seed,
        stratify=[labels[i] for i in holdout_indices]
    )

    train_data = [original_data[i] for i in train_indices]
    val_data = [original_data[i] for i in val_indices]
    test_data = [original_data[i] for i in test_indices]

    # Additional tickets enter training only
    train_data.extend(additional_data)

    return train_data, val_data, test_data
