
# Controlled Experiments — v0.2

## Objective

Evaluate the impact of adding 90 synthetic helpdesk tickets
to the training dataset while preserving identical validation
and test sets.

## Experimental Setup

- Base model: bert-base-uncased
- Fine-tuning: LoRA
- LoRA rank: 8
- LoRA alpha: 16
- Epochs: 20
- Batch size: 8
- Learning rate: 0.001
- Random seed: 42
- Validation examples: 13
- Test examples: 14

The original dataset contains 90 examples.

The expanded dataset adds 90 examples exclusively to
the training split, leaving validation and test unchanged.

## Dataset Splits

| Split | Experiment A | Experiment B |
|---|---:|---:|
| Training | 63 | 153 |
| Validation | 13 | 13 |
| Test | 14 | 14 |

## Results

| Metric | Experiment A | Experiment B |
|---|---:|---:|
| Accuracy | 0.5714 | 0.9286 |
| Precision Macro | 0.6944 | 0.9444 |
| Recall Macro | 0.5833 | 0.9167 |
| F1 Macro | 0.5897 | 0.9111 |

Experiment A correctly classified 8 of 14 test examples.

Experiment B correctly classified 13 of 14 test examples.

The expanded training dataset improved accuracy by
35.72 percentage points and macro F1 by 32.14 percentage points.

## Checkpoints

- Experiment A: checkpoints/controlled-baseline-90
- Experiment B: checkpoints/controlled-expanded-180

Both experiments saved LoRA adapters of approximately 1.15 MiB.

## Interpretation

Adding 90 labeled synthetic training examples was associated
with a substantial performance improvement on the shared
evaluation set.

The controlled data split allows a more meaningful comparison
than previous experiments that used different test sets.

## Limitations

- The test set contains only 14 examples.
- Both datasets contain synthetic helpdesk tickets.
- The original test examples were examined in prior experiments,
  so the test set is not an independent final benchmark.
- Only one random seed was evaluated.
- Exact-text overlap checks do not rule out semantic similarity
  or near-duplicate tickets.
- Performance on real-world helpdesk messages is unknown.

## Next Steps

1. Inspect per-class metrics and the confusion matrix.
2. Evaluate semantic similarity between dataset splits.
3. Run additional experiments with multiple random seeds.
4. Create an independent evaluation dataset using realistic
   tickets that were not used during development.
5. Compare LoRA against a suitable baseline.
