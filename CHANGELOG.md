
# Changelog

All notable changes to this project will be documented in this file.

The project follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Planned
- Expand and improve the helpdesk ticket dataset.
- Improve classification performance and error analysis.
- Establish reproducible model benchmarks.

## [0.1.0] - 2026-11-23

### Added
- BERT-based IT helpdesk ticket classification.
- LoRA fine-tuning using Hugging Face PEFT.
- Six helpdesk ticket categories.
- Gradio interface for ticket classification.
- Stratified training, validation, and test splits.
- Accuracy, macro precision, recall, and F1-score metrics.
- Automated tests for the LoRA implementation.

### Changed
- Improved prediction source transparency in the Gradio interface.
- Improved reproducibility with a fixed random seed.
- Selected the best checkpoint using validation F1-score.

### Known Limitations
- Dataset contains only 90 labeled examples.
- Model performance has not been validated on a larger independent dataset.
- Training results may vary across hardware and environments.
- The current base model is English BERT, not Portuguese BERTimbau.
