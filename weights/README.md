# Model Weights

Training writes `best_model.pt`, `last_model.pt`, and (for EMA experiments)
`best_model_raw.pt` inside `outputs/<run_name>_<timestamp>/`.

Binary checkpoints are intentionally excluded from Git because they are large
and reproducible from the recorded configuration. Run
`scripts/run_recommended_experiments.py` to regenerate the headline custom-CNN
and fine-tuned ResNet-18 models. Each run also records its configuration,
history, evaluation reports, confusion matrices, and learning curves.
