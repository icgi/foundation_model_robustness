# Foundation Model Robustness

Research code for making attention-based multiple-instance learning (ABMIL) classifiers more robust to scanner variation in digital pathology. Alongside the classification objective, paired scans of the same slide can be trained with two additional robustness losses: an embedding loss that aligns matched tile representations, and a score loss that aligns slide-level predictions. Models operate on precomputed pathology tile embeddings.

## Setup

Use a recent Python environment and install PyTorch for your CPU or CUDA setup. The remaining runtime dependencies are:

```bash
pip install h5py hydra-core lifelines loguru matplotlib numpy omegaconf pandas scikit-learn tqdm
```

Run all commands from the repository root.

## Data

Training and inference take a CSV with one scan per row. It must contain:

- `slide`: shared identifier for scans of the same slide
- `scanner`: scanner identifier
- `path`: path to the scan's HDF5 feature file
- the target label column named in the training config

`case_id` is optional. Each HDF5 file must contain `features`, `tile_ids`, and `coords` datasets. `out_of_bounds` is optional. Paired scans must have matching tile IDs.

## Configuration

Training and inference use the included Hydra configs:

```text
conf/default_config.yaml
conf/default_inference_config.yaml
```

For training, set `paths.data_table` and `data.target_label`; set `data.oversample_by_column` if oversampling is enabled. For inference, set `paths.data_table` and `paths.model_path`. Values can be set in the files or as command-line overrides.

## Usage

```bash
# Train
python -m src.training.run_training

# Run slide-level inference
python -m src.inference.run_inference

# Select a checkpoint from inference results
python -m src.tuning.run_tuning --inference_path /path/to/inference/results
```

Hydra values can be overridden on the command line, for example:

```bash
python -m src.training.run_training paths.data_table=data/train.csv data.target_label=diagnosis paths.output_dir=outputs/run-01
python -m src.inference.run_inference paths.data_table=data/test.csv paths.model_path=outputs/run-01/models/checkpoints
```

## Disclaimer

This README was written with AI assistance and verified by a human.
