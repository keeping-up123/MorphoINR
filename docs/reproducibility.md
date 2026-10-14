# Reproducibility checklist

1. Install the environment described by `environment.yml` or `requirements.txt`.
2. Download the licensed raw datasets listed in `data/README.md` and place the prepared files under `data/raw/<case>/`.
3. Change into the `MorphoINR/` root so `logs_final/...` and `data_final/...` resolve correctly.
4. Run the aligned AE/embedder stage with `python Code/train.py --mode embedder`.
5. Run the aligned FiLM-INR stage with `python Code/train.py --mode inr_film`.
6. Prediction is performed by the `train_inr` routine after the FiLM-INR training phase.
7. Run the evaluation scripts under `Scripts/analysis/` and `Scripts/validation/`.

Exact GPU, CUDA, PyTorch, dataset accession, and checkpoint versions should be recorded before the final public release.
