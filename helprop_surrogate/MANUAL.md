# HelProp Surrogate Manual

This is the canonical workflow for transfer-matrix surrogate data and FNO
training.  The shared dataset is public to all surrogate backends:

```text
theta -> M[ETOA, ELIS]
```

`ETOA` and `ELIS` are energy grids, not fluxes.  LIS flux is applied later by
folding through the predicted matrix.

## 1. Environment

Run from the repository root:

```bash
cd /path/to/Helprop1.0.01
```

Install Python dependencies in the training environment:

```bash
pip install numpy torch pymongo
```

Build HelProp first:

```bash
cmake -B build
cmake --build build
```

## 2. Generate Matrix Data

Use the backend-neutral data generator:

```bash
python -m helprop_surrogate.matrix_data \
  --helprop ./HelProp \
  --runs-root surrogate_runs \
  --n-runs 200 \
  --learn D0 m \
  --range D0:0.1:50 \
  --range m:-2:2 \
  --fixed A=1 \
  --fixed Z=1 \
  --fixed polarity=-1 \
  --fixed R0=1 \
  --etoa 0.1,100,30 \
  --elis 0.1,100,30 \
  --number 200 \
  --nthread 4 \
  --jobs 2
```

Parallelism:

```text
--nthread  threads inside one HelProp run
--jobs     independent HelProp parameter points run concurrently
```

Keep approximately:

```text
--nthread * --jobs <= physical CPU cores
```

Outputs:

```text
surrogate_runs/run_0001/config.json
surrogate_runs/run_0001/data/manifest.csv
surrogate_runs/run_0001/data/matrices.npz
surrogate_runs/run_0001/data/train_indices.txt
surrogate_runs/run_0001/data/val_indices.txt
surrogate_runs/run_0001/data/test_indices.txt
```

For discrete learned parameters, use choices:

```bash
--learn D0 m polarity
--range D0:0.1:50
--range m:-2:2
--choice polarity:-1,1
--integer-param polarity
```

Prefer fixing `A`, `Z`, and `polarity` unless you truly need the surrogate to
learn across particle species or polarities.

## 3. Train FNO

Train with the FNO backend:

```bash
python -m helprop_surrogate.fno.train \
  --dataset surrogate_runs/run_0001/data/matrices.npz \
  --epochs 300 \
  --batch-size 16 \
  --width 32 \
  --layers 4 \
  --modes 10 \
  --device cuda
```

Training outputs:

```text
surrogate_runs/run_0001/kernel_fno.pkl
surrogate_runs/run_0001/logs/loss_history.csv
surrogate_runs/run_0001/checkpoints/best.pt
surrogate_runs/run_0001/validation/val_predictions.npz
surrogate_runs/run_0001/validation/val_residuals.npz
surrogate_runs/run_0001/validation/val_metrics.csv
surrogate_runs/run_0001/testing/test_metrics.json
```

Use:

```text
logs/loss_history.csv              loss vs epoch
validation/val_residuals.npz       residual plots
validation/val_metrics.csv         validation RMSE/KL by parameter point
testing/test_metrics.json          final held-out RMSE only
```

## 4. Predict Or Use In MCMC

The saved model is compatible with the existing surrogate wrapper:

```bash
python -m helprop_surrogate.predict_kernel \
  surrogate_runs/run_0001/kernel_fno.pkl \
  --D0 5 \
  --m 0 \
  --matrix-out surrogate_runs/run_0001/M_pred.txt
```

Fold a LIS spectrum:

```bash
python -m helprop_surrogate.predict_kernel \
  surrogate_runs/run_0001/kernel_fno.pkl \
  --D0 5 \
  --m 0 \
  --lis Proton_spectrum.txt \
  --spectrum-out surrogate_runs/run_0001/spectrum_pred.txt
```

Use in MCMC:

```bash
python helprop_mcmc/mcmc_analysis.py \
  --backend surrogate \
  --surrogate-model surrogate_runs/run_0001/kernel_fno.pkl \
  --lis Proton_spectrum.txt \
  --obs ProtonModulated_ekin.txt
```
