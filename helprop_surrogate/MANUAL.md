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
  --n-runs 1000 \
  --learn D0 m indexA indexB angle hcs-osc-amp hcs-osc-phase hcs-omega \
  --range D0:0.1:15 \
  --range m:-4:4 \
  --range indexA:0:3.0 \
  --range indexB:0:3.0 \
  --range angle:5:45 \
  --range hcs-osc-amp:0:10 \
  --range hcs-osc-phase:0:360 \
  --range hcs-omega:0:4 \
  --fixed A=1 \
  --fixed Z=1 \
  --fixed polarity=-1 \
  --fixed R0=1 \
  --etoa 0.001,1e7,90\
  --elis 0.001,1e7,90\
  --number 400\
  --nthread 8 \
  --jobs 3
```

For FNO training, use grids wider than the final comparison range.  The model is
least reliable at the outer grid boundary, so if the physics comparison needs
`1e-3` to `1e6 GeV`, train `ETOA` out to about `1e7 GeV` and then write or cut
the final prediction back to `1e6 GeV`.  Keep `ELIS` wider than `ETOA` because
the transfer matrix columns represent the LIS energies used in the fold.

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

For an electron-only surrogate, keep `A` and `Z` fixed and learn only the
transport/solar parameters:

```bash
python -m helprop_surrogate.matrix_data \
  --helprop ./HelProp \
  --runs-root surrogate_runs \
  --n-runs 500 \
  --learn D0 m indexA indexB angle hcs-osc-amp hcs-osc-phase hcs-omega \
  --range D0:0.1:50 \
  --range m:-2:2 \
  --range indexA:0.5:2.0 \
  --range indexB:0.5:2.0 \
  --range angle:5:30 \
  --range hcs-osc-amp:0:10 \
  --range hcs-osc-phase:0:360 \
  --range hcs-omega:0:4 \
  --fixed A=0 \
  --fixed Z=-1 \
  --fixed polarity=-1 \
  --fixed R0=1 \
  --etoa 0.001,1e7,300 \
  --elis 0.001,1e8,450 \
  --number 200 \
  --nthread 4 \
  --jobs 2
```

For a positron-only surrogate, use the same command with `--fixed Z=1`.

If you intentionally learn over integer nuclear species, mark `A` and `Z` as
integer parameters so HelProp receives integer command-line values:

```bash
--learn D0 m A Z
--range D0:0.1:50
--range m:-2:2
--range A:1:56
--range Z:1:26
--integer-param A
--integer-param Z
```

`A` and `Z` are sampled independently in this generator.  This can create
unphysical pairs, so separate fixed-species runs are usually safer.

## 3. Train FNO

Train with the FNO backend:

```bash
python -m helprop_surrogate.fno.train \
  --dataset surrogate_runs/run_0001/data/matrices.npz \
  --epochs 300 \
  --batch-size 32 \
  --width 64 \
  --layers 6 \
  --modes 10 \
  --device cuda \
  --verbose-train
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
  surrogate_runs/run_0003/kernel_fno.pkl \
  --D0 5 \
  --m 0 \
  --matrix-out surrogate_runs/run_0001/M_pred.txt
```

Fold a LIS spectrum:

```bash
python -m helprop_surrogate.predict_kernel \
  surrogate_runs/run_0003/kernel_fno.pkl \
  --D0 5 \
  --m 0 \
  --angle 15\
  --indexA 1 \
  --indexB 1 \
  --lis Proton_spectrum.txt \
  --spectrum-out fno_runs/run_0003/spectrum_pred_new.txt \
  --spectrum-etoa 0.1,1e5,250
```

During folded-spectrum prediction, the LIS file is interpolated onto the
trained `ELIS` grid, the FNO folds on the trained `ETOA x ELIS` matrix grid,
and `--spectrum-etoa` only controls the final saved TOA spectrum grid.  Keep
`--spectrum-etoa` inside the trained `ETOA` range.

Use in MCMC:

```bash
python helprop_mcmc/mcmc_analysis.py \
  --backend surrogate \
  --surrogate-model surrogate_runs/run_0001/kernel_fno.pkl \
  --lis Proton_spectrum.txt \
  --obs ProtonModulated_ekin.txt
```

Two surrogate call:
```bash
python helprop_mcmc/mcmc_analysis.py \
    --backend surrogate \
    --surrogate-low-model surrogate_runs/run_0002/kernel_fno.pkl \
    --surrogate-high-model fno_runs/run_0003/kernel_fno.pkl \
    --surrogate-split-energy 1.0 \
    --surrogate-blend-dex 0.2 \
    --lis ./Proton_spectrum.txt \
    --obs ./ProtonModulated_ekin.txt \
    --sampler dynesty \
    --nwalkers 60 \
    --nsteps 7000 \
    --nburn 1200 \
    --nproc 1 \
    --A 1 --Z 1 --polarity -1 --R0 1 --B0 5 \
    --sample-param D0 \
    --sample-param m \
    --sample-param indexA \
    --sample-param indexB \
    --sample-param angle \
    --sample-range D0:0.1:20 \
    --sample-range m:-4:4 \
    --sample-range indexA:0.5:2.0 \
    --sample-range indexB:0.5:2.0 \
    --sample-range angle:5:45\
    --outdir chains_dual_5d

```


```bash
python -m helprop_surrogate.rebuild_matrix_npz \
    surrogate_runs/run_0003/data/*.bson \
    surrogate_runs/run_0004/data/*.bson \
    surrogate_runs/run_0005/data/*.bson \
    --out surrogate_runs/merged_1000/data/matrices.npz \
    --learn D0 m indexA indexB angle hcs-osc-amp hcs-osc-phase hcs-omega \
    --seed 12345
```
python -m helprop_surrogate.fno.train \
    --dataset surrogate_runs/merged_1003/data/matrices.npz \
    --epochs 500 \
    --batch-size 8 \
    --width 64 \
    --layers 6 \
    --modes-etoa 12 \
    --modes-elis 16 \
    --projection-size 128 \
    --dropout 0.03 \
    --learning-rate 5e-4 \
    --weight-decay 1e-4 \
    --device cuda \
    --fixed A=0 \
    --fixed Z=-1 \
    --fixed polarity=-1 \
    --fixed R0=1 \
    --range D0:0.1:50 \
    --range m:-2:2 \
    --range indexA:0.5:2.0 \
    --range indexB:0.5:2.0 \
    --range angle:5:30 \
    --range hcs-osc-amp:0:10 \
    --range hcs-osc-phase:0:360 \
    --range hcs-omega:0:4 \
    --checkpoint-every 50 \
    --verbose-train

python helprop_mcmc/mcmc_analysis.py \
    --backend surrogate \
    --surrogate-model surrogate_runs/run_0003/kernel_fno.pkl \
    --lis ./Proton_spectrum.txt \
    --obs ./ProtonModulated_ekin.txt \
    --sampler dynesty \
    --nwalkers 60 \
    --nsteps 7000 \
    --nburn 1200 \
    --nproc 1 \
    --A 1 --Z 1 --polarity -1 --R0 1 --B0 5 --angle 35\
    --sample-param D0 \
    --sample-param m \
    --sample-param indexA \
    --sample-param indexB \
    --sample-range D0:0.1:15 \
    --sample-range m:-4:4 \
    --sample-range indexA:0.5:2.0 \
    --sample-range indexB:0.5:2.0 \
    --outdir chains_dual_4d
