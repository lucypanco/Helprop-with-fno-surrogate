# HelProp Surrogate Manual

Canonical workflow for transfer-matrix surrogate data and FNO training:

```text
theta -> M[ETOA, ELIS]
```

`ETOA` and `ELIS` are energy grids. LIS flux is applied during folding.

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
  --run-dir surrogate_runs/run_proton_A1_800 \
  --n-runs 30000 \
  --learn D0 indexA indexB angle \
  --range D0:0.1:15 \
  --range indexA:0:3.0 \
  --range indexB:0:3.0 \
  --range angle:5:45 \
  --fixed A=1 \
  --fixed Z=-1 \
  --fixed polarity=1 \
  --fixed R0=1 \
  --fixed B0=5 \
  --etoa 0.1,120,80\
  --elis 0.1,150,80\
  --number 800\
  --nthread 5 \
  --jobs 3
```

For FNO training, make the `ETOA` and `ELIS` grids wider than the final
comparison range. Keep `ELIS` wider than `ETOA` for folding.

The FNO uses separate Fourier-mode counts for the two matrix axes:

| Matrix axis | Command option | Meaning |
|---|---|---|
| ETOA/output-energy axis | `--modes-etoa` | Retained Fourier modes along the matrix rows |
| ELIS/input-energy axis | `--modes-elis` | Retained Fourier modes along the matrix columns |

`--modes N` sets both counts to `N`. These are mode counts, not bin counts.

```bash
--modes-etoa 20 \
--modes-elis 20
```

Parallelism:

```text
--nthread  threads inside one HelProp run
--jobs     independent HelProp parameter points run concurrently
```

```text
--nthread * --jobs <= physical CPU cores
```

Starting and stopping individual matrix processes does not reduce CPU use
while HelProp is running; it only creates idle gaps and therefore lowers the
average load at the cost of throughput. On a busy server, reduce `--jobs`
first, then reduce `--nthread` (and leave some physical cores unused). Add
pauses only when server responsiveness matters more than completion time.

Outputs:

```text
surrogate_runs/run_0001/config.json
surrogate_runs/run_0001/data/manifest.csv
surrogate_runs/run_0001/data/matrices.npz
surrogate_runs/run_0001/data/train_indices.txt
surrogate_runs/run_0001/data/val_indices.txt
surrogate_runs/run_0001/data/test_indices.txt
```

If generation is interrupted, resume it by passing the existing run directory
to `--continue`:

```bash
python -m helprop_surrogate.matrix_data --continue surrogate_runs/run_0001
```

The command reads `config.json`, reconstructs the deterministic parameter
design, scans `data/matrix_*.bson` for completed matrix numbers, and generates
only the missing numbered files before rebuilding the dataset and split files.
No continuation-specific JSON file is created.

For discrete parameters:

```bash
--learn D0 m polarity
--range D0:0.1:50
--range m:-2:2
--choice polarity:-1,1
--integer-param polarity
```

Usually fix `A`, `Z`, and `polarity`.

For an electron-only surrogate, keep `A` and `Z` fixed and learn only the
transport/solar parameters:

```bash
python -m helprop_surrogate.matrix_data \
  --helprop ./HelProp \
  --run-root surrogate_runs/ \
  --n-runs 500 \
  --learn D0 indexA indexB angle hcs-osc-amp hcs-osc-phase hcs-omega \
  --range D0:0.1:50 \
  --range indexA:0.5:2.0 \
  --range indexB:0.5:2.0 \
  --range angle:5:30 \
  --range hcs-osc-amp:0:10 \
  --range hcs-osc-phase:0:10 \
  --range hcs-omega:0:4 \
  --fixed A=0 \
  --fixed Z=-1 \
  --fixed m=0 \
  --fixed polarity=-1 \
  --fixed R0=1 \
  --etoa 0.001,1e7,300 \
  --elis 0.001,1e8,450 \
  --number 200 \
  --nthread 4 \
  --jobs 2
```

For a positron-only surrogate, use the same command with `--fixed Z=1`.

For integer nuclear species:

```bash
--learn D0 m A Z
--range D0:0.1:50
--range m:-2:2
--range A:1:56
--range Z:1:26
--integer-param A
--integer-param Z
```

`A` and `Z` are sampled independently; fixed-species runs are safer.

## 3. Train FNO

Train from the `.npz` dataset and its three index files; use `--fixed` for
fixed parameters:

To override the saved split files with a new 80/10/10 split:

```bash
--train-indices 0.8 \
--val-indices 0.1 \
--test-indices 0.1
```

Use `0` to disable a split. For example, `--train-indices 0.9
--val-indices 0 --test-indices 0.1` trains on 90% of the matrices, performs no
validation, and reserves 10% for testing.

```bash
python -m helprop_surrogate.fno.train \
  --dataset surrogate_runs/run_proton_A-1_test/data/matrices.npz \
  --outdir surrogate_runs/run_proton_A-1_test \
  --lis Proton_spectrum.txt \
  --epochs 700 \
  --batch-size 36 \
  --width 96 \
  --layers 6 \
  --modes-etoa 32 \
  --modes-elis 40 \
  --projection-size 192 \
  --boundary-padding 8 \
  --dropout 0.02 \
  --learning-rate 1e-3 \
  --weight-decay 1e-5 \
  --lr-scheduler plateau \
  --lr-scheduler-factor 0.3 \
  --lr-scheduler-patience 100 \
  --lr-scheduler-cooldown 3 \
  --lr-scheduler-min-lr 1e-7 \
  --early-stopping \
  --early-stopping-patience 300 \
  --early-stopping-min-delta 0.001 \
  --early-stopping-min-epochs 300 \
  --no-checkpoints \
  --checkpoint-every 25 \
  --device cuda \
  --seed 123 \
  --train-indices 0.9 \
  --val-indices 0.05 \
  --test-indices 0.05 \
  --fixed A=1 \
  --fixed Z=-1 \
  --fixed polarity=-1 \
  --fixed R0=1 \
  --fixed B0=5 \
  --matrix-cross-entropy-weight 1 \
  --matrix-probability-loss-weight 1 \
  --spectrum-loss-weight 2 \
  --spectrum-max-error-percent 0.7 \
  --spectrum-max-error-temperature-percent 0.01 \
  --spectrum-huber-delta-percent 1 \
  --spectrum-top-k 8 \
  --verbose-train
```
--reserve-checkpoint surrogate_runs/run_proton_A1_fno/reserve.pt \
```
  --early-stopping \
  --early-stopping-patience 50 \
  --early-stopping-min-delta 0.1 \
  --early-stopping-min-epochs 100 \
```

Training outputs:

```text
surrogate_runs/run_Proton_A1_fno/kernel_fno.pkl
surrogate_runs/run_Proton_A1_fno/reserve.pt
surrogate_runs/run_Proton_A1_fno/logs/loss_history.csv
surrogate_runs/run_Proton_A1_fno/validation/val_predictions.npz
surrogate_runs/run_Proton_A1_fno/validation/val_residuals.npz
surrogate_runs/run_Proton_A1_fno/validation/val_metrics.csv
surrogate_runs/run_Proton_A1_fno/testing/test_metrics.json
```

Resume after interruption with:

```bash
--resume surrogate_runs/run_proton_A1_fno/reserve.pt
```

Use:

```text
logs/loss_history.csv              loss vs epoch
validation/val_residuals.npz       residual plots
validation/val_metrics.csv         validation RMSE/KL by parameter point
testing/test_metrics.json          final held-out matrix and spectrum metrics
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
  surrogate_runs/run_proton_A1_fno/kernel_fno.pkl \
  --D0 5 \
  --m 0 \
  --angle 15\
  --indexA 1 \
  --indexB 1 \
  --lis Proton_spectrum.txt \
  --spectrum-out surrogate_runs/run_proton_A1_fno/spectrum_pred_A1.txt \
  --spectrum-etoa 0.05,120,250
```

During folding, LIS is interpolated onto the trained `ELIS` grid. The
`--spectrum-etoa` option controls the saved TOA grid and must stay within the
trained `ETOA` range.

Check every folded spectrum and write a JSON report. The default threshold is a
maximum relative error below 1% per matrix:

```bash
python -m helprop_surrogate.folding_report \
  --model surrogate_runs/run_proton_A1_fno/kernel_fno.pkl \
  --dataset surrogate_runs/run_proton_A1_fno/data/matrices.npz \
  --lis Proton_spectrum.txt \
  --workers 4 \
  --progress-every 1000 \
  --max-matrices 1000 \
  --etoa-range 1,120 \
  --report-out surrogate_runs/run_proton_A1_fno/testing/folding_report.json
```

Exit code 0 means all matrices pass. Use `--max-matrices N` to test only the
first `N` entries. To evaluate the folded-spectrum error only over an inclusive
TOA energy window, add `--etoa-range MIN,MAX`; for example, to assess 1--100
GeV:

```bash
  --etoa-range 1,100
```

The report uses the existing ETOA grid points inside that window and records
the requested and actual range in the JSON output. It does not create new
energy points between grid values.

Use in MCMC:

```bash
python helprop_mcmc/mcmc_analysis.py \
  --surrogate-model surrogate_runs/run_proton_A1_fno/kernel_fno.pkl \
  --lis formula \
  --obs proton_BR_2531_obs.txt \
  --sampler dynesty \
  --A 1 --Z 1 --polarity 1 --R0 1 --B0 5 \
  --sampler dynesty \
  --nproc 1 \
  --angle 20 \
  --sample-param D0 \
  --sample-param indexA \
  --indexB 1.8 \
  --lis-a6 1.53 \
  --lis-a7 -1.2 \
  --lis_a1 -0.96 \
  --sample-param lis_a0 --sample-param lis_a2 \
  --sample-param lis_a3 --sample-param lis_a4 --sample-param lis_a5 \
  --sample-range D0:0.1:6 \
  --sample-range indexA:0:3 \
  --sample-range lis_a0:4800:6000 \
  --sample-range lis_a2:-0.1:0.2 \
  --sample-range lis_a3:-3.5:-0.2 \
  --sample-range lis_a4:-1.2:0.5 \
  --sample-range lis_a5:1:2.4 \
  --dynesty-nlive 2000 \
  --dynesty-dlogz 0.02 \
  --dynesty-maxcall 0 \
  --outdir chains/protonA1_2
```

For dynesty, `--nwalkers`, `--nsteps`, and `--nburn` do not control the run.
Use the nested-sampling precision options instead:

```bash
  --dynesty-nlive 2000 \
  --dynesty-dlogz 0.01 \
  --dynesty-maxcall 0
```

Larger `nlive` improves posterior resolution; smaller `dlogz` tightens the
evidence-based stopping tolerance. `0` for `--dynesty-maxcall` means unlimited
likelihood calls.

Two surrogate call:
```bash
python helprop_mcmc/mcmc_analysis.py \
    --backend surrogate \
    --surrogate-low-model surrogate_runs/run_0004/kernel_fno.pkl \
    --surrogate-high-model fno_runs/run_0004/kernel_fno.pkl \
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

## 5. Merge Matrix Datasets

Merge BSON matrix outputs before training when samples were generated in
separate runs:

```bash
python -m helprop_surrogate.rebuild_matrix_npz \
    "/home/zhengyuxu/project/HelProp/HelProp-1.1.0/surrogate_runs/run_proton_A-1_test/data/" \
    --out surrogate_runs/run_proton_A-1_test/data/matrices.npz \
    --learn D0 indexA indexB angle hcs-osc-amp hcs-osc-phase hcs-omega \
    --seed 12345
```
