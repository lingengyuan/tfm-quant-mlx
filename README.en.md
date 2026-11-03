# TimesFM-MLX Quantization and Long-Context KV Lab

[![tests](https://github.com/lingengyuan/tfm-quant-mlx/actions/workflows/tests.yml/badge.svg)](https://github.com/lingengyuan/tfm-quant-mlx/actions/workflows/tests.yml)

Weight precision, low-bit quantization, and long-context K/V working-set compression for TimesFM 3.0 (331M) on Apple MLX. Every run reports point error, nine quantiles, peak memory, and throughput on the same windows.

- Model: `google/timesfm-3.0-pytorch`, 20 layers, 1280 dims, 16 heads, context up to 15,360, horizon 64.
- Environment: Apple M5 / 32 GB / macOS 27.0.1, MLX 0.32.4, timesfm 3.0.2, Python 3.12.14.

The Chinese README ([README.md](README.md)) carries the full tables and per-section detail; this page is the English entry point.

## Findings

- `fp16` / `bf16`: 1323 → 661 MB (2.00×), pinball change ≤0.05%. `int8_g64`: 375 MB (3.53×), +0.11%.
- Plain `int4`: 191–251 MB, pinball +10–12%, quantile crossing 9–19%. Loss concentrates in the last layer (+7.9%) and the output head (+67.8% at 4-bit).
- 4-bit that works: `mix_l18_19_8` 227 MB (5.82×), +0.80%; clipping 0.95 (`mix_l18_19_8c95`) reaches −0.22% on ETT and +0.61% on Jena, but is worse on traffic (+2.00%).
- KV: `affine8` 1.78× (−0.01%), `affine4` 3.20× (+0.52%), `turbo3_mse` 3.10× (+2.71%). `qjl1` also reaches 1.78× but compresses only K, so it stays the worst at that ratio (+4.73%).
- KV error also concentrates in the last block: keeping layer 19 at fp16 moves `affine4` from +0.52% to **−0.36%** (137 MB, 2.88×) and `turbo3_mse` from +2.71% to +0.69% (141 MB, 2.80×).
- Multivariate: fp32 pinball 0.1152 vs 0.1168 univariate; KV loss grows ~6× because variate attention is sensitive. Sequence `affine4` + variate `affine8` gives +1.15% at 2.29×.
- Covariates: dropping them costs +4.98% pinball; int4 rises to +17.2%, while the seq/var KV mix stays at −0.91%.
- Throughput does not improve: at 482 tokens int4 GEMM is 9% slower; end-to-end differences stay ≤9%. The gain from 4-bit is memory (2774 → 1717 MB).
- Cross-backend baseline: MLX fp32 vs PyTorch on 56 series, mean normalized deviation 0.0013 (max 0.0081), pinball difference −0.001%.

## Results

### Weights (context 15,360)

| config | weights MB | ratio | bits/w | nMAE | ΔnMAE | pinball | Δpinball | cov err | cross % | series/s | peak MB |
|---|---|---|---|---|---|---|---|---|---|---|---|
| fp32 | 1323 | 1.00× | 32.00 | 0.3192 | — | 0.1258 | — | 0.0400 | 0.47 | 13 | 2774 |
| fp16 | 661 | 2.00× | 16.00 | 0.3192 | −0.01% | 0.1257 | −0.04% | 0.0415 | 0.51 | 13 | 2208 |
| bf16 | 661 | 2.00× | 16.00 | 0.3193 | +0.01% | 0.1258 | +0.03% | 0.0493 | 0.64 | 10 | 2208 |
| int8_g64 | 375 | 3.53× | 9.00 | 0.3194 | +0.07% | 0.1259 | +0.11% | 0.0459 | 0.72 | 8 | 1882 |
| int4_g64 | 210 | 6.31× | 5.00 | 0.3511 | +10.20% | 0.1385 | +10.86% | 0.0603 | 18.51 | 9 | 1717 |
| int4c95_g64 | 210 | 6.31× | 5.00 | 0.3483 | +9.74% | 0.1382 | +11.10% | 0.0618 | 16.74 | 9 | 1717 |
| int4_g32 | 251 | 5.27× | 6.00 | 0.3499 | +9.83% | 0.1380 | +10.36% | 0.0614 | 16.45 | 8 | 1756 |
| int4_g128 | 191 | 6.93× | 4.50 | 0.3549 | +11.78% | 0.1397 | +12.05% | 0.0687 | 12.12 | 10 | 1699 |
| mix_l19_8 | 219 | 6.04× | mix | 0.3234 | +1.14% | 0.1277 | +1.33% | 0.0560 | 0.73 | 9 | 1726 |
| mix_l18_19_8 | 227 | 5.82× | mix | 0.3225 | +0.84% | 0.1270 | +0.80% | 0.0544 | 0.70 | 9 | 1734 |
| mix_l18_19_8c95 | 227 | 5.82× | mix | 0.3183 | −0.14% | 0.1252 | −0.22% | 0.0546 | 0.73 | 9 | 1734 |
| mix_top3_8 | 235 | 5.62× | mix | 0.3222 | +0.74% | 0.1269 | +0.66% | 0.0532 | 0.68 | 9 | 1743 |

### KV compression (context 15,360, batch 8)

| KV mode | K bits | V bits | KV MB | reduction | nMAE | ΔnMAE | Δpinball | point dev | cov err | series/s | peak MB |
|---|---|---|---|---|---|---|---|---|---|---|---|
| affine8 | 9.00 | 9.00 | 222 | 1.78× | 0.3193 | +0.01% | −0.01% | 0.0022 | 0.0401 | 11 | 2856 |
| affine4 | 5.00 | 5.00 | 123 | 3.20× | 0.3202 | +0.62% | +0.52% | 0.0382 | 0.0431 | 11 | 2847 |
| affine4_l19fp16 | 5.55 | 5.55 | 137 | 2.88× | 0.3177 | −0.28% | −0.36% | 0.0236 | 0.0421 | 9 | 2847 |
| turbo3_mse | 5.16 | 5.16 | 127 | 3.10× | 0.3257 | +2.32% | +2.71% | 0.0607 | 0.0404 | 6 | 2922 |
| turbo3_l19fp16 | 5.70 | 5.70 | 141 | 2.80× | 0.3198 | +0.43% | +0.69% | 0.0382 | 0.0408 | 6 | 2922 |
| turbo3_prod | 5.58 | 5.58 | 138 | 2.87× | 0.3304 | +4.52% | +5.04% | 0.1004 | 0.0464 | 6 | 2804 |
| turbo2_mse | 3.58 | 3.58 | 88 | 4.48× | 0.3524 | +11.51% | +11.86% | 0.1443 | 0.0453 | 7 | 2858 |
| qjl1 | 2.00 | 16.00 | 222 | 1.78× | 0.3328 | +4.84% | +4.73% | 0.1095 | 0.0382 | 11 | 2969 |

More results: sensitivity ([weights](results/sensitivity.md), [KV layers/heads](results/kv_sensitivity.md)), [weight + KV combinations](results/combo_report.md), [multivariate](results/multi_report.md), [covariates](results/covariates.md), [domains](results/domains_report.md), [longer horizons](results/horizon.md), [profiling](results/profile.json), [PyTorch parity](results/parity.json).

## Usage

Requires Apple Silicon macOS, Python 3.12, and [uv](https://docs.astral.sh/uv/).

```bash
uv venv --python 3.12 .venv
uv pip install -e .
./scripts/download_checkpoint.sh models/timesfm-3.0
```

Weights use the non-commercial TimesFM license; they are not redistributed here.

```bash
# smoke test
.venv/bin/tfmq run --preset quick

# full grids
for cfg in full mixed combo multi uni_control multi_scope multi_kv_mixed domains domains2 kv_mixed; do
  .venv/bin/tfmq run --config configs/$cfg.yaml
done
.venv/bin/tfmq report results/full.json -o results/report.md

# probes
.venv/bin/tfmq sensitivity
.venv/bin/tfmq kv-sensitivity --kv affine4
.venv/bin/tfmq profile
.venv/bin/tfmq parity          # needs uv pip install -e ".[parity]"
.venv/bin/tfmq covariates
.venv/bin/tfmq horizon

# forecast a CSV
.venv/bin/tfmq predict data.csv --context 2048 --horizon 64 \
  --weights int8_g64 --kv affine4 -o forecasts.csv
```

`predict` decodes each numeric column independently by default. `--multi` decodes columns together as variates; `--covariates` uses the last column as the target and the others as past-only covariates.

Configuration is YAML under `configs/`. Key fields: `datasets`, `max_channels`, `contexts`, `windows`, `weights`, `kv`, `kv_contexts`, `kv_weights`, `kv_scopes`, `multi`, `out`.

## Method

- Data: ETT-small, Jena Climate (10 min), UCI Electricity (15 min), traffic (hourly), exchange (daily), and a deterministic synthetic suite.
- Windows: the last `context` points predict the next `horizon` points. Multivariate mode decodes one table's channels together; covariate mode uses the other channels as past-only and sin/cos calendar features as past-future.
- `nMAE` and pinball are normalized by each window's context standard deviation; `Δ` is relative to the same-window fp32 output.
- KV is the per-forward K/V working set: TimesFM 3.0 MLX is a single full-sequence pass with no cross-call cache.
- Weights use MLX affine quantization through `nn.QuantizedLinear`; clipping 0.95 trims each input group before quantization.

## Layout

```
src/tfmlab/
  data.py             datasets and rolling windows
  weights.py          dtype / affine / clipping / mixed precision
  sensitivity.py      layer-wise 4-bit sensitivity
  kv.py               affine, PolarQuant, QJL KV compressors
  kv_sensitivity.py   KV layer and head sensitivity
  profile.py          GEMM and end-to-end profile
  parity.py           PyTorch backend comparison
  covariates.py       covariate experiments
  horizon.py          horizons 64/256/1024
  predict.py          CSV forecasting CLI
  attention.py        attention patch and KV installation
  runtime.py          loading, inference, timing, memory
  metrics.py          point and quantile metrics
  runner.py           grid execution
  report.py           Markdown reports
configs/              grid YAML files
results/              JSON and Markdown results
tests/
```

## Limitations

- One checkpoint, zero-shot, horizon up to 1,024; longer horizons use stitching with a different error structure.
- Domains are capped at 8 channels; exchange has no 15,360-row history.
- `fp16`/`bf16` weights upcast in RoPE and running stats, so compute gains trail memory gains.
- KV compression has no fused kernel: decompression counts toward latency and peak memory does not drop.
- KV sensitivity covers 60 windows; sub-1% improvements should not be generalized.
- Throughput is a single median latency. Four quick-preset repeats give 67.9–77.3 series/s at batch 8; per-cell run-to-run spread has a median of 13% and a max of 69% on short-latency cells.
- PyTorch parity runs on CPU only.
- TimesFM 3.0 weights use a non-commercial license.

## License

Code is MIT ([LICENSE](LICENSE)). TimesFM 3.0 weights are not included and follow their non-commercial license.
