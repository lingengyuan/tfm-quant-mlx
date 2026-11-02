# TimesFM-MLX 量化与长上下文 KV 压缩实验室

[![tests](https://github.com/lingengyuan/tfm-quant-mlx/actions/workflows/tests.yml/badge.svg)](https://github.com/lingengyuan/tfm-quant-mlx/actions/workflows/tests.yml)

在 Apple Silicon 上评测 TimesFM 3.0（331M，MLX 后端）的权重精度、低比特量化与长上下文 KV 工作集压缩，统一测算点预测、9 个十分位、显存和吞吐。

- 模型：`google/timesfm-3.0-pytorch`，20 层、1280 维、16 头、上下文上限 15,360、horizon 64。
- 环境：Apple M5 / 32 GB / macOS 27.0.1，MLX 0.32.4，timesfm 3.0.2，Python 3.12.14。

## 结论

- `fp16` / `bf16`：权重 1323 → 661 MB（2.00×），pinball 变化 ≤0.05%；`int8_g64` 375 MB（3.53×），+0.11%。
- 纯 `int4`：191–251 MB（5.3–6.9×），pinball +10–12%，逆序率 9–19%。损失集中在末层（单独 4-bit +7.9%）和输出头（4-bit +67.8%）。
- 4-bit 体积下用 `mix_l18_19_8`（227 MB，5.82×，+0.80%）或 `mix_top3_8`（235 MB，5.62×，+0.66%），逆序率 0.7%。
- KV：`affine8` 1.78×（−0.01%）、`affine4` 3.20×（+0.52%）、`turbo3_mse` 4.96×（+2.71%）；1-bit QJL +73%，不可用。
- KV 末层：第 19 层留 fp16，`affine4` 从 +0.52% 降到 **−0.36%**（137 MB，2.88×）；`turbo3_mse` 从 +2.71% 降到 +0.69%（95 MB，4.14×）。逐头误差分散，无单头主导。
- 组合：`mix_l18_19_8 + affine4_l19fp16` 的 pinball +0.82%，权重 227 MB、KV 137 MB、峰值 1755 MB（fp32 2774 MB）。
- 多变量（ETT 7 通道同批）：fp32 pinball 0.1152，优于单变量 0.1168；KV 压缩损失放大约 6 倍，主要来自 variate attention；seq 用 `affine4`、var 用 `affine8` 后为 +1.15%（2418 MB，2.29×）。
- 量化不提速：模型层 GEMM 只有 482 token，int4 反而慢 9%；端到端差异 ≤9%。4-bit 的收益在显存（2774 → 1717 MB），不在吞吐。

选择：权重用 `int8_g64` 或 `mix_l18_19_8`；单变量 KV 用 19 层 `affine4` + 末层 fp16；多变量 KV 用 seq `affine4` + var `affine8`。

## 结果

### 权重与混合精度（上下文 15,360）

| config | weights MB | ratio | bits/w | nMAE | ΔnMAE | pinball | Δpinball | cov err | cross % | series/s | peak MB |
|---|---|---|---|---|---|---|---|---|---|---|---|
| fp32 | 1323 | 1.00× | 32.00 | 0.3192 | — | 0.1258 | — | 0.0400 | 0.47 | 13 | 2774 |
| fp16 | 661 | 2.00× | 16.00 | 0.3192 | −0.01% | 0.1257 | −0.04% | 0.0415 | 0.51 | 13 | 2208 |
| bf16 | 661 | 2.00× | 16.00 | 0.3193 | +0.01% | 0.1258 | +0.03% | 0.0493 | 0.64 | 10 | 2208 |
| int8_g64 | 375 | 3.53× | 9.00 | 0.3194 | +0.07% | 0.1259 | +0.11% | 0.0459 | 0.72 | 8 | 1882 |
| int4_g64 | 210 | 6.31× | 5.00 | 0.3511 | +10.20% | 0.1385 | +10.86% | 0.0603 | 18.51 | 9 | 1717 |
| int4_g32 | 251 | 5.27× | 6.00 | 0.3499 | +9.83% | 0.1380 | +10.36% | 0.0614 | 16.45 | 8 | 1756 |
| int4_g128 | 191 | 6.93× | 4.50 | 0.3549 | +11.78% | 0.1397 | +12.05% | 0.0687 | 12.12 | 10 | 1699 |
| mix_l19_8 | 219 | 6.04× | mix | 0.3234 | +1.14% | 0.1277 | +1.33% | 0.0560 | 0.73 | 9 | 1726 |
| mix_l18_19_8 | 227 | 5.82× | mix | 0.3225 | +0.84% | 0.1270 | +0.80% | 0.0544 | 0.70 | 9 | 1734 |
| mix_top3_8 | 235 | 5.62× | mix | 0.3222 | +0.74% | 0.1269 | +0.66% | 0.0532 | 0.68 | 9 | 1743 |

`int4_g128` 跳过输入维 192 的残差块，这部分保持 fp32。混合配置其余层 4-bit；`mix_l19_8` 把第 19 层和输入残差块留 8-bit，`mix_l18_19_8` 再留第 18 层，`mix_top3_8` 再留第 3 层。输出头始终 fp32。

### 为什么量化不提速（上下文 15,360）

MLX 的 `nn.quantize` 把 Linear 换成 `nn.QuantizedLinear`，矩阵乘走 `mx.quantized_matmul`，不是缺 kernel。问题在 shape：

| tokens | fp32 ms | int8 ms | int4 ms |
|---|---|---|---|
| 482（模型单层实际） | 0.71 | 0.80 | 0.77 |
| 15,424（大 batch） | 5.98 | 5.85 | 5.69 |

端到端（batch 8）：

| 模式 | ms | series/s | peak MB |
|---|---|---|---|
| fp32 | 626.8 | 12.8 | 2774 |
| int8_g64 | 599.8 | 13.3 | 1882 |
| int4_g64 | 594.8 | 13.4 | 1717 |
| fp16 | 637.1 | 12.6 | 2208 |
| fp16 权重 + fp16 计算 | 639.4 | 12.5 | 2208 |
| sdpa（无 KV 压缩） | 575.3 | 13.9 | 2702 |
| affine4 | 674.7 | 11.9 | 2846 |
| affine4 + sdpa | 600.6 | 13.3 | 2778 |

- 端到端差异都在 9% 以内。受控 A/B 下融合注意力为 3.4%（109.6 → 105.9 ms，batch 1）。
- KV `affine4` 的解码开销让端到端慢 7.6%（674.7 vs 626.8 ms）；加融合注意力后 600.6 ms。
- KV 提速需要“解压 + 注意力”融合的自定义 kernel，本仓库没有实现。
- `tfmq profile` 复现上表，输出 [`results/profile.json`](results/profile.json)。

### 逐层 4-bit 敏感度（上下文 2,048）

| 组件 | bits | Δpinball | cross % |
|---|---|---|---|
| 20 层中的 18 层，各自单独量化 | 4 | −0.20% ~ +0.49% | 0.00 |
| layer 19 | 4 | +7.91% | 16.27 |
| 输入残差块 | 4 | +2.55% | 0.00 |
| 全部 attention（20 层） | 4 | +4.96% | 2.86 |
| 全部 FFN（20 层） | 4 | +6.72% | 14.41 |
| 输出头 | 8 | +0.95% | 8.68 |
| 输出头 | 4 | +67.78% | 36.30 |

完整逐层表见 [`results/sensitivity.md`](results/sensitivity.md)，混合精度结果见 [`results/mixed_report.md`](results/mixed_report.md)。

### KV 压缩（上下文 15,360，batch 8）

| KV mode | K bits | V bits | KV MB | reduction | nMAE | ΔnMAE | Δpinball | point dev | mean dev | cov err | series/s | peak MB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| affine8 | 9.00 | 9.00 | 222 | 1.78× | 0.3193 | +0.01% | −0.01% | 0.0022 | 0.0024 | 0.0401 | 7 | 2856 |
| affine4 | 5.00 | 5.00 | 123 | 3.20× | 0.3202 | +0.62% | +0.52% | 0.0382 | 0.0405 | 0.0431 | 8 | 2846 |
| affine4_l19fp16 | 5.55 | 5.55 | 137 | 2.88× | 0.3177 | −0.28% | −0.36% | 0.0236 | 0.0256 | 0.0421 | 8 | 2846 |
| turbo3_mse | 3.23 | 3.23 | 80 | 4.96× | 0.3257 | +2.32% | +2.71% | 0.0607 | 0.0668 | 0.0404 | 5 | 2921 |
| turbo3_l19fp16 | 3.87 | 3.87 | 95 | 4.14× | 0.3198 | +0.43% | +0.69% | 0.0382 | 0.0433 | 0.0408 | 5 | 2921 |
| turbo3_prod | 3.23 | 3.23 | 80 | 4.95× | 0.3439 | +8.80% | +10.34% | 0.1399 | 0.1535 | 0.0419 | 4 | 2883 |
| turbo2_mse | 2.23 | 2.23 | 55 | 7.16× | 0.3524 | +11.51% | +11.86% | 0.1443 | 0.1560 | 0.0453 | 5 | 2857 |
| qjl1 | 2.00 | 16.00 | 222 | 1.78× | 0.5365 | +70.09% | +73.33% | 0.4491 | 0.5597 | 0.0294 | 7 | 2968 |

`_l19fp16` 只把第 19 层留在 fp16，其余 19 层照常压缩，多花 14–15 MB 逻辑 KV。峰值显存不变：压缩结果仍要解码回 fp32 再算注意力。

完整表格、四个上下文（512 / 2048 / 8192 / 15360）和每十分位偏移见 [`results/report.md`](results/report.md)。

### KV 逐层与逐头（上下文 15,360）

60 个窗口（ETTh1 + synthetic），只压序列注意力。数值与完整网格不同，因为数据集和窗口数不同。

| 配置 | affine4 Δpinball | turbo3_mse Δpinball | 逻辑 KV MB（affine4 / turbo3） |
|---|---|---|---|
| 全量压缩 | +0.652% | +2.261% | 123 / 80 |
| 只压第 19 层 | +0.778% | +1.534% | — |
| 19 层压缩，第 19 层不压 | −0.301% | +0.303% | 137 / 95 |
| 误差最大的单头 | +0.240%（head15） | +0.490%（head08） | — |
| 16 头合计 | +1.595% | +2.285% | — |

- 两种模式下第 19 层都是主要来源：单独压它就接近或超过全量误差。
- `affine4` 下逐层误差不可加：单层合计 +1.39%，大于全量 +0.65%；只有保留第 19 层稳定回收（+0.95 pp）。
- 逐头误差分散，16 头合计与全量同量级或更高，没有单头占主导；按头混合精度的收益低于按层。

完整表：[`results/kv_sensitivity.md`](results/kv_sensitivity.md)、[`results/kv_sensitivity_turbo3.md`](results/kv_sensitivity_turbo3.md)。

### 权重 + KV 组合（上下文 15,360）

只测 `mix_l18_19_8` 一个权重档，与 4 个 KV 模式逐对组合。fp32 + KV 的数值见上面的 KV 表。

| KV | 权重 MB | KV MB | Δpinball | ΔnMAE | peak MB |
|---|---|---|---|---|---|
| 不压缩 | 227 | 395 | +0.80% | +0.84% | 1734 |
| affine4 | 227 | 123 | +1.69% | +1.79% | 1755 |
| affine4_l19fp16 | 227 | 137 | +0.82% | +1.00% | 1755 |
| turbo3_mse | 227 | 80 | +3.22% | +3.05% | 1831 |
| turbo3_l19fp16 | 227 | 95 | +1.27% | +1.18% | 1831 |

- 误差接近相加但有交互：mix +0.80%、affine4 +0.52%，线性和 +1.32%，实测 +1.69%。
- 两个 turbo 组合低于线性和：`turbo3_mse` 3.51% → 3.22%，`turbo3_l19fp16` 1.49% → 1.27%。
- 峰值显存由权重主导（2846 → 1755 MB）；KV 压缩本身不降峰值。

完整表见 [`results/combo_report.md`](results/combo_report.md)。

### 多变量输入（上下文 15,360）

ETT 每张表的 7 个通道作为 7 个 variate 进入一次 decode；对照组用同一批窗口逐通道单独 decode。两组都是 4 个 ETT、每条序列 4 个窗口，共 112 条序列。

| 配置 | 单变量 Δpinball | 多变量 Δpinball |
|---|---|---|
| fp32 pinball | 0.1168 | 0.1152 |
| int8_g64 | +0.16% | +0.21% |
| int4_g64 | +12.37% | +12.91% |
| mix_l18_19_8 | +0.97% | +1.55% |
| KV affine4 | +0.91% | +5.51% |
| KV affine4_l19fp16 | +0.06% | +0.53% |
| KV turbo3_mse | +1.68% | +11.16% |

- 多变量点预测略好（pinball 0.1152 vs 0.1168），覆盖率误差变差（0.069 vs 0.063）。
- 权重量化在两种模式下接近，差异 ≤0.6 pp。
- KV 损失主要来自 variate attention：只压 seq 为 +0.92% / +1.49%（affine4 / turbo3），只压 var 为 +3.38% / +6.92%。
- 多变量探针含 7 个 variate 且统计两种注意力，逻辑字节约为单变量表的 14 倍（affine4 1727 MB）。比较 reduction，不要比较绝对值。

按 scope 分配精度：

| KV 策略 | KV MB | reduction | Δpinball | point dev |
|---|---|---|---|---|
| affine4 双侧 | 1727 | 3.20× | +5.51% | 0.0680 |
| seq affine4 + var affine8 | 2418 | 2.29× | +1.15% | 0.0353 |
| seq affine4 + var fp16 | 3628 | 1.52× | +0.92% | 0.0351 |
| turbo3 双侧 | 1115 | 4.96× | +11.16% | 0.1112 |

var 从 affine8 提到 fp16 只多回收 0.23 pp，KV 却从 2418 增到 3628 MB，不划算。

完整表：[`results/multi_report.md`](results/multi_report.md)、[`results/uni_control_report.md`](results/uni_control_report.md)、[`results/multi_scope_report.md`](results/multi_scope_report.md)、[`results/multi_kv_mixed_report.md`](results/multi_kv_mixed_report.md)。

## 使用

### 安装

需要 Apple Silicon macOS、Python 3.12 与 [uv](https://docs.astral.sh/uv/)。

```bash
uv venv --python 3.12 .venv
uv pip install -e .
./scripts/download_checkpoint.sh models/timesfm-3.0
```

权重使用 non-commercial 许可证，本仓库不重新分发。

### 快速开始

```bash
.venv/bin/tfmq run --preset quick
```

结果写入 `results/quick.json`，约 2 分钟。

### 配置

每个 YAML 字段的示例见 `configs/*.yaml`。

| 字段 | 作用 |
|---|---|
| `datasets` / `synthetic` | 评测数据集；`synthetic` 关闭时只用 ETT |
| `contexts` / `windows` | 上下文长度；每条序列的窗口数 |
| `weights` | 权重配置列表 |
| `kv` / `kv_contexts` | KV 模式；测 KV 的上下文 |
| `kv_weights` | 与 KV 网格配对的权重配置 |
| `kv_scopes` | `seq` / `var` / `both` |
| `multi` | 同一窗口的通道一起 decode |
| `out` | 结果 JSON 路径 |

### 完整网格与探针

```bash
for cfg in full mixed combo multi uni_control multi_scope multi_kv_mixed kv_mixed; do
  .venv/bin/tfmq run --config configs/$cfg.yaml
done
.venv/bin/tfmq report results/full.json -o results/report.md
.venv/bin/tfmq sensitivity
.venv/bin/tfmq kv-sensitivity --kv affine4
.venv/bin/tfmq kv-sensitivity --kv turbo3_mse \
  --out results/kv_sensitivity_turbo3.json --md results/kv_sensitivity_turbo3.md
.venv/bin/tfmq profile
```

完整网格在 M5 / 32 GB 上约 20–25 分钟。

### 自检

```bash
.venv/bin/python -m pytest tests -q
.venv/bin/tfmq sanity
```

## 方法

### 数据与指标

- 数据：ETT-small（ETTh1/ETTh2/ETTm1/ETTm2，每表 7 个通道）加 8 条确定性合成序列。
- 窗口：每条序列用最后 `context` 个点预测未来 64 点；多变量模式下同一时刻的 7 个通道一起 decode。
- `nMAE`、`pinball` 按每条窗口上下文标准差归一化；`Δ` 为相对同窗口 fp32 的变化。
- `cov err` 为 9 个十分位覆盖率与名义值之差的平均；`cross %` 为排序前的相邻十分位逆序比例。
- `KV MB` 按张量形状和比特数计算，`reduction` 相对 fp16 K/V；`peak MB` 为 MLX 峰值显存，含常驻权重。

### 权重量化

- `fp32` / `fp16` / `bf16`：整棵模块树转换 dtype。
- `int8_g64` / `int4_g64` / `int4_g32` / `int4_g128`：MLX affine 量化，输出头保持 fp32；`int4_g128` 跳过输入维不整除 128 的层。
- 混合精度按 `tfmq sensitivity` 的逐层结果保留末层、残差块或第 3 层为 8-bit。

### KV 压缩

TimesFM 3.0 的 MLX 后端是单次全序列前向，没有跨调用 KV cache；这里压缩每层注意力的 K/V 工作集。

- `affine8` / `affine4`：MLX affine 量化，按头拼平后 group=64。
- `turbo3_mse` / `turbo2_mse` / `turbo3_prod`：PolarQuant / PolarQuant+QJL，参考 `lingengyuan/qjl-mlx` 的 MLX 实现，head_dim 补零到 128。
- `qjl1`：1-bit sign 估计，只压 K，V 保持 fp16。
- `_l19fp16`：按层分配压缩器，第 19 层直通。
- `affine4_seq_var8` / `affine4_seq_varfp16`：按注意力类型分配压缩器。
- `kv_weights` 组合权重与 KV 网格；`kv_scopes` 拆分 seq / var 的压缩误差。

压缩后先解码再参与注意力；逻辑 K/V 变小，运行时峰值显存不一定下降。

## 目录

```
src/tfmlab/
  data.py            数据与滚动窗口
  weights.py         dtype / affine / 混合精度权重量化
  sensitivity.py     逐层 4-bit 敏感度
  kv.py              affine、PolarQuant、QJL KV 压缩
  kv_sensitivity.py  KV 逐层/逐头敏感度
  profile.py         GEMM 与端到端性能剖析
  attention.py       注意力补丁
  runtime.py         加载、推理、计时、显存
  metrics.py         点预测与分位数指标
  runner.py          网格执行
  report.py          Markdown 报告
configs/             full / quick / mixed / kv_mixed / combo / multi 等网格
results/             JSON、报告、敏感度表
tests/
```

## 限制

- 只测一个 checkpoint、零样本、horizon 64；更长的 horizon 走 stitching，误差结构不同。
- 数据以 ETT 和合成为主，不代表所有领域。
- 权重 `fp16/bf16` 在部分算子中自动升到 fp32（RoPE、running stats），算力收益小于体积收益。
- KV 压缩没有融合 kernel，解码开销计入吞吐；峰值显存下降需要自定义 Metal kernel。
- KV 敏感度只覆盖 ETTh1 + synthetic 的 60 个窗口。第 19 层的结论在两种压缩模式下一致；混合 KV 的 −0.36% 幅度小于 1%，不应外推。
- 多变量只测 ETT 的 7 通道同批；未覆盖变量数变化、变量子集和协变量输入。
- TimesFM 3.0 权重使用 non-commercial 许可证，本仓库不重新分发权重。

## 许可证

代码使用 MIT，见 [LICENSE](LICENSE)。TimesFM 3.0 权重不在本仓库内，遵循其 non-commercial 许可证。
