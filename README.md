# TimesFM-MLX 量化与长上下文 KV 压缩实验室

[![tests](https://github.com/lingengyuan/tfm-quant-mlx/actions/workflows/tests.yml/badge.svg)](https://github.com/lingengyuan/tfm-quant-mlx/actions/workflows/tests.yml)

[English](README.en.md)

在 Apple Silicon 上评测 TimesFM 3.0（331M，MLX 后端）的权重精度、低比特量化与长上下文 KV 工作集压缩，统一测算点预测、9 个十分位、显存和吞吐。

- 模型：`google/timesfm-3.0-pytorch`，20 层、1280 维、16 头、上下文上限 15,360、horizon 64。
- 环境：Apple M5 / 32 GB / macOS 27.0.1，MLX 0.32.4，timesfm 3.0.2，Python 3.12.14。

## 结论

- `fp16` / `bf16`：权重 1323 → 661 MB（2.00×），pinball 变化 ≤0.05%；`int8_g64` 375 MB（3.53×），+0.11%。
- 纯 `int4`：191–251 MB（5.3–6.9×），pinball +10–12%，逆序率 9–19%。损失集中在末层（单独 4-bit +7.9%）和输出头（4-bit +67.8%）。
- 4-bit 体积下默认用 `mix_l18_19_8`（227 MB，5.82×，+0.80%）。裁剪 0.95 的 `mix_l18_19_8c95` 在 ETT/Jena 上到 −0.22% / +0.61%，但在 traffic 上更差（+2.00%），不是通用默认。
- KV：`affine8` 1.78×（−0.01%）、`affine4` 3.20×（+0.52%）、`turbo3_mse` 3.10×（+2.71%）；`qjl1` 同为 1.78×，但只压 K、V 留 fp16，误差 +4.73%，是同压缩比下最差档。
- KV 末层：第 19 层留 fp16，`affine4` 从 +0.52% 降到 **−0.36%**（137 MB，2.88×）；`turbo3_mse` 从 +2.71% 降到 +0.69%（141 MB，2.80×）。该结论在 ETTh2 留出集上未复现：保留第 19 层只回收 0.45 pp，最大回收来自第 01 层。逐头误差分散，无单头主导。
- 组合：`mix_l18_19_8 + affine4_l19fp16` 的 pinball +0.82%，权重 227 MB、KV 137 MB、峰值 1755 MB（fp32 2774 MB）。
- 多变量（ETT 7 通道同批）：fp32 pinball 0.1152，优于单变量 0.1168；KV 压缩损失放大约 6 倍，主要来自 variate attention；seq 用 `affine4`、var 用 `affine8` 后为 +1.15%（2418 MB，2.29×）。
- 协变量：去掉协变量 pinball +4.98%；带上协变量后 int8 +0.35%、int4 +17.2%、mix −0.87%；KV `affine4` +3.55%，seq/var 混合 −0.91%。
- 跨域（ETTh1 + Jena + Electricity）：int8 +0.17%，int4 +22.7%（Jena +47.9%），mix +1.37%，`mix_l18_19_8c95` +0.11%（Jena +0.61%）；KV `affine4` +2.21%，第 19 层 fp16 +0.90%。
- 更长 horizon（64 → 1024）：int4 相对损失从 +10.1% 降到 +3.0%，mix 从 +1.6% 到 −0.8%；绝对 pinball 0.168 → 0.284。stitching 不放大相对量化误差。
- 量化不提速：模型层 GEMM 只有 482 token，三档差异 ≤3%；端到端差异 ≤7%，没有稳定收益。4-bit 的收益在显存（2774 → 1717 MB），不在吞吐。
- 基准对齐：MLX fp32 与 PyTorch 后端在 4 个 ETT 的 56 条序列上平均归一化偏差 0.0013、最大 0.0081，pinball 差 −0.001%（ctx 15,360）。

选择：权重用 `int8_g64` 或 `mix_l18_19_8`；单变量 KV 用 19 层 `affine4` + 末层 fp16（ETTh1 上验证，ETTh2 留出未复现，跨数据集需重新确认）；多变量 KV 用 seq `affine4` + var `affine8`。

## 结果

### 基准对齐：MLX vs PyTorch（上下文 15,360）

56 条序列（4 个 ETT，每条 2 个窗口），torch 2.15.0 在 CPU 上推理，与 MLX fp32 逐分位比较。

| 指标 | 值 |
|---|---|
| 最大归一化偏差 | 0.0081 |
| 平均归一化偏差 | 0.0013 |
| 95 分位归一化偏差 | 0.0034 |
| pinball 差（MLX − torch） | −0.001% |
| 相关系数 | 0.9999998 |

MLX fp32 基线与原实现一致，其余表格的 `Δ` 有可信参照。完整结果见 [`results/parity.json`](results/parity.json)。

### 权重与混合精度（上下文 15,360）

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

`int4_g128` 跳过输入维 192 的残差块，这部分保持 fp32。混合配置其余层 4-bit；`mix_l19_8` 把第 19 层和输入残差块留 8-bit，`mix_l18_19_8` 再留第 18 层，`mix_top3_8` 再留第 3 层。输出头始终 fp32。

### int4 裁剪（上下文 15,360）

量化前把每个输入分组裁到组内最大绝对值的 95%（`int4c95_g64`、`mix_l18_19_8c95`），压掉离群值，步长随之变小。

| 配置 | ETT + 合成 | 跨域 | Jena |
|---|---|---|---|
| int4_g64 | +10.86% | +22.70% | +47.88% |
| int4c95_g64 | +11.10% | +18.96% | +34.28% |
| mix_l18_19_8 | +0.80% | +1.37% | +3.62% |
| mix_l18_19_8c95 | −0.22% | +0.11% | +0.61% |

- 裁剪对混合精度在 ETT/Jena 上稳定有效：ETT 从 +0.80% 到 −0.22%，跨域从 +1.37% 到 +0.11%。
- 纯 int4 只在 Jena 上明显；ETT 持平（+10.86% → +11.10%），Electricity 变差（+9.14% → +13.21%）。
- 不是通用默认：traffic 上 `mix_l18_19_8` +1.43% → c95 +2.00%，int4 +32.9% → +42.4%（15360）；exchange 上 int4 从 +10.9% 降到 +5.1%（2048）。

默认仍用 `mix_l18_19_8`；ETT/Jena 类数据可用 `mix_l18_19_8c95`。

完整表见 [`results/clip_report.md`](results/clip_report.md)。

### 为什么量化不提速（上下文 15,360）

MLX 的 `nn.quantize` 把 Linear 换成 `nn.QuantizedLinear`，矩阵乘走 `mx.quantized_matmul`，不是缺 kernel。问题在 shape：

| tokens | fp32 ms | int8 ms | int4 ms |
|---|---|---|---|
| 482（模型单层实际） | 1.42 | 1.41 | 1.39 |
| 15,424（大 batch） | 6.41 | 5.78 | 5.62 |

端到端（batch 8）：

| 模式 | ms | series/s | peak MB |
|---|---|---|---|
| fp32 | 709.3 | 11.3 | 2774 |
| int8_g64 | 663.5 | 12.1 | 1882 |
| int4_g64 | 726.8 | 11.0 | 1717 |
| fp16 | 739.9 | 10.8 | 2208 |
| sdpa（无 KV 压缩） | 577.6 | 13.9 | 2702 |
| affine4 | 705.2 | 11.3 | 2846 |
| affine4 + sdpa | 587.3 | 13.6 | 2778 |

- 482 token 时三档相差 ≤3%，在测量波动内；15,424 token 时 int8 / int4 比 fp32 快 10–12%。
- 量化档与 fp32 的端到端差异 ≤7%；融合注意力档快 18%，但同一配置重复运行的波动可达 10%，不足以支撑稳定加速的结论。受控 A/B 下融合注意力为 3.4%（109.6 → 105.9 ms，batch 1）。
- KV `affine4` 与 fp32 持平（705.2 vs 709.3 ms）；加融合注意力后 587.3 ms。
- KV 提速需要“解压 + 注意力”融合的自定义 kernel，本仓库没有实现。
- `tfmq profile` 复现上表，输出 [`results/profile.json`](results/profile.json)。

### SDPA 后端（ETTh1，上下文 2,048 / 15,360）

同一配置分别跑手写注意力与 `mx.fast.scaled_dot_product_attention`。

| 配置 | 上下文 | manual pinball | sdpa pinball | Δpinball | ΔnMAE |
|---|---|---|---|---|---|
| fp32 | 2048 | 0.1492 | 0.1492 | −0.01% | −0.01% |
| fp32 | 15360 | 0.1373 | 0.1373 | −0.00% | −0.00% |
| mix_l18_19_8 | 2048 | 0.1531 | 0.1530 | −0.02% | −0.01% |
| mix_l18_19_8 | 15360 | 0.1384 | 0.1384 | −0.00% | −0.00% |
| affine4 | 15360 | 0.1374 | 0.1380 | +0.47% | +0.65% |
| affine4_l19fp16 | 15360 | 0.1364 | 0.1374 | +0.76% | +0.52% |

- 不压缩 KV 时两种路径的差异 ≤0.02%，SDPA 可以当等价加速档。
- 压缩 KV 后差异升到 +0.47% / +0.76%，不能把 SDPA 当作数值等价的替换。组合使用需要单独评测。
- 完整表见 [`results/attention.md`](results/attention.md)。

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
| affine8 | 9.00 | 9.00 | 222 | 1.78× | 0.3193 | +0.01% | −0.01% | 0.0022 | 0.0024 | 0.0401 | 11 | 2856 |
| affine4 | 5.00 | 5.00 | 123 | 3.20× | 0.3202 | +0.62% | +0.52% | 0.0382 | 0.0405 | 0.0431 | 11 | 2847 |
| affine4_l19fp16 | 5.55 | 5.55 | 137 | 2.88× | 0.3177 | −0.28% | −0.36% | 0.0236 | 0.0256 | 0.0421 | 9 | 2847 |
| turbo3_mse | 5.16 | 5.16 | 127 | 3.10× | 0.3257 | +2.32% | +2.71% | 0.0607 | 0.0668 | 0.0404 | 6 | 2922 |
| turbo3_l19fp16 | 5.70 | 5.70 | 141 | 2.80× | 0.3198 | +0.43% | +0.69% | 0.0382 | 0.0433 | 0.0408 | 6 | 2922 |
| turbo3_prod | 5.58 | 5.58 | 138 | 2.87× | 0.3304 | +4.52% | +5.04% | 0.1004 | 0.1088 | 0.0464 | 6 | 2804 |
| turbo2_mse | 3.58 | 3.58 | 88 | 4.48× | 0.3524 | +11.51% | +11.86% | 0.1443 | 0.1560 | 0.0453 | 7 | 2858 |
| qjl1 | 2.00 | 16.00 | 222 | 1.78× | 0.3328 | +4.84% | +4.73% | 0.1095 | 0.1133 | 0.0382 | 11 | 2969 |

`_l19fp16` 只把第 19 层留在 fp16，其余 19 层照常压缩，多花 14 MB 逻辑 KV。峰值显存不变：压缩结果仍要解码回 fp32 再算注意力。

完整表格、四个上下文（512 / 2048 / 8192 / 15360）和每十分位偏移见 [`results/report.md`](results/report.md)。

### KV 逐层与逐头（上下文 15,360）

60 个窗口（ETTh1 + synthetic），只压序列注意力。数值与完整网格不同，因为数据集和窗口数不同。

| 配置 | affine4 Δpinball | turbo3_mse Δpinball | 逻辑 KV MB（affine4 / turbo3） |
|---|---|---|---|
| 全量压缩 | +0.652% | +2.261% | 123 / 127 |
| 只压第 19 层 | +0.778% | +1.534% | — |
| 19 层压缩，第 19 层不压 | −0.301% | +0.303% | 137 / 141 |
| 误差最大的单头 | +0.240%（head15） | +0.490%（head08） | — |
| 16 头合计 | +1.595% | +2.285% | — |

- 两种模式下第 19 层都是主要来源：单独压它就接近或超过全量误差。
- `affine4` 下逐层误差不可加：单层合计 +1.39%，大于全量 +0.65%；只有保留第 19 层稳定回收（+0.95 pp）。
- 逐头误差分散，16 头合计与全量同量级或更高，没有单头占主导；按头混合精度的收益低于按层。

留出验证：策略选择用 ETTh1 + synthetic，验证用未参与选择的 ETTh2（affine4，3 个窗口）。

| 指标 | ETTh1 + synthetic | ETTh2（留出） |
|---|---|---|
| 全量压缩 Δpinball | +0.652% | +0.980% |
| 只压第 19 层 | +0.778% | +0.354% |
| 第 19 层不压 | −0.301% | +0.531% |
| 保留第 19 层回收 | +0.95 pp | +0.45 pp |
| 单层误差最大 | 第 19 层 +0.778% | 第 17 层 +0.423% |
| 保留单层回收最大 | — | 第 01 层 +1.282 pp |

第 19 层在 ETTh2 上仍是单层误差第二（+0.354%），但保留它只回收 0.45 pp；最大回收来自第 01 层。第 19 层 fp16 的策略在 ETTh2 上没有复现，按层混合精度需要逐数据集验证。

完整表：[`results/kv_sensitivity.md`](results/kv_sensitivity.md)、[`results/kv_sensitivity_turbo3.md`](results/kv_sensitivity_turbo3.md)、[`results/kv_sensitivity_holdout.md`](results/kv_sensitivity_holdout.md)。

### 权重 + KV 组合（上下文 15,360）

只测 `mix_l18_19_8` 一个权重档，与 4 个 KV 模式逐对组合。fp32 + KV 的数值见上面的 KV 表。

| KV | 权重 MB | KV MB | Δpinball | ΔnMAE | peak MB |
|---|---|---|---|---|---|
| 不压缩 | 227 | 395 | +0.80% | +0.84% | 1734 |
| affine4 | 227 | 123 | +1.69% | +1.79% | 1756 |
| affine4_l19fp16 | 227 | 137 | +0.82% | +1.00% | 1756 |
| turbo3_mse | 227 | 127 | +3.22% | +3.05% | 1831 |
| turbo3_l19fp16 | 227 | 141 | +1.27% | +1.18% | 1831 |

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
| turbo3 双侧 | 1784 | 3.10× | +11.16% | 0.1112 |

var 从 affine8 提到 fp16 只多回收 0.23 pp，KV 却从 2418 增到 3628 MB，不划算。

完整表：[`results/multi_report.md`](results/multi_report.md)、[`results/uni_control_report.md`](results/uni_control_report.md)、[`results/multi_scope_report.md`](results/multi_scope_report.md)、[`results/multi_kv_mixed_report.md`](results/multi_kv_mixed_report.md)。

### 协变量输入（上下文 15,360）

目标取每个 ETT 表的最后一个通道，其余 6 个通道作为 past-only 协变量，2 个 sin/cos 日历特征作为 past-future 协变量。4 个 ETT、每条 6 个窗口，共 24 条目标序列。

| 配置 | pinball | Δpinball | ΔnMAE | cross % | KV MB | reduction |
|---|---|---|---|---|---|---|
| fp32 无协变量 | 0.0969 | +4.982% | +5.917% | 0.07% | — | — |
| fp32 | 0.0923 | — | — | 0.08% | — | — |
| int8_g64 | 0.0926 | +0.349% | +0.228% | 0.17% | — | — |
| int4_g64 | 0.1081 | +17.150% | +13.254% | 21.92% | — | — |
| mix_l18_19_8 | 0.0915 | −0.872% | −1.708% | 0.21% | — | — |
| affine4 | 0.0956 | +3.550% | +3.652% | 5.69% | 2221 | 3.20× |
| affine4_seq_var8 | 0.0914 | −0.914% | −1.453% | 0.11% | 3109 | 2.29× |
| affine4_l19fp16 | 0.0906 | −1.794% | −3.008% | 0.07% | 2465 | 2.88× |

- 协变量本身是主要收益：去掉它们 pinball +4.98%。
- int4 在协变量下损失更大（+17.15%）；`mix_l18_19_8` 为 −0.87%。
- KV `affine4` 在 9 个 variate 下 +3.55%；seq/var 混合与第 19 层 fp16 分别回到 −0.91% 和 −1.79%。
- 24 条目标序列，幅度小于 2% 的负值不应外推。

完整表见 [`results/covariates.md`](results/covariates.md)。

### 跨域验证（上下文 15,360）

ETTh1 加 Jena Climate（10 分钟）与 UCI Electricity（15 分钟），每表取前 8 个通道，每条序列 4 个窗口。

| 配置 | ETTh1 | Jena | Electricity | 聚合 |
|---|---|---|---|---|
| int8_g64 | −0.07% | +0.44% | +0.15% | +0.17% |
| int4_g64 | +11.08% | +47.88% | +9.14% | +22.70% |
| int4c95_g64 | +9.38% | +34.28% | +13.21% | +18.96% |
| mix_l18_19_8 | +0.77% | +3.62% | −0.28% | +1.37% |
| mix_l18_19_8c95 | +0.45% | +0.61% | −0.73% | +0.11% |
| KV affine4 | +0.05% | +4.81% | +1.76% | +2.21% |
| KV affine4_l19fp16 | −0.67% | +3.44% | −0.06% | +0.90% |

- int8 跨域稳定，损失 ≤0.5 pp。
- int4 对域敏感：Jena +47.9%，是 ETTh1 的 4 倍；`mix_l18_19_8` 把 Jena 压到 +3.6%。
- KV `affine4` 在 Jena 上 +4.8%；第 19 层 fp16 把聚合损失从 +2.21% 降到 +0.90%。

朴素基线（重复最后一个上下文值）在 exchange 上优于 fp32 点预测：ctx 2048 的 nMAE 为 0.2718 vs 0.3126（fp32 差 +15.0%）；pinball 则相反，0.1359 vs 0.1115（fp32 好 18.0%）。该数据集上模型的点预测不如朴素基线，分位预测仍更好。

traffic（小时）与 exchange（日频，7,588 行，不支持 15,360）：

| 配置 | traffic 2,048 | exchange 2,048 | traffic 15,360 |
|---|---|---|---|
| int8_g64 | +0.22% | −0.15% | +0.22% |
| int4_g64 | +33.03% | +10.88% | +32.87% |
| int4c95_g64 | +37.89% | +5.14% | +42.43% |
| mix_l18_19_8 | +2.65% | −4.41% | +1.43% |
| mix_l18_19_8c95 | +1.34% | −7.29% | +2.00% |
| KV affine4 | +10.47% | +2.68% | +7.05% |
| KV affine4_l19fp16 | +5.74% | +2.15% | +2.41% |

- traffic 的 int4 损失高于 ETT（+33%），裁剪 0.95 进一步放大。
- exchange 上裁剪把 int4 从 +10.9% 降到 +5.1%，方向与 traffic 相反。

完整表见 [`results/domains_report.md`](results/domains_report.md)、[`results/domains2_report.md`](results/domains2_report.md)。

### 更长 horizon（上下文 2,048）

ETTh1，每条序列 4 个窗口。

| horizon | 配置 | Δpinball | cross % |
|---|---|---|---|
| 64 | int8_g64 | +0.052% | 0.00% |
| 64 | int4_g64 | +10.117% | 9.84% |
| 64 | mix_l18_19_8 | +1.573% | 0.00% |
| 64 | affine4 | +0.389% | 0.00% |
| 256 | int8_g64 | −0.144% | 0.00% |
| 256 | int4_g64 | +3.683% | 5.66% |
| 256 | mix_l18_19_8 | +0.561% | 0.00% |
| 256 | affine4 | +0.479% | 0.00% |
| 1024 | int8_g64 | −0.121% | 0.00% |
| 1024 | int4_g64 | +2.973% | 3.20% |
| 1024 | mix_l18_19_8 | −0.756% | 0.00% |
| 1024 | affine4 | +1.787% | 0.00% |

fp32 pinball：0.1681 / 0.2142 / 0.2840（h=64 / 256 / 1024）。batch 8 的 fp32 延迟 77–101 ms；重复运行波动最高约 20%，量化与 KV 没有稳定加速。

- int4 的相对损失随 horizon 下降（+10.12% → +2.97%），因为长 horizon 的绝对误差本身在变大。
- `mix_l18_19_8` 与 `affine4` 在各 horizon 保持 ≤1.6% / ≤1.8%。
- stitching 没有放大相对量化误差。

完整表见 [`results/horizon.md`](results/horizon.md)。

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

### 预测

```bash
.venv/bin/tfmq predict data.csv --context 2048 --horizon 64 \
  --weights int8_g64 --kv affine4 -o forecasts.csv
```

每个数值列默认作为独立单变量序列，取最后 `context` 个有效值。输出列为 `series, step, point, q0.1 … q0.9`，分位按行排序。`--weights` 与 `--kv` 省略时用 fp32 和未压缩。

- `--batch`：每次 decode 的序列数，默认 8。等长序列合批，输出顺序与列顺序一致。
- `--attention sdpa`：走 MLX 融合注意力；与手写路径不保证逐位一致，KV 压缩下差异更大。
- `--multi`：所有数值列作为 variate 一起 decode。
- `--covariates`：最后一列为目标，其余列为 past-only，sin/cos 日历特征为 past-future；`--period` 默认 24。

### 配置

每个 YAML 字段的示例见 `configs/*.yaml`。

| 字段 | 作用 |
|---|---|
| `datasets` / `synthetic` | 评测数据集；`synthetic` 关闭时只用真实数据 |
| `max_channels` | 每个数据集保留的通道数；0 表示全部 |
| `contexts` / `windows` | 上下文长度；每条序列的窗口数 |
| `weights` | 权重配置列表 |
| `kv` / `kv_contexts` | KV 模式；测 KV 的上下文 |
| `kv_weights` | 与 KV 网格配对的权重配置 |
| `kv_scopes` | `seq` / `var` / `both` |
| `attention` | `manual`（默认）或 `sdpa` 融合注意力 |
| `multi` | 同一窗口的通道一起 decode |
| `out` | 结果 JSON 路径 |

### 完整网格与探针

```bash
for cfg in full mixed combo multi uni_control multi_scope multi_kv_mixed domains domains2 kv_mixed; do
  .venv/bin/tfmq run --config configs/$cfg.yaml
done
.venv/bin/tfmq report results/full.json -o results/report.md
.venv/bin/tfmq sensitivity
.venv/bin/tfmq kv-sensitivity --kv affine4
.venv/bin/tfmq kv-sensitivity --kv turbo3_mse \
  --out results/kv_sensitivity_turbo3.json --md results/kv_sensitivity_turbo3.md
.venv/bin/tfmq kv-sensitivity --kv affine4 --offset 6  # 留出最近 6 个窗口

.venv/bin/tfmq run --config configs/attention_manual.yaml
.venv/bin/tfmq run --config configs/attention_sdpa.yaml
.venv/bin/tfmq attention-compare results/attention_manual.json results/attention_sdpa.json \
  -o results/attention.md
.venv/bin/tfmq profile
.venv/bin/tfmq parity  # 需要 torch：uv pip install -e ".[parity]"
.venv/bin/tfmq covariates
.venv/bin/tfmq horizon
```

完整网格在 M5 / 32 GB 上约 20–25 分钟。

### 自检

```bash
.venv/bin/python -m pytest tests -q
.venv/bin/tfmq sanity
```

## 方法

### 数据与指标

- 数据：ETT-small（ETTh1/ETTh2/ETTm1/ETTm2，每表 7 个通道）、Jena Climate（10 分钟，14 通道）、UCI Electricity（15 分钟，321 通道）、traffic（小时，862 通道）、exchange（日频，8 通道），加 8 条确定性合成序列。
- 窗口：每条序列用最后 `context` 个点预测未来 64 点；多变量模式下同一时刻的 7 个通道一起 decode。
- 协变量：最后一个通道为目标，其余通道为 past-only，sin/cos 日历特征为 past-future。
- `nMAE`、`pinball` 按每条窗口上下文标准差归一化；`Δ` 为相对同窗口 fp32 的变化。
- `朴素基线`：每个窗口重复最后一个上下文值，九个分位取同一值；报告给出 fp32 相对该基线的差。
- `留出`：`sensitivity` / `kv-sensitivity` 的 `--offset N` 把最近 N 个窗口留给评测网格，探针使用更早的预测起点，两组不重叠。
- `cov err` 为 9 个十分位覆盖率与名义值之差的平均；`cross %` 为排序前的相邻十分位逆序比例。
- `KV MB` 按张量形状和比特数计算，`reduction` 相对 fp16 K/V；`peak MB` 为 MLX 峰值显存，含常驻权重。

### 权重量化

- `fp32` / `fp16` / `bf16`：整棵模块树转换 dtype。
- `int8_g64` / `int4_g64` / `int4_g32` / `int4_g128`：MLX affine 量化，输出头保持 fp32；`int4_g128` 跳过输入维不整除 128 的层。
- 混合精度按 `tfmq sensitivity` 的逐层结果保留末层、残差块或第 3 层为 8-bit。
- 裁剪：`int4c95_g64` 与 `mix_l18_19_8c95` 在量化前把每个输入分组裁到组内最大绝对值的 95%。

### KV 压缩

TimesFM 3.0 的 MLX 后端是单次全序列前向，没有跨调用 KV cache；这里压缩每层注意力的 K/V 工作集。

- `affine8` / `affine4`：MLX affine 量化，按头拼平后 group=64。
- `turbo3_mse` / `turbo2_mse` / `turbo3_prod`：PolarQuant / PolarQuant+QJL，参考 `lingengyuan/qjl-mlx` 的 MLX 实现，head_dim 补零到 128。
- `qjl1`：1-bit sign 估计，只压 K，V 保持 fp16。
- `K bits` / `V bits` 按原始 `head_dim` 折算：PolarQuant 补零到 128 的角度位、每个值的 fp32 范数、prod 的 QJL 位与投影都计入。
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
  parity.py          与 PyTorch 后端的数值对齐
  covariates.py      协变量输入下的量化与 KV
  horizon.py         更长 horizon 下的量化与 KV
  predict.py         CSV 预测命令
  attention.py       注意力补丁
  attention_compare.py  manual 与 SDPA 结果对比
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
- 多变量只测 ETT 的 7 通道同批；未覆盖变量数变化和变量子集。
- 协变量只测 24 条目标序列；past-only 用其他 ETT 通道，past-future 用日历特征，未覆盖真实外生变量。
- 跨域只取 Jena、Electricity、traffic、exchange 的前 8 个通道；exchange 不支持 15,360；horizon 探针只测 ETTh1、ctx 2,048。
- 吞吐为单次中位延迟。quick 预设重复 4 次，batch 8 平均吞吐 67.9–77.3 series/s；单个 (config, ctx, batch) 的 run-to-run 波动中位数 13%，最大 69%（短延迟档）。
- PyTorch 对照只在 CPU 上跑，未验证 MPS 路径。
- TimesFM 3.0 权重使用 non-commercial 许可证，本仓库不重新分发权重。

## 许可证

代码使用 MIT，见 [LICENSE](LICENSE)。TimesFM 3.0 权重不在本仓库内，遵循其 non-commercial 许可证。
