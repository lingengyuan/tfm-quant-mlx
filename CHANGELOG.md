# Changelog

## 0.1.7 — 2026-11-03

- 修正协变量目标掩码缺少 variate 维：`context=65`、`batch=2` 可正常解码。
- 网格聚合跳过逐窗口数组，避免数据集窗口数不同导致 `IndexError`。
- `convert` 改用 `mx.load` 读取权重，bf16 导出可重新加载。
- `--offset` 配合 `window_mode: tail` 形成不重叠的留出区块。
- 配对 CI 按预测时点重采样，并按数据集等权聚合，与主表口径一致。
- `attention-compare` 校验配置、后端与窗口数，条件不一致时拒绝比较。
- `--time` 指定的数值时间列不再被当作序列；未来行的时间戳用于日历特征。

## 0.1.6 — 2026-11-03

- 新增 `tfmq convert`：导出 packed 量化权重、量化 spec 与模型架构；加载跳过 fp32 权重与重新量化，spec 不匹配时报错。实测 int8 导出 375 MB，预测差为 0。
- `predict --covariates` 新增 `--target`、`--time`、`--future-covariates`；缺失值改为掩码传入，不再跨列删行。
- 评估结果保存逐窗口 pinball；报告新增朴素基线与配对 bootstrap 95% CI。
- 新增 ETTh2 权重与 KV 留出验证。
- 重跑 full 网格以生成逐窗口数据。

## 0.1.5 — 2026-11-03

- `tfmq profile` 的 GEMM 行按配置生成实际 shape：batch 8 × 482 token = 3,856 行，同时保留 482 与 15,424 行对照。
- GEMM 三档交错计时（每档 15 次），减小热漂移对比较的影响；保留每次采样。
- `profile.json` 保存每档原始计时、git 提交与脏状态、checkpoint 文件哈希、合成输入哈希与环境信息。

## 0.1.4 — 2026-11-03

- 新增 `attention-compare` 命令与 `configs/attention_manual.yaml`、`configs/attention_sdpa.yaml`。
- 实测无 KV 时 manual / SDPA 差 ≤0.02%；KV 压缩后为 +0.47%（`affine4`）与 +0.76%（第 19 层 fp16）。
- `sensitivity` / `kv-sensitivity` 新增 `--datasets` 与 `--no-synthetic`，支持按数据集留出验证。
- ETTh2 留出验证：第 19 层策略未复现，保留第 19 层回收 0.45 pp（ETTh1 为 0.95 pp）。

## 0.1.3 — 2026-11-03

- CSV 预测按序列长度合批（`--batch`，默认 8），输出顺序不变；batch 1 与 batch 8 输出一致。
- 新增 `attention` 配置与 `--attention sdpa`，预测和网格共用；默认 `manual`，SDPA 与手写路径不保证逐位一致。
- 评估结果保存朴素基线（重复最后值），报告新增基线表。
- `sensitivity` / `kv-sensitivity` 新增 `--offset`，把最近窗口留给评测网格。
- 预测参数校验提前到加载模型之前。

## 0.1.2 — 2026-11-03

- 预测输入与输出增加有限值检查：`--multi` / `--covariates` 按共同有限行取窗口；非有限预测直接报错，不再写出无效 CSV。
- 删除性能探针中重复的 `fp16_compute` 档（与 `fp16` 配置相同），并重跑 `results/profile.json`。
- 修正 README 逐层 KV 表的旧体积（127 / 141 MB）。

## 0.1.1 — 2026-11-03

- 修正 KV 压缩器的位宽与重构：PolarQuant 按补零后的编码计位并折算到原始 `head_dim`；QJL 与 prod 残差使用 `sqrt(pi/2/d)` 缩放，零输入重构为零。
- 修正多变量与协变量输入的时间对齐：按共同有效行取窗口，协变量目标列不再被静默替换。
- 修正 `--config` 忽略 `--out` 等命令行覆盖；fp32 在权重网格中始终先跑。
- 修正 PyTorch 对照的分位排序与更长 horizon 的窗口一致性。
- 报告跳过失败运行，并在说明中单列。
- 上述修正改变 KV 位宽、QJL、horizon 与 parity 的数值；全部网格已重跑。

## 0.1.0 — 2026-11-02

- 权重：fp16 / bf16、int8、int4（group 32/64/128）、per-group 裁剪、混合精度。
- KV：affine、PolarQuant / TurboQuant、QJL；按层与按注意力类型的混合精度。
- 评测：ETT、Jena Climate、UCI Electricity、traffic、exchange、合成数据；上下文 512–15,360；horizon 64–1,024；多变量与协变量输入。
- 工具：`tfmq run`、`report`、`sensitivity`、`kv-sensitivity`、`profile`、`parity`、`covariates`、`predict`、`horizon`。
- 与 PyTorch 后端的数值对照。
- CI：不依赖 checkpoint 的测试，运行在 macOS arm64。
