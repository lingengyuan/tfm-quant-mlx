# Changelog

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
