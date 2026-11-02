# Changelog

## 0.1.0 — 2026-11-02

- 权重：fp16 / bf16、int8、int4（group 32/64/128）、per-group 裁剪、混合精度。
- KV：affine、PolarQuant / TurboQuant、QJL；按层与按注意力类型的混合精度。
- 评测：ETT、Jena Climate、UCI Electricity、traffic、exchange、合成数据；上下文 512–15,360；horizon 64–1,024；多变量与协变量输入。
- 工具：`tfmq run`、`report`、`sensitivity`、`kv-sensitivity`、`profile`、`parity`、`covariates`、`predict`、`horizon`。
- 与 PyTorch 后端的数值对照。
- CI：不依赖 checkpoint 的测试，运行在 macOS arm64。
