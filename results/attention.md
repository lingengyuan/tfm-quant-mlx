# 注意力后端对比（manual vs SDPA）

- manual：`results/attention_manual.json`
- sdpa：`results/attention_sdpa.json`
- Δ 为 SDPA 相对 manual 的变化，正值表示 SDPA 更差。

| 配置 | 数据集 | 上下文 | manual pinball | sdpa pinball | Δpinball | manual nMAE | sdpa nMAE | ΔnMAE |
|---|---|---|---|---|---|---|---|---|
| fp32 | ETTh1 | 2048 | 0.1492 | 0.1492 | -0.01% | 0.3849 | 0.3848 | -0.01% |
| fp32 | ETTh1 | 15360 | 0.1373 | 0.1373 | -0.00% | 0.3513 | 0.3513 | -0.00% |
| mix_l18_19_8 | ETTh1 | 2048 | 0.1531 | 0.1530 | -0.02% | 0.3956 | 0.3955 | -0.01% |
| mix_l18_19_8 | ETTh1 | 15360 | 0.1384 | 0.1384 | -0.00% | 0.3519 | 0.3519 | -0.00% |
| affine4 | ETTh1 | 15360 | 0.1374 | 0.1380 | +0.47% | 0.3504 | 0.3527 | +0.65% |
| affine4_l19fp16 | ETTh1 | 15360 | 0.1364 | 0.1374 | +0.76% | 0.3491 | 0.3510 | +0.52% |
