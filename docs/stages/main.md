---
title: MAIN 阶段说明
icon: material/flag-outline
---

# MAIN 阶段说明

`config/main` 目前还不是一个已经落地的可执行阶段目录；当前主分支真正可运行的实验入口都在 `config/gen`。

这个目录保留的意义，是说明后续如果进入 MAIN 阶段，预期会做什么：

1. 把已经在 GEN 阶段验证过的结构与组件向共享底座收敛。
2. 在统一实现上做 scale up、对照实验和消融实验。
3. 补充更偏工程侧的能力，例如更稳健的 checkpoint 兼容策略、批量评估、AMP / CUDA 优化等。

当前状态可以概括为：方向已命名，分支实现尚未进入独立 MAIN 包维护阶段。
