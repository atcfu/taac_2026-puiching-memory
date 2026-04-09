# Symbiosis

## 来源

`symbiosis` 是当前仓库里第一个明确按“闭源本地融合方案”思路设计的实验包，不对应单一公开 repo，而是主动吸收三个当前高分骨干的优点：

1. `unirec` 的 unified sequence tokenization 与混合序列建模。
2. `uniscaleformer` 的 memory compression 与 query-centric 聚合。
3. `onetrans` 的 non-sequential token auto-split 与紧凑的统一交互视角。

目标不是复刻某篇论文，而是在不破坏当前 parquet batch 契约的前提下，做一个面向 AUC 的多骨干协同模型。

## 结构

当前目录已经升级为可执行实验包，并满足 `config/gen` 的标准契约：

1. `__init__.py` 导出 `EXPERIMENT`。
2. `data.py` 复用当前最稳定的 baseline 数据管线。
3. `model.py` 内部同时实例化 `UniRec`、`UniScaleFormer`、`OneTrans` 三个骨干，再用样本级路由与校准头做融合。
4. `utils.py` 复用 `unirec` 已验证过的 ranking loss + optimizer 栈。

其中融合头不是简单固定平均，而是按样本的序列密度、上下文长度、候选特征覆盖率、`item_logq` 与各骨干 logit 分歧动态分配权重。

## 运行命令

```bash
uv run taac-train --experiment config/gen/symbiosis
uv run taac-evaluate single --experiment config/gen/symbiosis
```

如需把产物写到 smoke 目录：

```bash
uv run taac-train --experiment config/gen/symbiosis --run-dir outputs/smoke/symbiosis
```

## 当前验证状态

1. 当前已接入目录式实验包契约。
2. 当前应当至少通过 package build + forward regression。
3. smoke AUC 以实际训练产物为准；如果还没有对应 `summary.json`，不要在文档里补猜测值。