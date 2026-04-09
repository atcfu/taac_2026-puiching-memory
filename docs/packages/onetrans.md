# OneTrans

本文档记录两部分内容。

第一部分是依据论文整理的 OneTrans 方法说明，目的是把 unified tokenizer、mixed parameterization、causal attention 和 pyramid stack 的关键设计压成便于实现的 Markdown 版本。

第二部分是当前仓库中的 onetrans 实现选择，说明哪些部分已经按论文主线落下，哪些系统优化目前还没有接入。

如果想直接看完整论文整理版，可继续看 [OneTrans 论文拆解](../papers/onetrans.md)。

## 当前入口与验证

当前仓库把 OneTrans 作为目录式实验包维护：

```bash
uv run taac-train --experiment config/gen/onetrans
uv run taac-evaluate single --experiment config/gen/onetrans
```

默认数据集是 sample parquet，默认输出目录是 `outputs/gen/onetrans`。

当前可复核状态：

1. `tests/test_experiment_packages.py` 覆盖了 `test_experiment_package_builds_and_runs_forward`，其中包含 `config.gen.onetrans`。
2. 当前工作区已有一轮 10-epoch sample smoke，见 `outputs/smoke/onetrans/summary.json`。
3. 当前 10-epoch sample 结果的最佳 epoch 是 9，AUC 0.7476，PR-AUC 0.2849，平均时延 10.1586 ms / sample。

## 1. 论文主线

OneTrans 的目标是用单一 Transformer 主干同时完成序列建模和特征交互，从而替代传统的 encode-then-interaction 流水线。

论文把输入分成两类。

一类是 sequential features，也就是多条用户行为序列。

一类是 non-sequential features，也就是用户、候选物品、上下文等静态或请求级特征。

OneTrans 不先把行为压成一个向量再和静态特征拼接，而是先把两类特征都变成 token，然后拼成一个统一 token 序列：

X^(0) = [S-tokens; NS-tokens]

随后一组带因果 mask 的 OneTrans blocks 在这个统一序列上联合建模。这样模型在一个栈里同时获得：

1. 序列内部交互。
2. 多序列之间的交互。
3. 非序列特征之间的交互。
4. 序列与非序列之间的交互。

## 2. Unified Tokenizer

### 2.1 Non-Sequential Tokenization

论文给了两种方式把非序列特征变成 NS tokens。

1. Group-wise tokenizer：人工把特征分组后，每组一个 group-specific MLP。
2. Auto-Split tokenizer：先把全部 NS 特征拼接后过一个共享 MLP，再切分成多个 NS tokens。

论文 ablation 表明 Auto-Split tokenizer 是默认更优的选择，因此当前仓库默认实现也采用 Auto-Split。

### 2.2 Sequential Tokenization

论文允许多条行为序列并存。每条序列先把事件级输入映射到统一隐藏维度，然后再通过两种方式之一做融合。

1. Timestamp-aware fusion：按时间交错 interleave。
2. Timestamp-agnostic fusion：按行为强度或意图顺序拼接，并插入可学习的 [SEP]。

论文默认更偏向 timestamp-aware fusion；同时从图和 ablation 可以看出，分隔不同序列边界的 [SEP] token 也有帮助。

## 3. OneTrans Block

每个 OneTrans block 是 pre-norm 的 causal Transformer block，论文明确提到使用 RMSNorm、Mixed Causal Attention 和 Mixed FFN。

### 3.1 Mixed Causal Attention

论文不是让所有 token 共用同一套 QKV，而是采用 mixed parameterization。

1. 所有 sequential tokens 共用一套 Q/K/V 参数。
2. 每个 non-sequential token 使用自己专属的 Q/K/V 参数。

token 排列顺序固定为先 S-tokens、后 NS-tokens，并使用统一 causal mask。

这样 causal mask 会自然产生两种效果。

1. S-side 只能看过去的序列位置，完成序列建模。
2. NS-side 由于位于序列尾部，可以看到全部 S history 和之前的 NS tokens，从而把行为历史聚合到非序列侧。

### 3.2 Mixed FFN

FFN 也沿用同样的混合参数化策略。

1. sequential tokens 共享一个 FFN。
2. 每个 NS token 有自己的 token-specific FFN。

这保证了异质的非序列 token 不会被过度强迫共享参数，而同质的序列 token 仍然保留统一归纳偏置。

## 4. Pyramid Stack

论文提出的 pyramid stack 不是裁掉全部 token，而是只裁 sequential query tokens。

在每一层里：

1. keys 和 values 仍然由当前完整 token 序列计算。
2. queries 只保留最近的一段 S-tokens，再加全部 NS-tokens。
3. 注意力输出后只保留这段尾部 token，形成逐层缩短的金字塔。

论文强调这样做有两个收益。

1. Progressive distillation：长历史逐层蒸馏到更短的尾部表示。
2. Compute efficiency：FLOPs 和激活开销显著下降。

## 5. 训练与部署优化

论文在系统侧还做了几项优化。

1. Cross-request KV caching：同一请求的多个 candidate 共享 S-side 计算。
2. FlashAttention-2。
3. 混合精度训练。
4. 激活重计算。

这些优化对生产部署非常重要，但它们属于系统级扩展，不是 OneTrans block 的最小架构定义。

## 6. 当前仓库实现

当前实现放在 config/gen/onetrans 下，并保持数据、模型、损失私有。

### 6.1 当前已经实现的论文主线

1. Unified tokenizer：模型内部显式构造 S tokens 和 NS tokens，再拼成统一输入序列。
2. Auto-Split NS tokenizer：当前默认使用 Auto-Split，而不是人工 group-wise tokenizer。
3. Timestamp-aware 序列融合：序列侧以按时间合并后的 history 为基础，再结合 group id 和 [SEP] 边界信息。
4. Mixed parameterization：S tokens 共享 QKV/FFN，NS tokens 使用 token-specific QKV/FFN。
5. Causal attention：统一使用因果 mask，使 NS tokens 能聚合整个序列历史。
6. Pyramid stack：每层只保留一段最近的 sequential query tokens 和全部 NS tokens。

### 6.2 当前与论文仍有差距的部分

1. 当前数据管道仍然是仓库私有的哈希 token 方案，不是论文工业原始特征系统。
2. 当前没有接入 cross-request KV caching。
3. 当前没有接入 FlashAttention 和混合精度重计算这类系统优化。
4. 默认超参数是 smoke-scale，优先服务于当前仓库内的训练与回归验证，不是论文生产规模。

### 6.3 当前集成取舍

1. 为了与 TAAC_2026 当前数据契约兼容，S tokens 来自 history、post、author、action、time-gap、group-id 的统一事件表示，而不是论文中的工业原始特征 schema。
2. 为了落论文主线而不过早引入 serving 复杂度，本次集成只实现 unified backbone，不实现 KV caching。
3. 为了保持实验自包含，onetrans 的 data.py 和 utils.py 仍然私有放在本包内，而不是从其它实验包导入。
