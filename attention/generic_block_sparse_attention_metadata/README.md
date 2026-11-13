# GenericBlockSparseAttentionMetadata

## 产品支持情况

- <term>Ascend 950PR&950DT系列产品</term>：支持
- <term>Atlas A3系列产品</term>：支持
- <term>Atlas A2系列产品</term>：支持
- <term>Atlas 200I/500 A2推理产品</term>：不支持
- <term>Atlas推理系列产品</term>：不支持
- <term>Atlas训练系列产品</term>：不支持

## 功能说明

- 算子功能：

  `GenericBlockSparseAttentionMetadata`是`GenericBlockSparseAttention`的前置算子，根据Query序列长度、Q/KV头数和稀疏块信息生成主算子使用的分核结果`metadataOptional`。本算子不执行Attention计算。

  典型调用流程如下：

  1. 准备`sparseBlockIdx`、`sparseBlockCount`、序列长度和属性等输入。
  2. 调用`aclnnGenericBlockSparseAttentionMetadata`生成`metadataOptional`。
  3. 调用`aclnnGenericBlockSparseAttention`，将上一步得到的`metadataOptional`作为主算子输入。

## 参数说明

> **说明：**<br>
> B表示Batch Size，`totalQBlocks`表示按各Batch的Query存储长度划分后的Q块总数，`maxSparseBlockCount`表示`sparseBlockIdx`的最后一维。

<table style="undefined;table-layout: fixed; width: 1200px"><colgroup>
  <col style="width: 220px">
  <col style="width: 130px">
  <col style="width: 550px">
  <col style="width: 180px">
  <col style="width: 100px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出/属性</th>
      <th>描述</th>
      <th>数据类型</th>
      <th>数据格式</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>sparseBlockIdx</td>
      <td>输入</td>
      <td>稀疏块索引。当前shape为[numKeyValueHeads, totalQBlocks, maxSparseBlockCount]。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>sparseBlockCount</td>
      <td>输入</td>
      <td>每个Q块实际选择的KV块数量，当前shape为[numKeyValueHeads, totalQBlocks]。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>cuSeqLengthsQOptional</td>
      <td>输入</td>
      <td>各Batch中Query序列存储长度的前缀和，TND必须传入；BSND/BNSD不使用，建议传入nullptr，shape为(B+1,)。</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>cuSeqLengthsKvOptional</td>
      <td>输入</td>
      <td>各Batch中Key/Value序列存储长度的前缀和，layoutKv为"TND"时必传，PA_BBND/PA_BNBD时不传；BSND/BNSD时不使用，建议传入nullptr，shape为(B+1,)。</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>sequsedQOptional</td>
      <td>条件必选输入</td>
      <td>各Batch中Query的实际序列长度，shape为(B,)；BSND/BNSD必传，TND可选且不传时使用Query存储长度。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>sequsedKvOptional</td>
      <td>输入</td>
      <td>各Batch中Key/Value的实际序列长度，shape为(B,)；PA_BBND/PA_BNBD及BSND/BNSD KV必传，TND KV可选。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>maxQSeqLen</td>
      <td>属性</td>
      <td>Query最大序列长度；TND允许-1（未指定）或非负值；BSND/BNSD必须等于shape容量Sq且大于0。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>maxKvSeqLen</td>
      <td>属性</td>
      <td>Key/Value最大序列长度；允许-1（未指定）或非负值，不用于推导BSND/BNSD的实际KV长度。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>numQHeads</td>
      <td>属性</td>
      <td>主算子Query的head数。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>numKvHeads</td>
      <td>属性</td>
      <td>主算子Key/Value的head数。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>headDim</td>
      <td>属性</td>
      <td>主算子Query、Key和Value每个head的特征维度。headDim取值范围见<a href="#产品差异约束">产品差异约束</a>。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>blockShape</td>
      <td>属性</td>
      <td>稀疏块形状[blockShapeX, blockShapeY]。非union为[1,Y]；union（maskType=3）为[8,Y]（当前仅支持8）。须与主算子一致。<term>Ascend 950PR&950DT系列产品</term>非量化与常规 FP8（quantType=5，非union）支持1≤blockShapeY≤128的任意整数；MXFP8与主算子配套时blockShapeY仅支持4或8；Atlas A2/A3非量化支持1、2、4、8、16、32、64、128。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layoutSparsePattern</td>
      <td>属性</td>
      <td>稀疏pattern布局，当前仅支持4；同一KV head对应的Q heads共享稀疏pattern。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layoutQ</td>
      <td>属性</td>
      <td>主算子Query的数据排布格式，支持"TND"、"BSND"和"BNSD"，适用范围见<a href="#产品差异约束">产品差异约束</a>。</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layoutKv</td>
      <td>属性</td>
      <td>主算子Key/Value的数据排布格式，支持"PA_BBND"、"PA_BNBD"、"TND"、"BSND"或"BNSD"，适用范围见<a href="#产品差异约束">产品差异约束</a>。</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>maskType</td>
      <td>属性</td>
      <td>Attention计算的掩码类型，支持0（非因果）、1（causal）；Ascend 950 及非量化的 Atlas A2/A3 另支持3（union），适用范围见<a href="#产品差异约束">产品差异约束</a>。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>quantType</td>
      <td>属性</td>
      <td>量化类型，支持0、5或6，适用范围见<a href="#产品差异约束">产品差异约束</a>。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>softmaxPrecision</td>
      <td>属性</td>
      <td>Softmax计算精度级别，取值为0或1。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>winLeft</td>
      <td>属性</td>
      <td>滑窗Attention向前包含的token数，当前仅支持-1。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>winRight</td>
      <td>属性</td>
      <td>滑窗Attention向后包含的token数，当前仅支持-1。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>residualBlockMode</td>
      <td>属性</td>
      <td>0：尾部不完整块由显式稀疏列表决定；1：隐式加入尾部不完整块，该块不包含在显式列表中。取值的适用范围见<a href="#产品差异约束">产品差异约束</a>。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>isConsistentTopk</td>
      <td>属性</td>
      <td>同一batch、同一head内，各Q块选择的KV块最大数量是否一致。取值false或true，须与主算子/Metadata保持一致。</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>metadataOptional</td>
      <td>输出</td>
      <td>主算子使用的分核结果，shape固定为(1024,)。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
  </tbody>
</table>

- 稀疏块形状[blockShapeX, blockShapeY]：非union时X=1；union（maskType=3）时X=8。<term>Atlas A2系列产品</term>和<term>Atlas A3系列产品</term>的Y支持1、2、4、8、16、32、64、128。
- <term>Ascend 950PR&950DT系列产品</term>非量化与常规 FP8：`X=1`，`1≤Y≤128`；Union：`[8,Y]`（`1≤Y≤128`）。`MXFP8`（`quantType=6`）仅支持`[1,4]`或`[1,8]`。Union 不支持 `quantType=6`。


## 约束说明

- 本算子必须与`GenericBlockSparseAttention`配套使用。所有对应输入和属性须与随后调用的主算子一致，每次调用主算子前均须重新生成`metadataOptional`。
- 布局组合及适用范围见[产品差异约束](#产品差异约束)。`layoutSparsePattern=4`；非union时`blockShapeX=1`，union时`blockShapeX=8`；headDim范围见[产品差异约束](#产品差异约束)。
- `numQHeads >= numKvHeads`且`numQHeads % numKvHeads == 0`，`numQHeads / numKvHeads`不超过128。
- `sparseBlockIdx`最后一维须不小于`sparseBlockCount`中所有元素的最大值，且须大于0。容量上限：<term>Atlas A2系列产品</term>和<term>Atlas A3系列产品</term>非量化为INT32_MAX，<term>Ascend 950PR&950DT系列产品</term>非量化FLOAT16/BFLOAT16、MXFP8及常规 FP8（含 FP8 Union）为INT32_MAX。
- 每组有效稀疏索引须非负、严格升序且不重复，位于前`sparseBlockCount`个位置；随机升序即可，不要求等步长或物理页连续。
- <term>Atlas A2系列产品</term>和<term>Atlas A3系列产品</term>在`residualBlockMode=1`时显式列表仅包含当前可见范围内的完整KV块，不包含隐式尾块。
- 输出`metadataOptional`是固定为INT32、shape为(1024,)的不透明数据，不得解析、修改或跨不同输入和属性复用。
- 完整约束见[aclnnGenericBlockSparseAttentionMetadata](./docs/aclnnGenericBlockSparseAttentionMetadata.md)。

### 产品差异约束

- <term>Ascend 950PR&950DT系列产品</term>非量化FLOAT16/BFLOAT16场景：`headDim`支持1～256的任意值；`layoutQ/layoutKv`支持`TND/PA_BBND`、`TND/PA_BNBD`、`TND/TND`、`BSND/BSND`、`BNSD/BNSD`、`BSND/PA_BBND`、`BNSD/PA_BBND`、`BSND/PA_BNBD`和`BNSD/PA_BNBD`，不支持其他组合；`maskType`支持0（非因果）、1（causal）和3（union）；非union时`blockShape=[1,Y]`（`1≤Y≤128`），union时`blockShape=[8,Y]`；`residualBlockMode`支持0或1（union仅支持0）。
- <term>Atlas A2系列产品</term>和<term>Atlas A3系列产品</term>非量化FLOAT16/BFLOAT16场景：`headDim`支持1～256的任意值；`layoutQ/layoutKv`支持`TND/PA_BBND`和`TND/TND`；`blockShape=[1,Y]`，Y支持1、2、4、8、16、32、64、128；`maskType`支持0（非因果）、1（causal）和3（union，见下方Union约束）；`residualBlockMode`支持0或1。TND KV须传入`cuSeqLengthsKvOptional`；causal模式下各batch须满足实际KV长度不小于实际Q长度。
- <term>Atlas A2系列产品</term>和<term>Atlas A3系列产品</term> Union（`maskType=3`）支持 FP16/BF16 非量化、`blockShape=[8,Y]`、`residualBlockMode=0`。Y 支持 1、2、4、8、16、32、64、128，分页 KV 的页大小为 128。稀疏索引按各 batch 有效 Q 长度分组，须与主算子的 Union 参数一致。
- <term>Ascend 950PR&950DT系列产品</term> Union（`maskType=3`）支持 FP16/BF16 和 FP8；`blockShape=[8,Y]`（`1≤Y≤128`），`residualBlockMode=0`；FP8 Union 另要求 `headDim=128`。Union 不支持 `quantType=6`。须与主算子一致。
- <term>Ascend 950PR&950DT系列产品</term> `MXFP8`（`quantType=6`）须与当前主算子的grouped路径配套：`headDim=192/256`、`blockShape=[1,4]`或`[1,8]`、`numQHeads/numKvHeads=1～128`；Q为`TND/BSND/BNSD`，KV为同布局非分页输入或`PA_BBND/PA_BNBD`（页大小128）。支持`maskType=0/1`、`softmaxPrecision=0/1`，主算子不输出LSE；`residualBlockMode=0`。完整使用条件见[主算子产品差异约束](../generic_block_sparse_attention/README.md#产品差异约束)。Metadata接口直接拒绝其他`headDim`或稀疏块大小，返回`ACLNN_ERR_PARAM_INVALID`。Metadata参数校验通过不替代主算子的scale、地址及缓冲容量检查。
- `quantType=6`的Q/K使用group-32、V使用per-channel、P使用per-tensor缩放；scale的shape和dtype由主算子校验，metadata不读取scale。满足形状、任务数及工作量条件时，Metadata可生成Flash Decoding分裂调度；工作量根据实际有效稀疏块计数与`blockShapeY`估算，工作量不足时保留常规调度。
- <term>Ascend 950PR&950DT系列产品</term> 常规 FP8（quantType=5，maskType≠3）：`headDim`仅支持128，`layoutQ/layoutKv`仅支持`TND/PA_BBND`，`maskType=1`、`residualBlockMode=0`、`blockShape=[1,Y]`（`1≤Y≤128`）。FP8 Union 见上条 Union 约束；Union 不支持 `quantType=6`。
- 稀疏索引最后一维容量：<term>Atlas A2系列产品</term>和<term>Atlas A3系列产品</term>非量化支持1～INT32_MAX，<term>Ascend 950PR&950DT系列产品</term>非量化FLOAT16/BFLOAT16、MXFP8及常规 FP8（含 FP8 Union）支持1～INT32_MAX；容量均须不小于稀疏块有效计数的最大值。
- `quantType`：<term>Ascend 950PR&950DT系列产品</term>支持0（非量化）、5（FP8量化）或6（MXFP8）；<term>Atlas A2系列产品</term>和<term>Atlas A3系列产品</term>仅支持0。
- `softmaxPrecision`：<term>Ascend 950PR&950DT系列产品</term>非Union场景（maskType=0/1）下，非量化FLOAT16/BFLOAT16和`MXFP8`支持0或1，quantType=5的FP8量化仅支持1；Union场景（maskType=3）下非量化与FP8均仅支持0；<term>Atlas A2系列产品</term>和<term>Atlas A3系列产品</term>上FLOAT16支持0或1，BFLOAT16仅支持0，Union仅支持0。

## 调用说明

| 调用方式 | 样例代码 | 说明 |
| -------- | -------- | ---- |
| aclnn API | [test_aclnn_generic_block_sparse_attention_metadata](./examples/test_aclnn_generic_block_sparse_attention_metadata.cpp) | 通过[aclnnGenericBlockSparseAttentionMetadata](./docs/aclnnGenericBlockSparseAttentionMetadata.md)两段式接口调用GenericBlockSparseAttentionMetadata算子。 |
