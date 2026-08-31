# QuantSparseFlashMla

## 产品支持情况

| 产品                                                         | 是否支持 |
| :------------------------------------------------------------ | :------: |
|<term>Ascend 950PR/Ascend 950DT</term>                        | √  |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>        | ×  |
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>        | ×  |
|<term>Atlas 200I/500 A2 推理产品</term>                    | ×  |
|<term>Atlas 推理系列产品</term>                                | ×  |
|<term>Atlas 训练系列产品</term>                                | ×  |

## 功能说明

- 算子功能：

  `QuantSparseFlashMla`算子旨在完成全量化和稀疏场景下的MLA（Multi-head Latent Attention）注意力计算。该接口支持以下三类计算模式：
  - **SWA（Sliding Window Attention）**：仅传入`ori_kv`，对原始KV做滑动窗口注意力。
  - **CSA（Compressed Sparse Attention）**：同时传入`ori_kv`、`cmp_kv`和`cmp_sparse_indices`，对原始KV窗口和topK选择出的压缩KV共同做注意力。
  - **HCA（Heavily Compressed Attention）**：同时传入`ori_kv`和`cmp_kv`，对原始KV窗口和连续压缩KV段共同做注意力。

  `QuantSparseFlashMlaMetadata`是`QuantSparseFlashMla`的分核信息，在主算子执行前生成。当前版本主算子必须传入该metadata。典型调用流程如下：

  1. 根据调用场景准备对应输入。
  2. 调用`QuantSparseFlashMlaMetadata`生成`metadata`，作为`QuantSparseFlashMla`的入参。
  3. 调用`QuantSparseFlashMla`，生成计算结果。

- 计算公式：
  `QuantSparseFlashMla`采用MLA（Multi-head Latent Attention）对KV共享输入的稀疏注意力进行计算，其原理是对输入的KV进行选择性压缩与量化处理，再将Query与拼接后的KV计算结果通过Softmax得到注意力权重。

  MLA的计算公式一般定义如下，其中$\tilde{K}=\tilde{V}$为基于入参控制的实际参与计算的KV，由`ori_kv`的滑动窗口部分和`cmp_kv`的压缩部分共同组成，实际参与计算的KV范围由`cmp_ratio`、`ori_mask_mode`、`cmp_mask_mode`、`ori_win_left`、`ori_win_right`以及`cmp_sparse_indices`决定。

  $$
  O = \text{softmax}(Q@\tilde{K}^T \cdot \text{softmax\_scale})@\tilde{V}
  $$

## 参数说明

<table style="undefined;table-layout: fixed; width: 1576px">
  <colgroup>
  <col style="width: 220px">
  <col style="width: 150px">
  <col style="width: 700px">
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
      <td>q</td>
      <td>输入</td>
      <td>对应公式中的Q。</td>
      <td>HIFLOAT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>ori_kv</td>
      <td>可选输入</td>
      <td>对应公式中K和V的一部分，表示原始不经压缩的量化KV。</td>
      <td>HIFLOAT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>cmp_kv</td>
      <td>可选输入</td>
      <td>对应公式中K和V的一部分，表示经过压缩的量化KV。</td>
      <td>HIFLOAT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>q_descale</td>
      <td>输入</td>
      <td>q对应的量化参数。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>ori_kv_descale</td>
      <td>可选输入</td>
      <td>ori_kv对应的量化参数。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>cmp_kv_descale</td>
      <td>可选输入</td>
      <td>cmp_kv对应的量化参数。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>ori_sparse_indices</td>
      <td>可选输入</td>
      <td>表示从ori_kv中离散取数的索引，无效位置填-1。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>cmp_sparse_indices</td>
      <td>可选输入</td>
      <td>表示从cmp_kv中离散取数的索引，无效位置填-1。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>ori_block_table</td>
      <td>可选输入</td>
      <td>表示PageAttention中ori_kv使用的block映射表。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>cmp_block_table</td>
      <td>可选输入</td>
      <td>表示PageAttention中cmp_kv使用的block映射表。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>cu_seqlens_q</td>
      <td>可选输入</td>
      <td>表示TND布局下不同batch中q的累积序列长度。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>cu_seqlens_ori_kv</td>
      <td>可选输入</td>
      <td>表示TND布局下不同batch中ori_kv的累积序列长度。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>cu_seqlens_cmp_kv</td>
      <td>可选输入</td>
      <td>表示TND布局下不同batch中cmp_kv的累积序列长度。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>seqused_q</td>
      <td>可选输入</td>
      <td>表示不同batch中q实际参与计算的token数。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>seqused_ori_kv</td>
      <td>可选输入</td>
      <td>表示不同batch中ori_kv实际参与计算的token数。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>seqused_cmp_kv</td>
      <td>可选输入</td>
      <td>表示不同batch中cmp_kv实际参与计算的token数。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>cmp_residual_kv</td>
      <td>可选输入</td>
      <td>表示压缩KV余数，用于恢复cmp侧mask使用的压缩前KV长度。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>ori_topk_length</td>
      <td>可选输入</td>
      <td>用于标识ori_sparse_indices实际参与计算的长度。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>cmp_topk_length</td>
      <td>可选输入</td>
      <td>用于标识cmp_sparse_indices实际参与计算的长度。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>sinks</td>
      <td>可选输入</td>
      <td>表示各注意力头设置独立可学习虚拟偏移项，用于维持长文本推理时的稳定性。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>metadata</td>
      <td>可选输入</td>
      <td>QuantSparseFlashMlaMetadata生成的任务切分结果。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>quant_mode</td>
      <td>属性</td>
      <td>表示量化模式。量化模式1表示Q、K、V 为per-token量化，Q、K、V 数据类型为HIFLOAT8。</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>softmax_scale</td>
      <td>可选属性</td>
      <td>对应公式中的softmax_scale。</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cmp_ratio</td>
      <td>可选属性</td>
      <td>表示cmp_kv相对于压缩前KV长度的压缩倍率，用于恢复cmp侧mask使用的压缩前KV长度。默认值为1。</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>ori_mask_mode</td>
      <td>可选属性</td>
      <td>表示q和ori_kv计算的mask模式。<br/>0: No Mask。<br/>3: RightDownCausal模式。<br/>4: Band模式。</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cmp_mask_mode</td>
      <td>可选属性</td>
      <td>表示q和cmp_kv计算的mask模式。<br/>0: No Mask。<br/>3: RightDownCausal模式。</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>ori_win_left</td>
      <td>可选属性</td>
      <td>表示q和ori_kv计算中q对过去token计算的数量，支持-1或非负数，其中-1表示窗口不受限。</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>ori_win_right</td>
      <td>可选属性</td>
      <td>表示q和ori_kv计算中q对未来token计算的数量，支持-1或非负数，其中-1表示窗口不受限。</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layout_q</td>
      <td>可选属性</td>
      <td>表示输入q的数据排布格式，支持"BSND"和"TND"。</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layout_kv</td>
      <td>可选属性</td>
      <td>表示输入ori_kv和cmp_kv的数据排布格式，支持"BSND"、"TND"和"PA_BBND"。</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>topk_value_mode</td>
      <td>可选属性</td>
      <td>表示TopK索引取值模式。</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>return_softmax_lse</td>
      <td>可选属性</td>
      <td>表示是否返回softmax_lse。默认值False。</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>attn_out</td>
      <td>输出</td>
      <td>对应公式中的输出O。</td>
      <td>BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>softmax_lse</td>
      <td>可选输出</td>
      <td>返回softmax的log-sum-exp结果。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
  </tbody>
</table>

## 约束说明

- 该接口支持推理场景下使用。
- 该接口支持aclgraph模式。
- 该接口当前支持三种计算场景：SWA（Sliding Window Attention）场景仅传入`ori_kv`；CSA（Compressed Sparse Attention）场景同时传入`ori_kv`、`cmp_kv`及`cmp_sparse_indices`；HCA（Heavily Compressed Attention）场景同时传入`ori_kv`及`cmp_kv`；
- 通用规格约束如下：
  - N2仅支持1，D仅支持512。
  - `cmp_ratio`表示`cmp_kv`相对于压缩前KV长度的压缩倍率；仅传入`ori_kv`时，`cmp_ratio`不参与压缩KV计算，需保持默认值1；支持1到128。
  - `ori_mask_mode`支持0/3/4，`cmp_mask_mode`支持0/3，`ori_win_left`支持-1或非负数，`ori_win_right`支持-1或非负数，只有`ori_mask_mode`为4时，`ori_win_left`和`ori_win_right`可以>=0。
  - `layout_q`和`layout_kv`组合仅支持"BSND"/"BSND"、"TND"/"TND"、"BSND"/"PA_BBND"、"TND"/"PA_BBND"；非PA_BBND场景下`layout_q`和`layout_kv`必须一致；PA_BBND场景下`block_size`支持1到1024。
  - 全平台均不支持传入非空Tensor。

- 当`layout_q`为TND时，功能使用限制如下：
  - `q`的shape需要为[T1,N1,D]。
  - `ori_sparse_indices`的shape维度为[Q\_T, KV\_N, K1]，其中K1为对`ori_kv`一次离散选取的token数。
  - `cmp_sparse_indices`的shape需要为[Q\_T, KV\_N, K2]，其中K2为对`cmp_kv`一次离散选取的token数。
  - `cu_seqlens_q`必须传入，输入维度为B+1，大小为参数中每个元素的值表示当前batch与之前所有batch的token数总和，即前缀和，因此后一个元素的值必须>=前一个元素的值且首元素必须为0。

- 当`layout_q`为BSND时，功能使用限制如下：
  - `q`的shape需要为[B, Q\_S, Q\_N, D]。
  - `ori_sparse_indices`的shape需要为[B, Q\_S, KV\_N, K1]，其中K1为对`ori_kv`一次离散选取的token数。
  - `cmp_sparse_indices`的shape需要为[B, Q\_S, KV\_N, K2]，其中K2为对`cmp_kv`一次离散选取的token数。

- PageAttention场景下，功能使用限制如下：
  - `ori_kv`和`cmp_kv`的shape分别为[ori\_block\_num, ori\_block\_size, KV\_N, D]和[cmp\_block\_num, cmp\_block\_size, KV\_N, D]，其中ori\_block\_num和cmp\_block\_num为PageAttention时block总数，ori\_block\_size和cmp\_block\_size为一个block的token数，ori\_block\_size和cmp\_block\_size取值为1到1024，KV_N仅支持1。
  - `ori_block_table`和`cmp_block_table`的shape为2维，其中第一维长度为B，第二维长度不小于所有batch中最大的S2和S3对应的block数量，即S2\_max / block\_size和S3\_max / block\_size向上取整。
- `metadata`为算子实际需要使用的分核结果，目前该参数必传，shape大小固定为[1024]。
- `layout_kv`支持输入"BSND"、"TND"和"PA_BBND"，需满足上述`layout_q`和`layout_kv`组合约束。
  - 当输入为PA_BBND时，`seqused_ori_kv`和`ori_block_table`必须传入；当输入为BSND时，`seqused_ori_kv`可用于表达每个batch的`ori_kv`有效长度；当输入为TND时，`ori_kv`有效长度由`cu_seqlens_ori_kv`表达。
  - 当输入为BSND时，`ori_kv`和`cmp_kv`的layout都必须为BSND，ori_kv的shape为[B, S2, N2,D]，cmp_kv的shape为[B, S3, N2,D]。
  - 当输入为TND时，`cu_seqlens_ori_kv`必须传入；若存在`cmp_kv`，`cu_seqlens_cmp_kv`也必须传入。
- `return_softmax_lse`为False时返回占位Tensor；为True时返回softmax的log-sum-exp结果。
- 传入Tensor不支持为空。
- `seqused_cmp_kv`为所有`layout_kv`下的可选输入，显式传入时用于覆盖cmp侧逻辑有效长度；未传时由`cmp_kv` shape、`cu_seqlens_cmp_kv`或PA block table相关语义推导。
- `cmp_residual_kv`为算子的可选入参；传入后用于按`cmp_len * cmp_ratio + residual`恢复cmp侧mask使用的压缩前KV长度，其中`cmp_len`优先来自显式传入的`seqused_cmp_kv`。
- `q`、`ori_kv`、`cmp_kv`数据排布格式支持从多种维度解读，B（Batch）表示输入样本批量大小、S（Seq-Length）表示输入样本序列长度、H（Hidden-Size）表示隐藏层的大小、N（Head-Num）表示多头数、D（Head-Dim）表示hidden层最小的单元尺寸，且满足D=H/N、T表示所有Batch输入样本序列长度的累加和。
- Q\_S和S1表示q shape中的S，S2表示ori_kv shape中的S，S3表示cmp_kv shape中的S；Q\_N和N1表示num\_q\_heads，KV\_N和N2表示num\_ori_kv\_heads和num\_cmp_kv\_heads；Q\_T和T1表示q shape中的输入样本序列长度的累加和。

## 调用说明

| 调用方式  | 样例代码                                                     | 说明                                                         |
| --------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| aclnn API | [test_aclnn_quant_sparse_flash_mla](./examples/test_aclnn_quant_sparse_flash_mla.cpp) | 通过[aclnnQuantSparseFlashMla](./docs/aclnnQuantSparseFlashMla.md)调用QuantSparseFlashMla算子 |
| PyTorch API | [quant_sparse_flash_mla](../../torch_extension/cann_ops_transformer/docs/zh/quant_sparse_flash_mla.md) | 通过`cann_ops_transformer.quant_sparse_flash_mla`调用QuantSparseFlashMla算子 |
