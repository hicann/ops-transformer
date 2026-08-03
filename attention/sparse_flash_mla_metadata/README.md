# SparseFlashMlaMetadata

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Ascend 950PR/Ascend 950DT</term>                        | √  |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>        | √  |
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>        | √  |
|<term>Atlas 200I/500 A2推理系列产品</term>                    | ×  |
|<term>Atlas 推理系列产品</term>                                | ×  |
|<term>Atlas 训练系列产品</term>                                | ×  |

## 功能说明

- 算子功能：`SparseFlashMlaMetadata`算子完成`SparseFlashMla`算子的tiling计算，包含每个AIcore的Attention计算任务的起止点的Batch、Head、以及Q和K的分块的索引，供后续`SparseFlashMla`算子使用。
- 场景简称：SWA（Sliding Window Attention）、CSA（Compressed Sparse Attention）、HCA（Heavily Compressed Attention）。

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
      <td>num_heads_q</td>
      <td>属性</td>
      <td>表示`q`的头数，支持[1, 128]。</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>num_heads_kv</td>
      <td>属性</td>
      <td>表示`ori_kv`和`cmp_kv`的头数，仅支持1。</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>head_dim</td>
      <td>属性</td>
      <td>注意力头的维度，仅支持512。</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cu_seqlens_q</td>
      <td>可选输入</td>
      <td>表示TND布局下不同batch中`q`的累积序列长度，shape为(B+1, )。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>cu_seqlens_ori_kv</td>
      <td>可选输入</td>
      <td>表示TND布局下不同batch中`ori_kv`的累积序列长度，shape为(B+1, )。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>cu_seqlens_cmp_kv</td>
      <td>可选输入</td>
      <td>表示TND布局下不同batch中`cmp_kv`的累积序列长度，shape为(B+1, )。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>seqused_q</td>
      <td>可选输入</td>
      <td>表示不同batch中`q`实际参与计算的token数，shape为(B, )。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>seqused_ori_kv</td>
      <td>可选输入</td>
      <td>表示不同batch中`ori_kv`实际参与计算的token数，shape为(B, )。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>seqused_cmp_kv</td>
      <td>可选输入</td>
      <td>表示不同batch中`cmp_kv`实际参与计算的token数，shape为(B, )。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>cmp_residual_kv</td>
      <td>可选输入</td>
      <td>表示压缩KV余数，用于恢复cmp侧mask使用的压缩前KV长度，shape为(B, )。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>ori_topk_length</td>
      <td>可选输入</td>
      <td>表示不同q token对应的ori_kv部分关键稀疏token的个数，shape为(B, S1, N2)或(T1, N2)。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>cmp_topk_length</td>
      <td>可选输入</td>
      <td>表示不同q token对应的cmp_kv部分关键稀疏token的个数，shape为(B, S1, N2)或(T1, N2)。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>batch_size</td>
      <td>可选属性</td>
      <td>表示输入样本批量大小；传入0时表示由接口推导，默认值为0。</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>max_seqlen_q</td>
      <td>可选属性</td>
      <td>表示所有batch中`q`的最大有效token数；传入0时表示由接口推导，默认值为0。</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>max_seqlen_ori_kv</td>
      <td>可选属性</td>
      <td>表示所有batch中`ori_kv`的最大有效token数；传入0时表示由接口推导，默认值为0。</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>max_seqlen_cmp_kv</td>
      <td>可选属性</td>
      <td>表示所有batch中`cmp_kv`的最大有效token数；传入0时表示由接口推导，默认值为0。</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>ori_topk</td>
      <td>可选属性</td>
      <td>表示从`ori_kv`中筛选出的关键稀疏token个数，默认值为0。</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cmp_topk</td>
      <td>可选属性</td>
      <td>表示从`cmp_kv`中筛选出的关键稀疏token个数，默认值为0。</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cmp_ratio</td>
      <td>可选属性</td>
      <td>表示`cmp_kv`相对于压缩前KV长度的压缩倍率，用于恢复cmp侧mask使用的压缩前KV长度；仅传入`ori_kv`时不参与压缩KV计算。支持[1, 128]，默认值为1。</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>ori_mask_mode</td>
      <td>可选属性</td>
      <td>表示`q`和`ori_kv`计算的mask模式，默认值为0。<br/>0: No Mask。<br/>3: RightDownCausal模式。<br/>4: Band模式。</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cmp_mask_mode</td>
      <td>可选属性</td>
      <td>表示`q`和`cmp_kv`计算的mask模式，默认值为0。<br/>0: No Mask。<br/>3: RightDownCausal模式。</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>ori_win_left</td>
      <td>可选属性</td>
      <td>表示`q`和`ori_kv`计算中`q`对过去token计算的数量，支持-1或非负数，其中-1表示窗口不受限，默认值为-1。</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>ori_win_right</td>
      <td>可选属性</td>
      <td>表示`q`和`ori_kv`计算中`q`对未来token计算的数量，支持-1或非负数，其中-1表示窗口不受限，默认值为-1。</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layout_q</td>
      <td>可选属性</td>
      <td>表示输入`q`的数据排布格式，支持"BSND"和"TND"，默认值为"BSND"。</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layout_kv</td>
      <td>可选属性</td>
      <td>表示输入`ori_kv`和`cmp_kv`的数据排布格式，支持"BSND"、"TND"和"PA_BBND"，默认值为"BSND"。</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>has_ori_kv</td>
      <td>可选属性</td>
      <td>表示`SparseFlashMla`主算子是否传入`ori_kv`，默认值为true。</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>has_cmp_kv</td>
      <td>可选属性</td>
      <td>表示`SparseFlashMla`主算子是否传入`cmp_kv`，默认值为true。</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>metadata</td>
      <td>输出</td>
      <td>表示`SparseFlashMla`主算子使用的任务切分结果，shape固定为(1024, )。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
  </tbody>
</table>
<ul>
  <li><term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> ：num_heads_q/num_heads_kv仅支持1、2、4、8、16、32、64、128，不支持seqused_q、ori_topk_length、cmp_topk_length，ori_topk仅支持0，cmp_topk仅支持0、512、1024，ori_mask_mode仅支持4，cmp_mask_mode仅支持3，ori_win_left仅支持127，ori_win_right仅支持0，cmp_ratio仅支持1、4、128。</li>
  <li><term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> ：num_heads_q/num_heads_kv仅支持1、2、4、8、16、32、64、128，不支持seqused_q、ori_topk_length、cmp_topk_length，ori_topk仅支持0，cmp_topk仅支持0、512、1024，ori_mask_mode仅支持4，cmp_mask_mode仅支持3，ori_win_left仅支持127，ori_win_right仅支持0，cmp_ratio仅支持1、4、128。</li>
</ul>

## 约束说明

- 该接口支持训练、推理场景下使用。
- 该接口支持aclgraph模式。
- 通用规格约束如下：
  - B（Batch）表示输入样本批量大小。
  - 参数`cu_seqlens_q`、`cu_seqlens_ori_kv`及`cu_seqlens_cmp_kv`要求其值为当前Batch与前序Batch有效token数的累加值，第一个元素固定为0，后一个元素的值必须大于等于前一个元素的值。
  - 参数`seqused_q`、`seqused_ori_kv`、`seqused_cmp_kv`要求其值表示每个Batch中的有效token数。
  - `layout_q`和`layout_kv`组合仅支持"BSND"/"BSND"、"TND"/"TND"、"BSND"/"PA_BBND"、"TND"/"PA_BBND"；非PA_BBND场景下`layout_q`和`layout_kv`必须一致。
  - 参数`cmp_residual_kv`需满足`cmp_residual_kv`[i] < `cmp_ratio`。
- Ascend 950PR/Ascend 950DT约束：
  - has_ori_kv为true，且ori_topk不为0且ori_mask_mode为0时，ori_topk_length必须传入。
  - has_cmp_kv为true，且cmp_top不为0且cmp_mask_mode为0时，cmp_topk_length必须传入。
  - layout_q=BSND场景
    - seqused_q和max_seqlen_q至少需要传入1个。
    - ori_topk不为0且传入ori_topk_length时，或cmp_topk不为0且传入cmp_topk_length时，max_seqlen_q必须传入query shape中的S值。
  - layout_kv=BSND场景
    - has_ori_kv为true，且ori_topk为0时，seqused_ori_kv和max_seqlen_ori_kv至少需要传入1个。
    - has_cmp_kv为true，且cmp_topk为0时，seqused_cmp_kv和max_seqlen_cmp_kv至少需要传入1个。
  - layout_q=TND场景
    - cu_seqlens_q必须传入。
  - layout_kv=TND场景
    - has_ori_kv为true时，cu_seqlens_ori_kv必须传入。
    - has_cmp_kv为true，cu_seqlens_cmp_kv必须传入。
  - layout_kv=PA_BBND场景
    - has_ori_kv为true，且ori_mask_mode不为0或ori_topk为0时，必传seqused_ori_kv参数。
    - has_cmp_kv为true，且cmp_mask_mode不为0或cmp_topk为0时，必传seqused_cmp_kv参数。
  - Batch取值规则
    - layout_q为BSND时，优先通过seqused_q的shape推导batch，seqused_q未传入则通过batch_size获取batch数。
    - layout_q为TND时，优先通过seqused_q的shape推导batch，seqused_q未传入则通过cu_seqlens_q的shape推导batch。
  - Query Seqlen取值规则
    - layout_q为BSND时，优先通过seqused_q中的元素获取seqlen，seqused_q未传入则通过max_seqlen_q获取seqlen。
    - layout_q为TND时，优先通过seqused_q中的元素获取seqlen，seqused_q未传入则通过cu_seqlens_q中的元素获取seqlen。
  - Ori_kv Seqlen取值规则
    - layout_kv为BSND时，优先通过seqused_ori_kv中的元素获取seqlen，seqused_ori_kv未传入则通过max_seqlen_ori_kv获取seqlen，若max_seqlen_ori_kv未传入且ori_topk不为0，则通过ori_topk_length或ori_topk获取seqlen（ori_topk_length优先级高于ori_topk）。
    - layout_kv为TND时，优先通过seqused_ori_kv中的元素获取seqlen，seqused_ori_kv未传入则通过cu_seqlens_ori_kv中的元素获取seqlen。
  - Cmp_kv Seqlen取值规则
    - layout_kv为BSND时，优先通过seqused_cmp_kv中的元素获取seqlen，seqused_cmp_kv未传入则通过max_seqlen_cmp_kv获取seqlen，若max_seqlen_cmp_kv未传入且cmp_topk不为0，则通过cmp_topk_length或cmp_topk获取seqlen（cmp_topk_length优先级高于cmp_topk）。
    - layout_kv为TND时，优先通过seqused_cmp_kv中的元素获取seqlen，seqused_cmp_kv未传入则通过cu_seqlens_cmp_kv中的元素获取seqlen。
- Atlas A3 训练系列产品/Atlas A3 推理系列产品约束：
  - `cmp_ratio`表示`cmp_kv`相对于压缩前KV长度的压缩倍率；仅传入`ori_kv`时不参与压缩KV计算。CSA场景传4，HCA场景传128。
  - `cmp_topk`在CSA场景支持512或1024，SWA、HCA场景传0。
- Atlas A2 训练系列产品/Atlas A2 推理系列产品约束：
  - `cmp_ratio`表示`cmp_kv`相对于压缩前KV长度的压缩倍率；仅传入`ori_kv`时不参与压缩KV计算。CSA场景传4，HCA场景传128。
  - `cmp_topk`在CSA场景支持512或1024，SWA、HCA场景传0。

## 调用说明

| 调用方式  | 样例代码                                                     | 说明                                                         |
| --------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| aclnn API | [test_aclnn_sparse_flash_mla_metadata](./examples/test_aclnn_sparse_flash_mla_metadata.cpp) | 通过[aclnnSparseFlashMlaMetadata](./docs/aclnnSparseFlashMlaMetadata.md)调用SparseFlashMlaMetadata算子 |
| PyTorch API | [test_torch_sparse_flash_mla_metadata](./examples/test_torch_sparse_flash_mla_metadata.py) | 通过[sparse_flash_mla_metadata](../../torch_extension/cann_ops_transformer/docs/zh/sparse_flash_mla.md)接口生成SparseFlashMla主算子使用的metadata |
