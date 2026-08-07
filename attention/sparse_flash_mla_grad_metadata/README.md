# SparseFlashMlaGradMetadata

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
|<term>Ascend 950PR/Ascend 950DT</term>|      √     |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>|      ×     |
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>|      ×     |
|<term>Atlas 200I/500 A2 推理产品</term>|      ×     |
|<term>Atlas 推理系列产品</term>|      ×     |
|<term>Atlas 训练系列产品</term>|      ×     |

## 功能说明

- 算子功能：`SparseFlashMlaGradMetadata`算子旨在生成一个任务列表，包含每个AIcore的Attention计算任务的起止点的Batch、Head、以及Q和K的分块的索引，供后续`SparseFlashMlaGrad`算子使用。

## 参数说明

<table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
  <col style="width: 180px">
  <col style="width: 100px">
  <col style="width: 700px">
  <col style="width: 90px">
  <col style="width: 80px">
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
      <td>cu_seqlens_q</td>
      <td>可选输入</td>
      <td>表示不同Batch中q的有效Sequence Length，shape为(B+1, )。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>cu_seqlens_ori_kv</td>
      <td>可选输入</td>
      <td>表示不同Batch中ori_kv的有效Sequence Length，shape为(B+1, )。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>cu_seqlens_cmp_kv</td>
      <td>可选输入</td>
      <td>表示不同Batch中cmp_kv的有效Sequence Length，shape为(B+1, )。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>seqused_q</td>
      <td>可选输入</td>
      <td>表示不同Batch中q实际参与运算的Sequence Length，shape为(B, )。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>seqused_ori_kv</td>
      <td>可选输入</td>
      <td>表示不同Batch中ori_kv实际参与运算的Sequence Length，shape为(B, )。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>seqused_cmp_kv</td>
      <td>可选输入</td>
      <td>表示不同Batch中cmp_kv实际参与运算的Sequence Length，shape为(B, )。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>cmp_residual_kv</td>
      <td>可选输入</td>
      <td>表示不同Batch中cmp_kv压缩后Sequence Length的余数，配合cmp_ratio实现cmp_kv部分的mask和负载计算。cmp_mask_mode=3且cmp_ratio≠1时必须传入，shape为(B, )。</td>
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
      <td>num_heads_q</td>
      <td>属性</td>
      <td>表示q的head个数，当前支持[1, 128]。</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>num_heads_kv</td>
      <td>属性</td>
      <td>表示ori_kv、cmp_kv对应的多头数，当前仅支持1。</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>head_dim</td>
      <td>属性</td>
      <td>表示注意力头的维度，当前仅支持512。</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>batch_size</td>
      <td>可选属性</td>
      <td>表示Batch数量，默认值为0。</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>max_seqlen_q</td>
      <td>可选属性</td>
      <td>表示q的最长Sequence Length，默认值为0。</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>max_seqlen_ori_kv</td>
      <td>可选属性</td>
      <td>表示ori_kv的最长Sequence Length，默认值为0。</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>max_seqlen_cmp_kv</td>
      <td>可选属性</td>
      <td>表示cmp_kv的最长Sequence Length，默认值为0。</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>ori_topk</td>
      <td>可选属性</td>
      <td>表示ori_kv中筛选出的关键稀疏token的个数，0表示非稀疏场景，默认值为0。</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cmp_topk</td>
      <td>可选属性</td>
      <td>表示cmp_kv中筛选出的关键稀疏token的个数，0表示非稀疏场景，默认值为0。</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cmp_ratio</td>
      <td>可选属性</td>
      <td>表示对cmp_kv的压缩率，默认值为1，当前支持[1, 128]。</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>ori_mask_mode</td>
      <td>可选属性</td>
      <td>表示q和ori_kv计算的mask模式，0表示No mask，3表示rightDownCausal模式，4表示sliding window模式，默认值为0。</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cmp_mask_mode</td>
      <td>可选属性</td>
      <td>表示q和cmp_kv计算的mask模式，0表示No mask，3表示rightDownCausal模式，默认值为0。</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>ori_win_left</td>
      <td>可选属性</td>
      <td>表示q和ori_kv计算中q对过去token计算的数量，-1表示无穷大，默认值为-1。</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>ori_win_right</td>
      <td>可选属性</td>
      <td>表示q和ori_kv计算中q对未来token计算的数量，-1表示无穷大，默认值为-1。</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layout_q</td>
      <td>可选属性</td>
      <td>表示q的排列格式，支持BSND、TND，默认值为BSND。</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layout_kv</td>
      <td>可选属性</td>
      <td>表示ori_kv、cmp_kv的排列格式，支持BSND、TND，默认值为BSND。</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>has_ori_kv</td>
      <td>可选属性</td>
      <td>用于标识是否含有ori_kv，默认值为true。</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>has_cmp_kv</td>
      <td>可选属性</td>
      <td>用于标识是否含有cmp_kv，默认值为true。</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>metadata</td>
      <td>输出</td>
      <td>表示负载均衡结果输出，shape固定为(1024, )</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
  </tbody>
  </table>

## 约束说明

- SparseFlashMlaGradMetadata算子需要与SparseFlashMlaGrad算子配套使用。
- B（Batch）表示输入样本批量大小，q、ori_kv、cmp_kv为配套的SparseFlashMlaGrad算子的入参，S1表示layout_q=BSND时，q shape中的S轴的大小，T1表示layout_q=TND时，q shape中的T轴的大小，S2表示layout_kv=BSND时，ori_kv shape中的S轴的大小，S3表示layout_kv=BSND时，cmp_kv shape中的S轴的大小，N2表示ori_kv、cmp_kv shape中的N轴的大小。
- layout_q、layout_kv须相同。
- 参数cu_seqlens_q、cu_seqlens_ori_kv及cu_seqlens_cmp_kv要求其值为当前Batch与前序Batch有效token数的累加值，第一个元素固定为0，后一个元素的值必须大于等于前一个元素的值。
- 参数seqused_q、seqused_ori_kv、seqused_cmp_kv要求其值表示每个Batch中的有效token数。
- 参数cmp_residual_kv需满足cmp_residual_kv[i] < cmp_ratio。
- ori_mask_mode及cmp_mask_mode所表示的mask模式的详细介绍见[sparse_mode参数说明](../../docs/zh/context/sparse_mode_introduction.md)。
- has_ori_kv为true时，ori_topk大于0认为ori_kv部分是稀疏的，ori_topk为0则认为ori_kv部分是非稀疏的。
- has_cmp_kv为true时，cmp_topk大于0认为cmp_kv部分是稀疏的，cmp_topk为0则认为cmp_kv部分是非稀疏的。
- has_ori_kv为true，ori_topk不为0且ori_mask_mode为0时，ori_topk_length必须传入，此时取ori_mask_mode规则与ori_topk_length元素的最小值作为当前q token对应的ori_kv的有效seqlen，其他ori_kv稀疏场景取ori_mask_mode规则与ori_topk的最小值作为当前q token对应的ori_kv的有效seqlen。
- has_cmp_kv为true，cmp_topk不为0且cmp_mask_mode为0时，cmp_topk_length必须传入，此时取cmp_mask_mode规则与cmp_topk_length元素的最小值作为当前q token对应的cmp_kv的有效seqlen，其他cmp_kv稀疏场景取cmp_mask_mode规则与cmp_topk的最小值作为当前q token对应的cmp_kv的有效seqlen。
- layout_q=BSND场景
  - max_seqlen_q必须传入S1的值。
- layout_kv=BSND场景
  - has_ori_kv为true时，max_seqlen_ori_kv必须传入S2的值。
  - has_cmp_kv为true时，max_seqlen_cmp_kv必须传入S3的值。
- layout_q=TND场景
  - cu_seqlens_q必须传入。
- layout_kv=TND场景
  - has_ori_kv为true时，cu_seqlens_ori_kv必须传入。
  - has_cmp_kv为true时，cu_seqlens_cmp_kv必须传入。
- Batch取值规则
  - layout_q为BSND时，优先通过seqused_q的shape推导batch，seqused_q未传入则通过batch_size获取batch数。
  - layout_q为TND时，优先通过seqused_q的shape推导batch，seqused_q未传入则通过cu_seqlens_q的shape推导batch。
- q Seqlen取值规则
  - layout_q为BSND时，优先通过seqused_q中的元素获取seqlen，seqused_q未传入则通过max_seqlen_q获取seqlen。
  - layout_q为TND时，优先通过seqused_q中的元素获取seqlen，seqused_q未传入则通过cu_seqlens_q中的元素获取seqlen。
- ori_kv Seqlen取值规则
  - layout_kv为BSND时，优先通过seqused_ori_kv中的元素获取seqlen，seqused_ori_kv未传入则通过max_seqlen_ori_kv获取seqlen。
  - layout_kv为TND时，优先通过seqused_ori_kv中的元素获取seqlen，seqused_ori_kv未传入则通过cu_seqlens_ori_kv中的元素获取seqlen。
- cmp_kv Seqlen取值规则
  - layout_kv为BSND时，优先通过seqused_cmp_kv中的元素获取seqlen，seqused_cmp_kv未传入则通过max_seqlen_cmp_kv获取seqlen。
  - layout_kv为TND时，优先通过seqused_cmp_kv中的元素获取seqlen，seqused_cmp_kv未传入则通过cu_seqlens_cmp_kv中的元素获取seqlen。

## 调用说明

| 调用方式  | 样例代码                                                     | 说明                                                         |
| --------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| aclnn API | [test_aclnn_sparse_flash_mla_grad_metadata](./examples/test_aclnn_sparse_flash_mla_grad_metadata.cpp) | 通过[aclnnSparseFlashMlaGradMetadata](./docs/aclnnSparseFlashMlaGradMetadata.md)接口调用SparseFlashMlaGradMetadata算子。 |
| PyTorch API | [test_torch_sparse_flash_mla_grad_metadata](./examples/test_torch_sparse_flash_mla_grad_metadata.py) | 通过[sparse_flash_mla_grad_metadata](../../torch_extension/cann_ops_transformer/docs/zh/sparse_flash_mla_grad.md)接口调用SparseFlashMlaGradMetadata算子。 |
