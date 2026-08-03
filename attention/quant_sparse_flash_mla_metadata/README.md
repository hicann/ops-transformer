# QuantSparseFlashMlaMetadata

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

- 算子功能：`QuantSparseFlashMlaMetadata`是`QuantSparseFlashMla`算子的前置算子，用于后续Attention计算生成负载均衡的任务划分方案。本算子不执行实际的Attention计算，而是根据输入参数在AI CPU计算出每个AI Core应处理的Attention计算起止范围，从而最大化计算资源的利用率，避免各Core间负载不均衡的问题。

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
      <td>表示不同Batch中Query的有效Sequence Length，shape为(B+1, )，仅layout_q为TND场景需传入。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>cu_seqlens_ori_kv</td>
      <td>可选输入</td>
      <td>表示不同Batch中ori_kv的有效Sequence Length，shape为(B+1, )，仅layout_kv为TND场景需传入。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>cu_seqlens_cmp_kv</td>
      <td>可选输入</td>
      <td>表示不同Batch中cmp_kv的有效Sequence Length，shape为(B+1, )，仅layout_kv为TND场景需传入。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>seqused_q</td>
      <td>可选输入</td>
      <td>表示不同Batch中Query实际参与运算的Sequence Length，shape为(B, )。</td>
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
      <td>预留参数，当前不生效。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>cmp_topk_length</td>
      <td>可选输入</td>
      <td>预留参数，当前不生效。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>num_heads_q</td>
      <td>属性</td>
      <td>表示Query的head个数，支持1到128。</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>num_heads_kv</td>
      <td>属性</td>
      <td>表示Key和Value对应的多头数，当前仅支持1。</td>
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
      <td>quant_mode</td>
      <td>属性</td>
      <td>表示量化模式，1表示Q、K、V 为per-tensor量化，Q、K、V类型为HIFLOAT8。</td>
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
      <td>表示Query的最长Sequence Length，默认值为0。</td>
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
      <td>表示对cmp_kv的压缩率，默认值为1。</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>ori_mask_mode</td>
      <td>可选属性</td>
      <td>表示q和ori_kv计算的mask模式。<br/>0: No Mask。<br/>3: RightDownCausal模式。<br/>4: Band模式。</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cmp_mask_mode</td>
      <td>可选属性</td>
      <td>表示q和cmp_kv计算的mask模式。<br/>0: No Mask。<br/>3: RightDownCausal模式。</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>ori_win_left</td>
      <td>可选属性</td>
      <td>表示q和ori_kv计算中q对过去token计算的数量，支持-1或非负数，其中-1表示窗口不受限。</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>ori_win_right</td>
      <td>可选属性</td>
      <td>表示q和ori_kv计算中q对未来token计算的数量，支持-1或非负数，其中-1表示窗口不受限。</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layout_q</td>
      <td>可选属性</td>
      <td>表示Query的排列格式，支持BSND、TND，默认值为BSND。</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layout_kv</td>
      <td>可选属性</td>
      <td>表示Key的排列格式，支持BSND、TND、PA_BBND，默认值为BSND。</td>
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
      <td>表示负载均衡结果输出，shape固定为[1024]。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
  </tbody>
  </table>

## 约束说明

- QuantSparseFlashMlaMetadata算子需要与QuantSparseFlashMla算子配套使用。
- B（Batch）表示输入样本批量大小。
- 参数cu_seqlens_q、cu_seqlens_ori_kv及cu_seqlens_cmp_kv要求其值为当前Batch与前序Batch有效token数的累加值，后一个元素的值必须大于等于前一个元素的值。
- 参数seqused_q、seqused_ori_kv、seqused_cmp_kv要求其值表示每个Batch中的有效token数。
- 参数cmp_residual_kv需满足cmp_residual_kv[i] < cmp_ratio。
- ori_mask_mode及cmp_mask_mode所表示的mask模式的详细介绍见[sparse_mode参数说明](../../docs/zh/context/sparse_mode_introduction.md)。

## 调用说明

| 调用方式  | 样例代码                                                     | 说明                                                         |
| --------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| aclnn API | [test_aclnn_quant_sparse_flash_mla_metadata](./examples/test_aclnn_quant_sparse_flash_mla_metadata.cpp) | 通过[aclnnQuantSparseFlashMlaMetadata](./docs/aclnnQuantSparseFlashMlaMetadata.md)接口调用QuantSparseFlashMlaMetadata算子 |
| PyTorch API | [test_torch_quant_sparse_flash_mla_metadata](./examples/test_torch_quant_sparse_flash_mla_metadata.py) | 通过[quant_sparse_flash_mla_metadata](../../torch_extension/cann_ops_transformer/docs/zh/quant_sparse_flash_mla.md)接口调用QuantSparseFlashMlaMetadata算子 |
