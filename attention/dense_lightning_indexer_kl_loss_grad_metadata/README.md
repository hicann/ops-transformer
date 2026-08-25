# DenseLightningIndexerKLLossGradMetadata

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
|<term>Ascend 950PR/Ascend 950DT</term>|      √     |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>|      x     |
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>|      x     |
|<term>Atlas 200I/500 A2 推理产品</term>|      ×     |
|<term>Atlas 推理系列产品</term>|      ×     |
|<term>Atlas 训练系列产品</term>|      ×     |

## 功能说明

- 算子功能：`DenseLightningIndexerKLLossGradMetadata`算子旨在根据`DenseLightningIndexerKLLossGrad`算子的输入shape、layout、mask和压缩比例信息，计算并输出分核切分metadata，供后续`DenseLightningIndexerKLLossGrad`算子使用。

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
      <td>表示不同Batch中Query的有效Sequence Length，shape为(B+1, )，仅layout_q为TND场景下必传，第一个值固定为0。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>cu_seqlens_k</td>
      <td>可选输入</td>
      <td>表示不同Batch中Key的有效Sequence Length，shape为(B+1, )，仅layout_k为TND场景下必传，第一个值固定为0。</td>
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
      <td>seqused_k</td>
      <td>可选输入</td>
      <td>表示不同Batch中Key实际参与运算的Sequence Length，shape为(B, )。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>cmp_residual_k</td>
      <td>可选输入</td>
      <td>表示不同Batch中cmp_kv压缩后Sequence Length的余数，配合cmp_ratio实现cmp_kv部分的mask和负载计算，shape为(B, )。cmp_ratio不为1且mask_mode为3场景下必传。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>num_heads_q</td>
      <td>属性</td>
      <td>表示Query的head个数，当前支持[1, 128]。</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>num_heads_k</td>
      <td>属性</td>
      <td>表示Key的head个数，当前仅支持1。</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>head_dim</td>
      <td>属性</td>
      <td>表示注意力头的维度，当前仅支持128。</td>
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
      <td>max_seqlen_k</td>
      <td>可选属性</td>
      <td>表示Key的最长Sequence Length，默认值为0。</td>
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
      <td>layout_k</td>
      <td>可选属性</td>
      <td>表示Key的排列格式，支持BSND、TND，默认值为BSND。</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>mask_mode</td>
      <td>可选属性</td>
      <td>表示mask模式，0表示No mask，3表示rightDownCausal模式，默认值为0。</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cmp_ratio</td>
      <td>可选属性</td>
      <td>表示Key的压缩率，取值范围[1, 128]，默认值为1，表示无压缩。</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>metadata</td>
      <td>输出</td>
      <td>表示负载均衡结果输出，shape固定为[64]。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
  </tbody>
  </table>

## 约束说明

- DenseLightningIndexerKLLossGradMetadata算子需要与DenseLightningIndexerKLLossGrad算子配套使用。
- B（Batch）表示输入样本批量大小。
- 参数cu_seqlens_q、cu_seqlens_k要求其值为当前Batch与前序Batch有效token数的累加值，后一个元素的值必须大于等于前一个元素的值。
- 参数seqused_q、seqused_k要求其值表示每个Batch中的有效token数。
- 参数cmp_residual_k需满足cmp_residual_k[i] < cmp_ratio。
- mask_mode所表示的mask模式的详细介绍见[sparse_mode参数说明](../../docs/zh/context/sparse_mode_introduction.md)。

## 问题定位说明

- 关于AI CPU算子Kernel常见执行问题或异常错误，问题定位方法请参考《故障处理》中“[故障案例集>算子执行问题>AI CPU算子Kernel执行报错](https://www.hiascend.com/document/detail/zh/canncommercial/latest/maintenref/troubleshooting/troubleshooting_0151.html)”。

## 调用说明

| 调用方式  | 样例代码                                                     | 说明                                                         |
| --------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| aclnn API | [test_aclnn_dense_lightning_indexer_kl_loss_grad_metadata](./examples/test_aclnn_dense_lightning_indexer_kl_loss_grad_metadata.cpp) | 通过[aclnnDenseLightningIndexerKLLossGradMetadata](./docs/aclnnDenseLightningIndexerKLLossGradMetadata.md)接口调用DenseLightningIndexerKLLossGradMetadata算子。 |
| PyTorch API | [test_torch_dense_lightning_indexer_kl_loss_grad_metadata](./examples/test_torch_dense_lightning_indexer_kl_loss_grad_metadata.py) | 通过[dense_lightning_indexer_kl_loss_grad_metadata](../../torch_extension/cann_ops_transformer/docs/zh/dense_lightning_indexer_kl_loss_grad.md)接口调用DenseLightningIndexerKLLossGradMetadata算子。 |
