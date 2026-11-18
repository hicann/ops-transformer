# QuantSparseLightningIndexerMetadata

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT系列产品</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3系列产品</term>：支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2系列产品</term>：支持
<!-- end id3 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2推理产品</term>：不支持
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas推理系列产品</term>：不支持
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas训练系列产品</term>：不支持
<!-- end id6 -->

> 本算子是 `QuantSparseLightningIndexer`（两级 TopK 第二级 consumer）的配套前置算子，输出 `metadata` 直接作为主算子的必传输入；

## 功能说明

- 算子功能：`QuantSparseLightningIndexerMetadata`是`QuantSparseLightningIndexer`（两级 TopK 第二级 consumer）算子的前置算子，用于生成负载均衡的任务划分方案。本算子不执行实际的LightningIndexer计算，而是根据输入参数在AI CPU计算出每个AI Core应处理的计算起止范围，从而最大化计算资源的利用率，避免各Core间负载不均衡的问题。

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
      <td>表示不同Batch中q的有效Sequence Length，shape为(B+1, )，仅layout_q为TND场景下必传，第一个值固定为0。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>cu_seqlens_k</td>
      <td>可选输入</td>
      <td>表示不同Batch中k的有效Sequence Length，shape为(B+1, )，仅layout_k为TND场景下必传，第一个值固定为0。</td>
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
      <td>seqused_k</td>
      <td>可选输入</td>
      <td>表示不同Batch中k实际参与运算的Sequence Length，shape为(B, )。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>cmp_residual_k</td>
      <td>可选输入</td>
      <td>表示不同Batch中k压缩后Sequence Length的余数，配合cmp_ratio实现mask和负载计算，shape为(B, )。cmp_ratio不为1且mask_mode为3场景下必传。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>candidate_block_length</td>
      <td>可选输入</td>
      <td>各Query行的实际有效候选数，取值范围为[0,2048]，传入规则见约束说明。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>num_heads_q</td>
      <td>属性</td>
      <td>Query的head个数，支持[1,32]及64。</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>num_heads_k</td>
      <td>属性</td>
      <td>表示k的head个数，当前仅支持1。</td>
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
      <td>topk</td>
      <td>属性</td>
      <td>表示从q中筛选出的关键稀疏token的个数，当前仅支持[1, 8192]。</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>quant_mode</td>
      <td>属性</td>
      <td>用于标识输入的量化模式。</td>
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
      <td>表示q的最长Sequence Length，-1表示任意可能长度，默认值为-1。</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>max_seqlen_k</td>
      <td>可选属性</td>
      <td>表示k的最长Sequence Length，-1表示任意可能长度，默认值为-1。</td>
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
      <td>layout_k</td>
      <td>可选属性</td>
      <td>表示k的排列格式，支持BSND、TND、PA_BBND，默认值为BSND。</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>mask_mode</td>
      <td>可选属性</td>
      <td>表示sparse模式，0表示No mask，3表示rightDownCausal模式，默认值为0。</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cmp_ratio</td>
      <td>可选属性</td>
      <td>表示k的压缩率，取值范围[1, 128]，默认值为1，表示无压缩。</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>candidate_block_size</td>
      <td>可选属性</td>
      <td>每个候选块包含的Key位置数，默认值为8，当前仅支持8，必须与QuantSparseLightningIndexer一致。</td>
      <td>INT32</td>
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

- QuantSparseLightningIndexerMetadata算子需要与QuantSparseLightningIndexer算子配套使用。
- B（Batch）表示输入样本批量大小，q、k为配套的QuantSparseLightningIndexer算子的入参，S1表示layout_q=BSND时，q shape中的S轴的大小，S2表示layout_k=BSND时，k shape中的S轴的大小。
- 参数cu_seqlens_q、cu_seqlens_k要求其值为当前Batch与前序Batch有效token数的累加值，第一个元素固定为0，后一个元素的值必须大于等于前一个元素的值。
- 参数seqused_q、seqused_k要求其值表示每个Batch中的有效token数。
- 参数cmp_residual_k需满足cmp_residual_k[i] < cmp_ratio。
- mask_mode所表示的mask模式的详细介绍见[sparse_mode参数说明](../../docs/zh/context/sparse_mode_introduction.md)。

<!-- npu="950" id7 -->
- <term>Ascend 950PR&950DT系列产品</term>：
  - `quant_mode`仅支持5，即MXFP4量化。
  - `num_heads_q`支持[1,32]，`num_heads_k`仅支持1，`head_dim`仅支持128。
  - `topk`支持[1,8192]，`candidate_block_size`仅支持8。
  - `candidate_block_length`必须传入，数据类型为INT32，每个元素表示对应Query的实际有效候选数，取值范围为[0,2048]。
  - 输出`metadata`的数据类型为INT32，shape固定为`(1024,)`，使用连续存储，且与输入位于同一NPU设备。
  - 序列信息、布局、mask和压缩参数必须与计算接口保持一致；上述参数或候选有效长度变化后，必须重新生成metadata。
  - 本接口生成的metadata仅供QuantSparseLightningIndexer使用，不能与QuantLightningIndexerV2的metadata混用。
  - layout_q支持BSND、TND，layout_k支持BSND、TND、PA_BBND；非分页布局必须相同。
  - Query或Key为TND布局时，必须传入对应的cu_seqlens_q或cu_seqlens_k；BSND布局不传入前缀和。
  - Key为BSND布局时，max_seqlen_k必须等于Key的物理序列长度；为PA_BBND布局时，seqused_k与max_seqlen_k至少传入一个，未传入seqused_k时，max_seqlen_k必须与计算接口的逻辑Key容量一致。
<!-- end id7 -->

<!-- npu="A3,910b" id8 -->
- <term>Atlas A3系列产品</term>、<term>Atlas A2系列产品</term>：
  - num_heads_q仅支持32、64，quant_mode仅支持2，topk仅支持[1,2048]，layout_k仅支持PA_BBND，cmp_ratio仅支持1、2、4、8、16、32、64、128。
  - candidate_block_length为预留参数，不传入或传入空Tensor，不支持非空Tensor。
  - layout_q=BSND场景
    - max_seqlen_q必须传入S1的值。
  - layout_q=TND场景
    - cu_seqlens_q必须传入。
  - layout_k=PA_BBND场景
    - seqused_k必须传入。
  - Batch取值规则
    - layout_q为BSND时，优先通过seqused_q的shape推导batch，seqused_q未传入则通过batch_size获取batch数。
    - layout_q为TND时，优先通过seqused_q的shape推导batch，seqused_q未传入则通过cu_seqlens_q的shape推导batch。
  - q Seqlen取值规则
    - layout_q为BSND时，优先通过seqused_q中的元素获取seqlen，seqused_q未传入则通过max_seqlen_q获取seqlen。
    - layout_q为TND时，优先通过seqused_q中的元素获取seqlen，seqused_q未传入则通过cu_seqlens_q中的元素获取seqlen。
<!-- end id8 -->

## 调用说明

| 调用方式  | 样例代码                                                     | 说明                                                         |
| --------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| aclnn API | [test_aclnn_quant_sparse_lightning_indexer_metadata](./examples/test_aclnn_quant_sparse_lightning_indexer_metadata.cpp) | 通过[aclnnQuantSparseLightningIndexerMetadata](./docs/aclnnQuantSparseLightningIndexerMetadata.md)接口调用QuantSparseLightningIndexerMetadata算子 |
| PyTorch API | [test_torch_quant_sparse_lightning_indexer_metadata](./examples/test_torch_quant_sparse_lightning_indexer_metadata.py) | 通过[quant_sparse_lightning_indexer_metadata](docs/torchapi_quant_sparse_lightning_indexer_metadata.md)接口调用QuantSparseLightningIndexerMetadata算子 |
