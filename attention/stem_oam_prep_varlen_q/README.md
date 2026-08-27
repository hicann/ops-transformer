# StemOamPrepVarlenQ

## 产品支持情况

|产品      | 是否支持 |
|:----------------------------|:-----------:|
|<term>Ascend 950PR/Ascend 950DT</term>|      √     |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>|      ×     |
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>|      ×     |
|<term>Atlas 200I/500 A2 推理产品</term>|      ×     |
|<term>Atlas 推理系列产品</term>|      ×     |
|<term>Atlas 训练系列产品</term>|      ×     |

## 功能说明

- **算子功能**：推理场景，Stem OAM block-sparse attention中Q侧预处理计算。将变长Q tensor从paged存储格式转化为按stem block(stemBlockSize token)分组的flattened qFlat输出，供后续OAM score计算。主要计算过程如下：
    - 首先对Q tensor进行Scale Fusion，将per-token scale factor应用到Q上
    - 然后根据cuSeqLensQ进行De-page Varlen，从paged存储中提取变长Q tensor
    - 对Q进行Weighted Group Sum（自然顺序，不翻转stride维度）
    - Flatten输出为qFlat：shape为[batch, H_q, max_Qb, kflat_dim]
    - Cast输出为BFLOAT16格式
- **计算公式**：

    阶段1 Scale Fusion

    $$q\_scale[b, h, pos] = qscale[b, h, pos]$$

    阶段2 De-page Varlen

    $$Q\_dense[b] = Cast(q[cu\_seqlens\_q[b]:cu\_seqlens\_q[b]+q\_len[b], :, :], \text{FP32})$$

    阶段3 Weighted Group Sum(自然顺序，NO flip)

    $$Q\_group\_sum[b,h,qb,g,:] = \sum_{r=0}^{R-1} Q\_blocks[b,h,qb,r,g,:] \times q\_scale[b,h,position(qb,r,g)]$$

    阶段4 Flatten

    $$qflat[b, h, qb, g \times D : (g+1) \times D] = Q\_group\_sum[b, h, qb, g, :]$$

    阶段5 Cast输出

    $$qflat\_out = qflat.to(\text{BF16})$$

- **关键特性**：Q侧stride维度为自然顺序（g ∈ [0, S-1]），不翻转（与K侧处理不同）。

## 参数说明

<table style="undefined;table-layout: fixed; width: 1427px"><colgroup>
<col style="width: 150px">
<col style="width: 100px">
<col style="width: 650px">
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
  </tr></thead>
<tbody>
  <tr>
    <td>q</td>
    <td>输入</td>
    <td>Device侧的aclTensor，变长Q tensor，所有batch的token拼接存储。shape为[total_tokens, H_q, D]，其中D必须等于128。</td>
    <td>FLOAT8_E4M3FN</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>qScale</td>
    <td>可选输入</td>
    <td>Device侧的aclTensor，Q的per-token scale factor，shape为[total_tokens, H_q]。</td>
    <td>FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>qSeqLens</td>
    <td>输入</td>
    <td>Device侧的aclTensor，每个batch的Q序列长度，shape为[batch]。长度等于batch，每个值≥0，不支持空Array。batch最大值为1024。</td>
    <td>INT32</td>
    <td>-</td>
  </tr>
  <tr>
    <td>cuSeqLensQ</td>
    <td>输入</td>
    <td>Device侧的aclTensor，Q的累积序列长度偏移量，用于varlen索引。shape为[batch+1]。cuSeqLensQ[0]必须为0，cuSeqLensQ[batch]必须等于total_tokens，单调递增。</td>
    <td>INT32</td>
    <td>-</td>
  </tr>
  <tr>
    <td>stemBlockSize</td>
    <td>属性</td>
    <td>stem block大小，控制每个stem block的token数量，决定Q Processing的分组粒度。默认值128。当前仅支持128。</td>
    <td>INT64</td>
    <td>-</td>
  </tr>
  <tr>
    <td>stemStride</td>
    <td>属性</td>
    <td>stem stride大小，控制stem block内stride group的token数量，决定qFlat的维度粒度。默认值16。当前仅支持16。</td>
    <td>INT64</td>
    <td>-</td>
  </tr>
  <tr>
    <td>qFlat</td>
    <td>输出</td>
    <td>Device侧的aclTensor，flattened Q输出，供OAM score计算使用。shape为[batch, H_q, max_Qb, kflat_dim]。其中max_Qb=ceil(max(qSeqLens)/stemBlockSize)，kflat_dim=stemStride×D（默认16×128=2048）。数据类型固定为BFLOAT16。</td>
    <td>BFLOAT16</td>
    <td>ND</td>
  </tr>
</tbody></table>

## 约束说明

- **确定性说明**：aclnnStemOamPrepVarlenQ默认确定性实现。
- **shape格式字段含义及约束**
    - total_tokens：total_tokens表示所有batch的token总数
    - H_q：H_q表示Q多头数，取值范围：1~128
    - stemBlockSize：stem block大小，当前仅支持128
    - stemStride：stem stride大小，当前仅支持16
    - D：Q矩阵最后一维，必须等于128
    - max_Qb：最大Q block数，max_Qb=ceil(max(qSeqLens)/stemBlockSize)
    - kflat_dim：qFlat输出维度，kflat_dim=stemStride×D（默认16×128=2048）
- **stride维度约束**：Q侧stride维度为自然顺序（g ∈ [0, S-1]），不翻转（与K侧处理不同）
- **空值处理**：当qSeqLens中某batch值为0时，该batch对应的qFlat输出填充为零
- **无条件Group Sum**：当qScale传入空指针时，Kernel内部Q Processing执行无权重Group Sum(Σ_r Q_blocks[:, r, :, :])，而非加权Group Sum(Σ_r Q_blocks×qScale)。
- **FP8输入约束**：FP8输入时，qScale最后一维需要能覆盖max(qSeqLens)个位置（padding到stemBlockSize对齐）

## 调用说明

| 调用方式  | 样例代码                                                     | 说明                                                         |
| --------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| aclnn接口 | [test_aclnn_stem_oam_prep_varlen_q](./examples/test_aclnn_stem_oam_prep_varlen_q.cpp) | 通过[aclnnStemOamPrepVarlenQ](./docs/aclnnStemOamPrepVarlenQ.md)调用StemOamPrepVarlenQ算子 |
| 图模式 | [test_geir_stem_oam_prep_varlen_q](./examples/test_geir_stem_oam_prep_varlen_q.cpp) | 通过[算子IR](./op_graph/stem_oam_prep_varlen_q_proto.h)调用StemOamPrepVarlenQ算子 |
| torch接口 | [test_torch_stem_oam_prep_varlen_q](./examples/test_torch_stem_oam_prep_varlen_q.py) | 通过`torch.ops.cann_ops_transformer.npu_stem_oam_prep_varlen_q`调用StemOamPrepVarlenQ算子 |
