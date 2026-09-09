# stem_oam_prep_varlen_q

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR/Ascend 950DT</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：不支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：不支持
<!-- end id3 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2 推理产品</term>：不支持
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas 推理系列产品</term>：不支持
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas 训练系列产品</term>：不支持
<!-- end id6 -->


## 功能说明

- **接口功能**：

  完成Stem OAM block-sparse attention中Q侧预处理计算。将变长Q tensor从paged存储格式转化为按stem block分组的flattened qFlat输出，供后续OAM score计算。

- **计算公式**：

  阶段1 Scale Fusion:
  $$q\_scale[b, h, pos] = qscale[b, h, pos]$$

  阶段2 De-page Varlen:
  $$Q\_dense[b] = Cast(q[cu\_seqlens\_q[b]:cu\_seqlens\_q[b]+q\_len[b], :, :], \text{FP32})$$

  阶段3 Weighted Group Sum (自然顺序，NO flip):
  $$Q\_group\_sum[b,h,qb,g,:] = \sum_{r=0}^{R-1} Q\_blocks[b,h,qb,r,g,:] \times q\_scale[b,h,position(qb,r,g)]$$

  阶段4 Flatten:
  $$qflat[b, h, qb, g \times D : (g+1) \times D] = Q\_group\_sum[b, h, qb, g, :]$$

  阶段5 Cast输出:
  $$qflat\_out = qflat.to(\text{BF16})$$

- **关键特性**：Q侧stride维度为自然顺序（不翻转），与K侧的翻转处理不同。

## 函数原型

```python
cann_ops_transformer.stem_oam_prep_varlen_q(q, q_seq_lens, cu_seq_lens_q, *, q_scale=None, stem_block_size=128, stem_stride=16) -> Tensor
```

## 参数说明

<table style="undefined;table-layout: fixed; width:1200px"><colgroup>
<col style="width: 150px">
<col style="width: 120px">
<col style="width: 100px">
<col style="width: 350px">
<col style="width: 180px">
<col style="width: 200px">
</colgroup>
<thead>
<tr>
    <th>参数名</th>
    <th>参数类型</th>
    <th>可选/必选</th>
    <th>描述</th>
    <th>数据类型</th>
    <th>维度(shape)</th>
</tr>
</thead>
<tbody>
    <tr>
        <td>q</td>
        <td>Tensor</td>
        <td>必选</td>
        <td>表示变长Q tensor，所有batch的token拼接存储。对应公式中q。最后一维必须等于128。</td>
        <td>float8_e4m3fn</td>
        <td>(total_tokens, H_q, D)，D=128</td>
    </tr>
    <tr>
        <td>q_seq_lens</td>
        <td>List[int]</td>
        <td>必选</td>
        <td>表示每个batch的Q序列长度。对应公式中q_seq_lens。长度等于batch，取值范围为(0, 1024]，每个值≥0。</td>
        <td>int64</td>
        <td>(batch,)</td>
    </tr>
    <tr>
        <td>cu_seq_lens_q</td>
        <td>List[int]</td>
        <td>必选</td>
        <td>表示Q的累积序列长度偏移量，用于varlen索引。对应公式中cu_seqlens_q。长度等于batch+1，cu_seq_lens_q[0]必须为0，cu_seq_lens_q[batch]必须等于total_tokens，单调递增（允许相等）。</td>
        <td>int64</td>
        <td>(batch+1,)</td>
    </tr>
    <tr>
        <td>q_scale</td>
        <td>Tensor</td>
        <td>必选</td>
        <td>表示Q的per-token scale factor。对应公式中qscale。q为float8_e4m3fn时必填，数据类型必须为float32。</td>
        <td>float32</td>
        <td>(total_tokens, H_q)</td>
    </tr>
    <tr>
        <td>stem_block_size</td>
        <td>int</td>
        <td>可选</td>
        <td>表示stem block大小，对应公式中B。控制每个stem block的token数量，决定Q Processing的分组粒度。默认值128。</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>stem_stride</td>
        <td>int</td>
        <td>可选</td>
        <td>表示stem stride大小，对应公式中S。控制stem block内stride group的token数量，决定qFlat的维度粒度。默认值16。</td>
        <td>-</td>
        <td>-</td>
    </tr>
    </tbody>
</table>

## 输出说明

<table style="undefined;table-layout: fixed; width:1200px"><colgroup>
<col style="width: 150px">
<col style="width: 120px">
<col style="width: 100px">
<col style="width: 350px">
<col style="width: 180px">
<col style="width: 200px">
</colgroup>
<thead>
<tr>
    <th>参数名</th>
    <th>参数类型</th>
    <th>可选/必选</th>
    <th>描述</th>
    <th>数据类型</th>
    <th>维度(shape)</th>
</tr>
</thead>
<tbody>
    <tr>
        <td>q_flat</td>
        <td>Tensor</td>
        <td>必选</td>
        <td>表示flattened Q输出，供OAM score计算使用。对应公式中qflat_out。其中max_Qb=ceil(max(q_seq_lens)/stem_block_size)，kflat_dim=stem_stride×D=2048。</td>
        <td>bfloat16</td>
        <td>(batch, H_q, max_Qb, kflat_dim)</td>
    </tr>
</tbody>
</table>

## 约束说明

- 该接口支持训练、推理场景下使用。
- 该接口支持单算子模式和TorchAir图模式调用。
- q的最后一维D必须等于128。
- stem_block_size仅支持128，stem_stride仅支持16。派生值：R = stem_block_size / stem_stride = 8, kflat_dim = stem_stride × D = 2048。
- q仅支持float8_e4m3fn数据类型，q_scale必填且数据类型为float32，q_flat输出固定为bfloat16。
- q_seq_lens长度取值范围为(0, 1024]。
- 当q_seq_lens中某batch值为0时，该batch对应的q_flat输出填充为零。
- Q侧stride维度为自然顺序（g ∈ [0, S-1]），不翻转（与K侧处理不同）。
- 支持空Tensor输入：当q为空或q_seq_lens长度为0时，直接返回。

## 确定性计算

默认支持确定性计算。

## 调用说明

- 单算子模式调用：

  ```python
  import torch
  import torch_npu
  from cann_ops_transformer.ops import stem_oam_prep_varlen_q

  total_tokens = 256
  H_q = 32
  D = 128
  batch = 2

  q = torch.randn(total_tokens, H_q, D, dtype=torch.float32).to(torch.float8_e4m3fn).npu()
  q_scale = torch.randn(total_tokens, H_q, dtype=torch.float32).npu()

  q_seq_lens = [128, 128]
  cu_seq_lens_q = [0, 128, 256]

  q_flat = stem_oam_prep_varlen_q(
      q, q_seq_lens, cu_seq_lens_q, q_scale=q_scale
  )
  ```

- 图模式调用

  ```python
  import torch
  import torch_npu
  import torchair
  from cann_ops_transformer.ops import stem_oam_prep_varlen_q

  torch_npu.npu.set_device(0)

  total_tokens = 256
  H_q = 32
  D = 128
  batch = 2

  class StemOamPrepVarlenQModel(torch.nn.Module):
      def forward(self, q, q_scale):
          return stem_oam_prep_varlen_q(
              q, [128, 128], [0, 128, 256], q_scale=q_scale
          )

  model = StemOamPrepVarlenQModel().npu()
  npu_backend = torchair.get_npu_backend()
  model = torch.compile(model, backend=npu_backend, dynamic=False)

  q = torch.randn(total_tokens, H_q, D, dtype=torch.float32).to(torch.float8_e4m3fn).npu()
  q_scale = torch.randn(total_tokens, H_q, dtype=torch.float32).npu()

  output = model(q, q_scale)
  ```
