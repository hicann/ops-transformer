# InplacePartialRotaryMul

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                             |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×     |

## 功能说明

- **算子功能**：执行单路旋转位置编码的Inplace计算，直接修改输入张量，不产生新的输出张量。支持通过`partial_slice`参数对输入张量的最后一维进行部分旋转位置编码计算，仅对指定范围内的数据进行旋转位置编码。
- **计算公式**：

    interleave模式（rotary_mode等于1）：
    $$
    x1 = x[..., ::2]
    $$

    $$
    x2 = x[..., 1::2]
    $$

    $$
    x\_rotate = torch.cat((-x2, x1), dim=-1)
    $$

    $$
    x = x * cos + x\_rotate * sin
    $$

## 参数说明

<table style="undefined;table-layout: fixed; width: 1200px">
<colgroup>
  <col style="width: 50px">
  <col style="width: 50px">
  <col style="width: 200px">
  <col style="width: 100px">
  <col style="width: 50px">
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
    <td>xRef</td>
    <td>输入</td>
    <td>公式中的x，待执行旋转位置编码的张量。Inplace模式，xRef同时作为输出写入结果。</td>
    <td>BFLOAT16、FLOAT16、FLOAT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>cos</td>
    <td>输入</td>
    <td>公式中的cos，参与计算的位置编码张量。</td>
    <td>BFLOAT16、FLOAT16、FLOAT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>sin</td>
    <td>输入</td>
    <td>公式中的sin，参与计算的位置编码张量。</td>
    <td>BFLOAT16、FLOAT16、FLOAT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>rotary_mode</td>
    <td>输入</td>
    <td>公式中的旋转模式，当前仅支持interleave模式（rotary_mode=1）。</td>
    <td>INT64</td>
    <td>-</td>
  </tr>
  <tr>
    <td>partial_slice</td>
    <td>输入</td>
    <td>部分旋转的切片范围[start, end)，作用于最后一维。默认值为[0, 0]。</td>
    <td>INT64数组</td>
    <td>-</td>
  </tr>
</tbody>
</table>

## 约束说明

- 不支持非连续。
- 输入张量xRef最后一维（D）大小不超过1024。
- interleave模式（rotary_mode = 1）下，xRef最后一维（D）大小必须为2的倍数。
- 输入张量cos、sin最后一维大小必须相同，且必须等于partialSlice的切片长度（即partialSlice[1] - partialSlice[0]）。
- partialSlice约束：sliceStart ≥ 0，sliceEnd ≥ 0，sliceEnd ≤ xRef最后一维（D）大小，sliceLength = sliceEnd - sliceStart >= 0，当sliceEnd和sliceStart相同时，不做旋转位置编码，直接返回。
- 仅支持interleave模式（rotary_mode = 1）。
- Inplace执行：输入xRef和输出共享同一个Tensor，计算结果直接写回输入xRef。
- xRef的shape为BSND。
- cos/sin的shape必须与xRef满足广播关系，且存在如下约束：

  - <term>Ascend 950PR/Ascend 950DT</term>：
    cos/sin的shape当前只支持BSND、B1ND、B11D、111D排布。

  - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：
    cos/sin的shape当前只支持BS1D、B11D排布。

## 调用说明

| 调用方式           | 调用样例                                                                                    | 说明                                                                                                  |
|----------------|-----------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------|
| aclnn调用 | [test_aclnn_inplace_partial_rotary_mul](./examples/test_aclnn_inplace_partial_rotary_mul.cpp) | 通过[aclnnInplacePartialRotaryMul](./docs/aclnnInplacePartialRotaryMul.md)接口方式调用InplacePartialRotaryMul算子。             |
| PyTorch API | [inplace_partial_rotary_mul](../../posembedding/inplace_partial_rotary_mul/docs/torchapi_inplace_partial_rotary_mul.md) | 通过[cann_ops_transformer.inplace_partial_rotary_mul](../../posembedding/inplace_partial_rotary_mul/docs/torchapi_inplace_partial_rotary_mul.md)接口方式调用InplacePartialRotaryMul算子。 |
| 图模式调用 | [test_geir_inplace_partial_rotary_mul](./examples/test_geir_inplace_partial_rotary_mul.cpp) | 通过[算子IR](./op_graph/inplace_partial_rotary_mul_proto.h)构图方式调用InplacePartialRotaryMul算子。 |
