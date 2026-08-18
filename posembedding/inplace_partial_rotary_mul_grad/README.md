# InplacePartialRotaryMulGrad

## 产品支持情况

| 产品                                                         |  是否支持   |
| :----------------------------------------------------------- |:-------:|
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    ×    |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    ×    |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×    |
| <term>Atlas 推理系列产品</term>                             |    ×    |
| <term>Atlas 训练系列产品</term>                              |    ×    |

## 功能说明

- **算子功能**：执行局部旋转位置编码InplacePartialRotaryMul的反向计算。该算子对输入dy的D维度上切片[start, end)区域执行旋转位置编码梯度计算，dx'的结果inplace写回dy的[start, end)区间。dy的[start, end)之外的数据保持不变。

- **计算公式**：

    取旋转位置编码的正向计算中，broadcast的轴列表为`dims`，在D维度上的切片范围为`[start, end)`，令参与计算的切片数据为：
    $$
    dy' = dy[..., start:end]
    $$
    $$
    cos' = cos[..., start:end]
    $$
    $$
    sin' = sin[..., start:end]
    $$
    则梯度计算公式可表达如下：

    （1）half模式（rotary_mode等于0）：
    $$
    dy1', dy2' = chunk(dy', chunks=2, dim=-1)
    $$

    $$
    cos1', cos2' = chunk(cos', chunks=2, dim=-1)
    $$

    $$
    sin1', sin2' = chunk(sin', chunks=2, dim=-1)
    $$

    $$
    dx' = cat((cos1' * dy1' + sin2' * dy2', cos2' * dy2' - sin1' * dy1'), dim=-1)
    $$



    （2）interleave模式（rotary_mode等于1）：
    $$
    dy1', dy2' = dy'[..., :: 2], dy'[..., 1 :: 2]
    $$

    $$
    cos1', cos2' = cos'[..., :: 2], cos'[..., 1 :: 2]
    $$

    $$
    sin1', sin2' = sin'[..., :: 2], sin'[..., 1 :: 2]
    $$

    $$
    dx' = stack((cos1' * dy1' + sin2' * dy2', cos2' * dy2' - sin1' * dy1'), dim=-1).reshape(dy'.shape)
    $$



    （3）quarter模式（rotary_mode等于2）：
    $$
    dy1', dy2', dy3', dy4' = chunk(dy', chunks=4, dim=-1)
    $$

    $$
    cos1', cos2', cos3', cos4' = chunk(cos', chunks=4, dim=-1)
    $$

    $$
    sin1', sin2', sin3', sin4' = chunk(sin', chunks=4, dim=-1)
    $$

    $$
    dx' = cat((cos1' * dy1' + sin2' * dy2', cos2' * dy2' - sin1' * dy1', cos3' * dy3' + sin4' * dy4', cos4' * dy4' - sin3' * dy3'), dim=-1)
    $$



    （4）interleave-half模式（rotary_mode等于3）：
    $$
    dy1', dy2' = chunk(dy', chunks=2, dim=-1)
    $$

    $$
    cos1', cos2' = chunk(cos', chunks=2, dim=-1)
    $$

    $$
    sin1', sin2' = chunk(sin', chunks=2, dim=-1)
    $$

    $$
    dx' = stack((cos1' * dy1' + sin2' * dy2', cos2' * dy2' - sin1' * dy1'), dim=-1).reshape(dy'.shape)
    $$



## 参数说明

<table style="table-layout: auto; width: 100%">
<colgroup>
  <col style="width: 10%">
  <col style="width: 10%">
  <col style="width: 40%">
  <col style="width: 20%">
  <col style="width: 10%">
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
    <td>dy</td>
    <td>输入</td>
    <td>公式中的dy，表示正向计算输出y的导数，inplace更新为正向输入x的导数。Inplace模式，dy同时作为输出写入结果。</td>
    <td>BFLOAT16、FLOAT16、FLOAT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>cos</td>
    <td>输入</td>
    <td>公式中的cos，正向计算输入，需与sin数据类型一致。</td>
    <td>BFLOAT16、FLOAT16、FLOAT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>sin</td>
    <td>输入</td>
    <td>公式中的sin，正向计算输入，需与cos数据类型一致。</td>
    <td>BFLOAT16、FLOAT16、FLOAT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>rotary_mode</td>
    <td>属性</td>
    <td>旋转模式，0=half，1=interleave，2=quarter，3=interleave-half。当前仅支持interleave模式（rotary_mode=1）。</td>
    <td>INT64</td>
    <td>-</td>
  </tr>
  <tr>
    <td>partial_slice</td>
    <td>属性</td>
    <td>dy最后一维(D维)上的切片范围，为[start, end)，表示对dy[..., start:end]执行旋转位置编码梯度计算。</td>
    <td>IntArray</td>
    <td>-</td>
  </tr>
</tbody>
</table>

## 约束说明

- 该算子仅支持Ascend 950 AI Processor。
- 该算子仅支持连续Tensor，不支持非连续Tensor。
- 该算子当前版本仅支持 interleave 模式（`rotary_mode=1`）。其他模式暂不支持。
- Inplace执行：输入dy和输出共享同一个Tensor，计算结果直接写回输入dy。
- 输入dy当前只支持BSND排布，输入cos/sin的shape必须与dy满足B/S/N维度的广播关系（如BSND、111D、1SND、B1ND、BS1D、11ND、B11D、1S1D等）。各参数的shape约束可以描述如下：
  - 输入张量dy的最后一维大小D必须小于等于1024。
  - 当切片长度(end - start)大于0时，输入张量cos、sin的最后一维大小必须等于切片长度。
  - 输入张量cos和sin的shape必须完全相同，cos和sin的B、S、N维度需要与dy满足[broadcast关系](../../docs/zh/context/broadcast_relationship.md)，且广播后的B、S、N必须等于dy的B、S、N。
  - half、interleave和interleave-half模式下，当切片长度(end - start)大于0时，切片长度必须能被2整除。
  - quarter模式下，当切片长度(end - start)大于0时，切片长度必须能被4整除。
  - 输入张量cos和sin的数据类型必须相同。
- **空输入支持**：当切片长度为0（`partial_slice`的start等于end，如`[0, 0]`），或dy、cos、sin中存在空Tensor（某维大小为0）时，算子执行no-op操作，不做旋转位置编码计算，直接返回。


## 调用说明

| 调用方式           | 调用样例                                                                                    | 说明                                                                                                  |
|----------------|-----------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------|
| aclnn调用 | [test_aclnn_inplace_partial_rotary_mul_grad](./examples/arch35/test_aclnn_inplace_partial_rotary_mul_grad.cpp) | 通过[aclnnInplacePartialRotaryMulGrad](./docs/aclnnInplacePartialRotaryMulGrad.md)接口方式调用InplacePartialRotaryMulGrad算子。             |
| 图模式调用 | [test_geir_inplace_partial_rotary_mul_grad](./examples/arch35/test_geir_inplace_partial_rotary_mul_grad.cpp) | 通过[算子IR](./op_graph/inplace_partial_rotary_mul_grad_proto.h)构图方式调用InplacePartialRotaryMulGrad算子。 |
