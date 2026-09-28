# MatmulSwiglu

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT系列产品</term>   |     ×    |
|  <term>Atlas A3系列产品</term>   |     √    |
|  <term>Atlas A2系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2推理产品</term>    |     ×    |
|  <term>Atlas推理系列产品</term>     |     ×    |
|  <term>Atlas训练系列产品</term>    |     ×    |

## 功能说明

- 算子功能：将门控投影（gate_proj/up_proj）的 MatMul 与 SwiGLU 激活融合为单算子，用于大模型 FFN/MLP 前向。gate、up 两路共享同一激活 `x`，权重以 `[K, 2N]` 打包（前 N 列 gate、后 N 列 up）。
- 计算公式：

$$
[gate \mid up] = x @ weight (+ bias), \quad y = SiLU(gate) \ast up, \quad SiLU(a) = a \ast sigmoid(a)
$$

其中：
- x：输入激活，形状 `[M, K]`。
- weight：打包权重，形状 `[K, 2N]`（`transpose_weight=true` 时为 `[2N, K]`）。
- bias：可选偏置，形状 `[2N]`。
- y：输出，形状 `[M, N]`，`N = 2N / 2`。

中间结果 `[M, 2N]` 在片上（L0C→UB）就地完成 SwiGLU，不落回 GM。

## 参数说明

<table style="table-layout: auto; width: 100%">
<thead>
    <tr>
    <th style="white-space: nowrap">参数名</th>
    <th style="white-space: nowrap">输入/输出/属性</th>
    <th style="white-space: nowrap">描述</th>
    <th style="white-space: nowrap">数据类型</th>
    <th style="white-space: nowrap">数据格式</th>
    </tr>
</thead>
<tbody>
    <tr>
    <td>x</td>
    <td>输入</td>
    <td>激活张量，形状 `[M, K]`，M 为前置维之积。</td>
    <td>FLOAT16、BFLOAT16、FLOAT32</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>weight</td>
    <td>输入</td>
    <td>打包权重，形状 `[K, 2N]`（前 N 列 gate，后 N 列 up）；`transpose_weight=true` 时为 `[2N, K]`。类型同 x。</td>
    <td>FLOAT16、BFLOAT16、FLOAT32</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>bias</td>
    <td>输入（可选）</td>
    <td>偏置，形状 `[2N]`。固定为 FLOAT32（在 cube 内以 fp32 累加，不随 x 变化）。自动生成的 aclnn 接口会对其做非空校验，无偏置时传全 0 张量。</td>
    <td>FLOAT32</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>transpose_weight</td>
    <td>属性</td>
    <td>weight 是否转置，默认 false。true 时 weight 形状为 `[2N, K]`。</td>
    <td>BOOL</td>
    <td>-</td>
    </tr>
    <tr>
    <td>y</td>
    <td>输出</td>
    <td>结果张量，形状 `[M, N]`，`N = 2N / 2`。类型同 x。</td>
    <td>FLOAT16、BFLOAT16、FLOAT32</td>
    <td>ND</td>
    </tr>
</tbody></table>

## 约束说明

- x 的 rank ≥ 2，weight 的 rank == 2，且 weight 的 2N 维必须为偶数。
- weight 的 K 维需与 x 的 K 维一致。
- bias 固定为 FLOAT32，且不能传 nullptr（自动生成的 aclnn 接口会做非空校验），无偏置时传形状为 `[2N]` 的全 0 张量。
- x、weight、y 的数据类型需一致。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_matmul_swiglu.cpp](examples/test_aclnn_matmul_swiglu.cpp) | 通过 [aclnnMatmulSwiglu](docs/aclnnMatmulSwiglu.md) 接口方式调用 MatmulSwiglu 算子。 |
