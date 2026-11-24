# AttentionWorkerCombine

## 产品支持情况

| 产品 | 是否支持 |
| :---- | :----: |
|<term>Ascend 950PR&950DT系列产品</term>| √ |
|<term>Atlas A3系列产品</term>| √ |
|<term>Atlas A2系列产品</term>| √ |
|<term>Atlas 200I/500 A2推理产品</term>| × |
|<term>Atlas推理系列产品</term>| × |
|<term>Atlas训练系列产品</term>| × |

## 功能说明

+ 算子功能：将多个计算单元处理的注意力token数据进行融合，结合专家权重对结果进行加权，输出最终的注意力融合结果，并更新层ID。

## 参数说明

`R` 为 token 数，`K` 为每个 token 的路由专家数（不含共享专家），`H` 为隐藏维度大小，由 `hidden_size` 指定。

<table class="tg"><thead>
  <tr>
    <th class="tg-0pky">参数名</th>
    <th class="tg-0pky">输入/输出/属性</th>
    <th class="tg-0pky">描述</th>
    <th class="tg-0pky">数据类型</th>
    <th class="tg-0pky">数据格式</th>
  </tr></thead>
<tbody>
  <tr>
    <td class="tg-0pky">schedule_context</td>
    <td class="tg-0pky">输入</td>
    <td class="tg-0pky">包含调度上下文信息。</td>
    <td class="tg-0pky">INT8</td>
    <td class="tg-0pky">ND</td>
  </tr>
  <tr>
    <td class="tg-0pky">expert_scales</td>
    <td class="tg-0pky">输入</td>
    <td class="tg-0pky">表示专家权重，MXFP反量化时类型为FLOAT8_E8M0。</td>
    <td class="tg-0pky">FLOAT，FLOAT8_E8M0</td>
    <td class="tg-0pky">ND</td>
  </tr>
  <tr>
    <td class="tg-0pky">layer_id</td>
    <td class="tg-0pky">输入</td>
    <td class="tg-0pky">当前的模型层ID。</td>
    <td class="tg-0pky">INT32</td>
    <td class="tg-0pky">ND</td>
  </tr>
  <tr>
    <td class="tg-0pky">y</td>
    <td class="tg-0pky">输出</td>
    <td class="tg-0pky">最终的注意力合并结果。</td>
    <td class="tg-0pky">FLOAT16，BFLOAT16</td>
    <td class="tg-0pky">ND</td>
  </tr>
  <tr>
    <td class="tg-0pky">next_layer_id</td>
    <td class="tg-0pky">输出</td>
    <td class="tg-0pky">下一个要处理的层ID。</td>
    <td class="tg-0pky">INT32</td>
    <td class="tg-0pky">ND</td>
  </tr>
  <tr>
    <td class="tg-0pky">hidden_size</td>
    <td class="tg-0pky">属性</td>
    <td class="tg-0pky">token_data的隐藏维度大小，用于确定输出y的第二维大小。必要属性。</td>
    <td class="tg-0pky">Int</td>
    <td class="tg-0pky">-</td>
  </tr>
  <tr>
    <td class="tg-0pky">token_dtype</td>
    <td class="tg-0pky">属性</td>
    <td class="tg-0pky">指定schedule_context中token数据的原始精度类型，0表示FLOAT16，1表示BFLOAT16;<term>Ascend 950PR&950DT系列产品</term> 支持MXFP类型反量化，2表示MXFP8_E5M2, 3表示MXFP8_E4M3FN, 4表示MXFP4_E2M1。</td>
    <td class="tg-0pky">Int</td>
    <td class="tg-0pky">-</td>
  </tr>
    <tr>
    <td class="tg-0pky">need_schedule</td>
    <td class="tg-0pky">属性</td>
    <td class="tg-0pky">指定是否等待token数据填充完成后再执行，0表示不等待，1表示等待。</td>
    <td class="tg-0pky">Int</td>
    <td class="tg-0pky">-</td>
  </tr>
</tbody></table>

## 约束说明

* 所有输入和输出 Tensor 均不支持空 Tensor。
* `R` 和 `H` 必须为正整数，`K` 的取值范围为 `[1, 64]`。
* 非量化场景下，`expert_scales` 为 FLOAT 类型的二维 Tensor，shape 为 `(R, K)`。
* MXFP 反量化场景下，`expert_scales` 为 FLOAT8_E8M0 类型的三维 Tensor，shape 为 `(R, K + 1, S)`。专家维的最后一项对应共享专家。`S` 为 `ceil(H / 32)` 向上对齐到偶数后的值。
* MXFP 反量化场景下，`token_dtype` 取值为 2、3 或 4，`need_schedule` 取值为 0 或 1。
* `y` 为二维 Tensor，shape 为 `(R, H)`。`token_dtype` 为 0 时，`y` 为 FLOAT16，其余支持的取值对应 BFLOAT16。

## 调用说明

| 调用方式 | 样例代码 | 说明 |
| ---- | ---- | ---- |
| 图模式调用 | [test_geir_attention_worker_combine.cpp](examples/test_geir_attention_worker_combine.cpp) | 通过[算子IR](op_graph/attention_worker_combine_proto.h)构图方式调用AttentionWorkerCombine算子。 |
| aclnn接口 | [test_aclnn_attention_worker_combine.cpp](examples/test_aclnn_attention_worker_combine.cpp) | 通过[aclnnAttentionWorkerCombine](docs/aclnnAttentionWorkerCombine.md)接口方式调用AttentionWorkerCombine算子。 |
| PyTorch接口 | [torchapi_attention_worker_combine.md](docs/torchapi_attention_worker_combine.md) | 通过 `cann_ops_transformer.attention_worker_combine` 调用，支持非量化和MXFP反量化。 |
