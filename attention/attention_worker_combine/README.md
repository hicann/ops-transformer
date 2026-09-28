# AttentionWorkerCombine

## 产品支持情况

| 产品 | 是否支持 |
| :---- | :----: |
|<term>Ascend 950PR&950DT系列产品</term>| × |
|<term>Atlas A3系列产品</term>| √ |
|<term>Atlas A2系列产品</term>| √ |
|<term>Atlas 200I/500 A2推理产品</term>| × |
|<term>Atlas推理系列产品</term>| × |
|<term>Atlas训练系列产品</term>| × |

## 功能说明

+ 算子功能：将多个计算单元处理的注意力token数据进行融合，结合专家权重对结果进行加权，输出最终的注意力融合结果，并更新层ID。

## 参数说明

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
    <td class="tg-0pky">表示专家权重。</td>
    <td class="tg-0pky">FLOAT</td>
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
    <td class="tg-0pky">指定schedule_context中token数据的原始精度类型，0表示FLOAT16，1表示BFLOAT16。</td>
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

* schedule_context为1D的Tensor。
* expert_scales为2D的Tensor，[BatchSize, K]。
* y为2D的Tensor，[BatchSize, HiddenSize]，即第二维由属性hidden_size确定。
* layer_id和next_layer_id为1D的Tensor。

## 调用说明

| 调用方式 | 样例代码 | 说明 |
| ---- | ---- | ---- |
| 图模式调用 | [test_geir_attention_worker_combine.cpp](examples/test_geir_attention_worker_combine.cpp) | 通过[算子IR](op_graph/attention_worker_combine_proto.h)构图方式调用AttentionWorkerCombine算子。 |
| aclnn接口 | [test_aclnn_attention_worker_combine.cpp](examples/test_aclnn_attention_worker_combine.cpp) | 通过[aclnnAttentionWorkerCombine](docs/aclnnAttentionWorkerCombine.md)接口方式调用AttentionWorkerCombine算子。 |
