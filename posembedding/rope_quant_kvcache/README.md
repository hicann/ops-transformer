# RopeQuantKvcache

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     √    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     ×    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>    |     ×    |
|  <term>Atlas 训练系列产品</term>    |     ×    |
|  <term>Kirin X90 处理器系列产品</term> | √ |
|  <term>Kirin 9030 处理器系列产品</term> | √ |

## 功能说明

- 算子功能：对输入张量的尾轴进行切分，划分为q、k、v，对q、k进行旋转位置编码；当前Kernel写出q，并将量化后的k、v按照indice更新到kCacheRef和vCacheRef，保留的k、v输出当前未写入。

## 参数说明

<table style="table-layout: auto; width: 100%">
  <thead>
    <tr>
      <th style="white-space: nowrap">参数名</th>
      <th style="white-space: nowrap">输入/输出/属性</th>
      <th style="white-space: nowrap">描述</th>
      <th style="white-space: nowrap">数据类型</th>
      <th style="white-space: nowrap">数据格式</th>
    </tr></thead>
  <tbody>
    <tr>
      <td style="white-space: nowrap">qkv</td>
      <td style="white-space: nowrap">输入</td>
      <td style="white-space: nowrap">Device侧的aclTensor，需要切分的张量。</td>
      <td style="white-space: nowrap">FLOAT16</td>
      <td style="white-space: nowrap">ND</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">cos</td>
      <td style="white-space: nowrap">输入</td>
      <td style="white-space: nowrap">Device侧的aclTensor，用于旋转位置编码的张量。</td>
      <td style="white-space: nowrap">FLOAT16</td>
      <td style="white-space: nowrap">ND</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">sin</td>
      <td style="white-space: nowrap">输入</td>
      <td style="white-space: nowrap">Device侧的aclTensor，用于旋转位置编码的张量。</td>
      <td style="white-space: nowrap">FLOAT16</td>
      <td style="white-space: nowrap">ND</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">quant_scale</td>
      <td style="white-space: nowrap">输入</td>
      <td style="white-space: nowrap">Device侧的aclTensor，表示量化缩放参数的张量。</td>
      <td style="white-space: nowrap">FLOAT32</td>
      <td style="white-space: nowrap">ND</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">quant_offset</td>
      <td style="white-space: nowrap">输入</td>
      <td style="white-space: nowrap">Device侧的aclTensor，表示量化偏移量的张量。</td>
      <td style="white-space: nowrap">INT32</td>
      <td style="white-space: nowrap">ND</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">k_cache</td>
      <td style="white-space: nowrap">输入</td>
      <td style="white-space: nowrap">用于原地更新的输入。</td>
      <td style="white-space: nowrap">INT8</td>
      <td style="white-space: nowrap">ND</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">v_cache</td>
      <td style="white-space: nowrap">输入</td>
      <td style="white-space: nowrap">用于原地更新的输入。</td>
      <td style="white-space: nowrap">INT8</td>
      <td style="white-space: nowrap">ND</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">indice</td>
      <td style="white-space: nowrap">输入</td>
      <td style="white-space: nowrap">用于更新量化结果的下标</td>
      <td style="white-space: nowrap">INT32</td>
      <td style="white-space: nowrap">ND</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">size_splits</td>
      <td style="white-space: nowrap">属性</td>
      <td style="white-space: nowrap">用于对qkv进行切分。</td>
      <td style="white-space: nowrap">INT64</td>
      <td style="white-space: nowrap">-</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">layout</td>
      <td style="white-space: nowrap">属性</td>
      <td style="white-space: nowrap">表示qkv的数据排布方式。</td>
      <td style="white-space: nowrap">String</td>
      <td style="white-space: nowrap">-</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">kv_output</td>
      <td style="white-space: nowrap">属性</td>
      <td style="white-space: nowrap">控制是否输出原本的k、v。</td>
      <td style="white-space: nowrap">BOOL</td>
      <td style="white-space: nowrap">-</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">q</td>
      <td style="white-space: nowrap">输出</td>
      <td style="white-space: nowrap">切分出的q执行旋转位置编码后的结果。</td>
      <td style="white-space: nowrap">FLOAT16</td>
      <td style="white-space: nowrap">ND</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">k</td>
      <td style="white-space: nowrap">输出</td>
      <td style="white-space: nowrap">切分出的k执行旋转位置编码后的结果。</td>
      <td style="white-space: nowrap">FLOAT16</td>
      <td style="white-space: nowrap">ND</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">v</td>
      <td style="white-space: nowrap">输出</td>
      <td style="white-space: nowrap">切分出的v。</td>
      <td style="white-space: nowrap">FLOAT16</td>
      <td style="white-space: nowrap">ND</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">k_cache</td>
      <td style="white-space: nowrap">输出</td>
      <td style="white-space: nowrap">切分出的k执行旋转位置编码并量化后的结果。</td>
      <td style="white-space: nowrap">INT8</td>
      <td style="white-space: nowrap">ND</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">v_cache</td>
      <td style="white-space: nowrap">输出</td>
      <td style="white-space: nowrap">切分出的v量化后的结果。</td>
      <td style="white-space: nowrap">INT8</td>
      <td style="white-space: nowrap">ND</td>
    </tr>
  </tbody></table>

## 约束说明

- `qkv`为3维`[B, S, H]`，`H=(Nq+Nkv+Nkv)*D`；`cos`、`sin`为4维`[B, S, 1, D]`，其B、S、D必须与切分后的k一致，不支持Broadcast。
- `quant_scale`和`quant_offset`均为1维`[D]`；kCache和vCache均为4维`[B, S_cache, Nkv, D]`且shape相同，`S_cache`必须能容纳更新位置；`indice`的元素数为B。
- `size_splits`为3项`[Nq*D, Nkv*D, Nkv*D]`，三项之和必须等于`qkv`尾轴H；`layout`和`kv_output`当前均未被实现消费。
- 不支持动态Shape（-1）和动态Rank（-2）。

## 调用说明

| 调用方式      | 调用样例                 | 说明                                                         |
|--------------|-------------------------|--------------------------------------------------------------|
| 图模式调用 | [test_geir_rope_quant_kvcache](examples/test_geir_rope_quant_kvcache.cpp) | 通过图模式方式调用算子。 |
