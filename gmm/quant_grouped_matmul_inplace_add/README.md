# QuantGroupedMatmulInplaceAdd

## 产品支持情况

|产品      | 是否支持 |
|:----------------------------|:-----------:|
|<term>Ascend 950PR&950DT系列产品</term>|      √     |
|<term>Atlas A3系列产品</term>|      ×     |
|<term>Atlas A2系列产品</term>|      ×     |
|<term>Atlas 200I/500 A2推理产品</term>|      ×     |
|<term>Atlas推理系列产品</term>|      ×     |
|<term>Atlas训练系列产品</term>|      ×     |

## 功能说明

- 算子功能：在micro-batch训练场景，需要做micro-batch的梯度累计，会存在大量GroupedMatMul后接InplaceAdd的融合场景。QuantGroupedMatmulInplaceAdd算子将上述算子融合起来，提高网络性能。实现分组矩阵乘计算和加法计算，基本功能为矩阵乘和加法的组合，如T-C量化场景下$y_i[m,n]=(x1_i[m,k_i] \times x2_i[k_i,n]) * scale2_i[n] * scale1_i + y_i[m,n], i=1...g$，其中g为分组个数，$m/k_i/n$为对应的维度。

    相较于[GroupedMatmulV4](../grouped_matmul/docs/aclnnGroupedMatmulV4.md)接口，**此接口变化：**
    - 输入输出参数类型均为aclTensor。
    - 在GroupedMatMul计算结束后增加了InplaceAdd计算。
    - 仅支持量化场景（1.mx量化；2.T-C量化；3.T-T量化）。量化方式请参见[量化介绍](../../docs/zh/context/quant_mode_introduction.md)。
    - 仅支持x1、x2是FLOAT8_E5M2、FLOAT8_E4M3FN、HIFLOAT8的输入。

- 计算公式：
    - **mx量化：**

    $$
     y_i[m,n] = \sum_{j=0}^{kLoops-1} ((\sum_{k=0}^{gsK-1} (x1Slice_i * x2Slice_i)) * (scale1_i[m, j] * scale2_i[j, n])) + y_i[m,n]
    $$

    其中，gsK代表K轴的量化的block size即32，$x1Slice_i$代表$x1_i$第m行长度为gsK的向量，$x2Slice_i$代表$x2_i$第n列长度为gsK的向量，K轴均从$j*gsK$起始切片，j的取值范围[0, kLoops), kLoops=ceil($K_i$ / gsK)，支持最后的切片长度不足gsK。
    - **T-T/T-C量化：**

    $$
     y_i=(x1_i\times x2_i) * scale2_i * scale1_i + y_i
    $$

## 参数说明

<table style="undefined;table-layout: fixed;width: 1567px"><colgroup>
<col style="width: 170px">
<col style="width: 120px">
<col style="width: 300px">
<col style="width: 330px">
<col style="width: 212px">
<col style="width: 100px">
<col style="width: 190px">
<col style="width: 145px">
</colgroup>
<thead>
  <tr>
    <th>参数名</th>
    <th style="white-space: nowrap">输入/输出/属性</th>
    <th>描述</th>
    <th>数据类型</th>
    <th>数据格式</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>x1</td>
    <td rowspan="1">输入</td>
    <td>公式中的输入x1，表示输入矩阵（左矩阵）。</td>
    <td>FLOAT8_E5M2、FLOAT8_E4M3FN、HIFLOAT8</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>x2</td>
    <td rowspan="1">输入</td>
    <td>公式中的输入x2，表示权重矩阵（右矩阵）。</td>
    <td>FLOAT8_E5M2、FLOAT8_E4M3FN、HIFLOAT8</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>scale1</td>
    <td rowspan="1">可选输入</td>
    <td>量化参数中由x1量化引入的缩放因子。综合约束请参见<a href="#约束说明">约束说明</a>。</td>
    <td>FLOAT32、FLOAT8_E8M0</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>scale2</td>
    <td rowspan="1">输入</td>
    <td>量化参数中由x2量化引入的缩放因子。综合约束请参见<a href="#约束说明">约束说明</a>。</td>
    <td>FLOAT32、FLOAT8_E8M0</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>group_list</td>
    <td rowspan="1">输入</td>
    <td>分组轴方向的matmul大小分布，前缀和的分组索引列表。当group_list_type为0时须为非负单调非递减数列，为1时须为非负数列。</td>
    <td>INT64</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>y</td>
    <td rowspan="1">输入输出</td>
    <td>公式中的输入输出y，InplaceAdd的累加器。计算matmul结果后与y相加并写回。</td>
    <td>FLOAT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>group_list_type</td>
    <td rowspan="1">属性</td>
    <td>整数型属性，group_list的解析方式：0表示累积和（cumsum），1表示每组大小（count）。</td>
    <td>INT64</td>
    <td>-</td>
  </tr>
  <tr>
    <td>group_size</td>
    <td rowspan="1">属性</td>
    <td>整数型属性，m/n/k方向的量化分组大小。</td>
    <td>INT64</td>
    <td>-</td>
  </tr>
</tbody>
</table>

## 约束说明

  - x1和x2的每一维大小在32字节对齐后都应小于int32的最大值2147483647，且内轴大小需小于2097152。
    - 动态量化（T-T/T-C量化）场景支持的输入类型为：
      - 不为空的参数支持的数据类型组合要满足下表：

        | x1       | x2  | scale2 | scale1Optional |yRef     |
        |:-------:|:-------:| :------      | :------   | :------ |
        |HIFLOAT8  |HIFLOAT8| FLOAT32    | FLOAT32   | FLOAT32 |

      - scale1Optional/scale2要满足以下约束（其中g为matmul组数即分组数）：

        | 参数 | shape限制 |
        |:---------:| :------ |
        |scale1Optional| 2维tensor或1维tensor，shape为(g, 1)或(g,)|
        |scale2| pertensor场景：2维tensor或1维tensor，shape为(g, 1)或(g,)；perchannel场景：2维tensor，shape为(g, N)|

    - 动态量化（mx量化）场景支持的数据类型为：
      - 数据类型组合要满足下表：

        | x1       | x2  |  scale2  | scale1Optional |yRef     |
        |:-------:|:-------:| :-------    | :------   | :------ |
        |FLOAT8_E5M2/FLOAT8_E4M3FN  |FLOAT8_E5M2/FLOAT8_E4M3FN| FLOAT8_E8M0   | FLOAT8_E8M0    | FLOAT32 |

      - scale1Optional/scale2要满足以下约束（其中g为matmul组数即分组数，g\_i为第i个分组（下标从0开始））：

        | 参数 | shape限制 |
        |:---------:| :------ |
        |scale1Optional| 3维tensor，shape为((K / 64) + g, M, 2)，scale\_i起始地址偏移为((K\_0 + K\_1 + ...+ K\_{i-1})/ 64 + g\_i) \* M \* 2，即scale_0的起始地址偏移为0，scale_1的起始地址偏移为(K\_0 / 64 + 1) \* M \* 2， scale_2的起始地址偏移为((K\_0 + K\_1) / 64 + 2) \* M \* 2,依此类推|
        |scale2| 3维tensor，shape为((K / 64) + g, N, 2),起始地址偏移与scale1Optional同理|

## 调用说明

| 调用方式      | 调用样例                 | 说明                                                         |
|--------------|-------------------------|--------------------------------------------------------------|
| aclnn调用 | [test_aclnn_quant_grouped_matmul_inplace_add](examples/test_aclnn_quant_grouped_matmul_inplace_add.cpp) | 通过接口方式调用[QuantGroupedMatmulInplaceAdd](docs/aclnnQuantGroupedMatmulInplaceAdd.md)算子。 |
