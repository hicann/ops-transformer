# GroupedMatMulAllReduce

> 注意：
> 本文档仅仅是算子功能的简介，不支持用户直接调用，因为当前不支持kernel直调，等后续支持再完善文档!!!!!!

**该接口后续版本会废弃，请不要使用该接口。**

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT系列产品</term>                             |    ×     |
| <term>Atlas A3系列产品</term>       |    ×     |
| <term>Atlas A2系列产品</term> |    √     |
| <term>Atlas 200I/500 A2推理产品</term>                      |    ×     |
| <term>Atlas推理系列产品</term>                               |    ×     |
| <term>Atlas训练系列产品</term>                              |    ×     |

## 功能说明

算子功能：在grouped_matmul的基础上实现多卡并行AllReduce功能，实现分组矩阵乘计算，每组矩阵乘的维度大小可以不同。根据x、weight、y的Tensor数量支持如下4种场景：

- x、weight、y的Tensor数量等于组数，即每组的数据对应的Tensor是独立的。
- x的Tensor数量为1，weight/y的Tensor数量等于组数，此时需要通过可选参数group_list说明x在行上的分组情况，如group_list[0]=10说明x的前10行参与第一组矩阵乘计算。
- x、weight的Tensor数量等于组数，y的Tensor数量为1，此时每组矩阵乘的结果放在同一个Tensor中连续存放。
- x、y的Tensor数量为1，weight数量等于组数，属于前两种情况的组合。

计算公式：
非量化场景：

$$
y_i=x_i\times weight_i + bias_i
$$

## 参数说明

<table style="undefined;table-layout: fixed; width: 1576px"> <colgroup>
 <col style="width: 170px">
 <col style="width: 170px">
 <col style="width: 800px">
 <col style="width: 800px">
 <col style="width: 200px">
 </colgroup>
 <thead>
  <tr>
   <th>参数名</th>
   <th>输入/输出</th>
   <th>描述</th>
   <th>数据类型</th>
   <th>数据格式</th>
  </tr></thead>
 <tbody>
  <tr>
   <td>x</td>
   <td>输入</td>
   <td>必选参数，Device侧的aclTensorList，公式中的输入x，支持的最大长度为64个。</td>
   <td>FLOAT16、BFLOAT16（列表内张量数据类型）</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>weight</td>
   <td>输入</td>
   <td>必选参数，Device侧的aclTensorList，公式中的weight，支持的最大长度为64个。</td>
   <td>FLOAT16、BFLOAT16（列表内张量数据类型）</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>bias</td>
   <td>输入</td>
   <td>可选参数，Device侧的aclTensorList，公式中的bias，支持的最大长度为64个。</td>
   <td>FLOAT16、FLOAT32（列表内张量数据类型）</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>groupListOptional</td>
   <td>输入</td>
   <td>可选参数，Host侧的aclIntArray类型，代表输入和输出M方向的matmul大小分布，支持的最大长度为64个。</td>
   <td>INT64（数组元素类型）</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>splitItem</td>
   <td>输入</td>
   <td>可选属性，代表输入和输出是否要做tensor切分：0（默认）=输入输出都不切分；1=输入切分、输出不切分；2=输入不切分、输出切分；3=输入输出都切分。</td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>group</td>
   <td>输入</td>
   <td>Host侧标识列组的字符串，即通信域名称，通过Hccl接口HcclGetCommName获取commName作为该参数。</td>
   <td>STRING</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>reduceOp</td>
   <td>输入</td>
   <td>reduce操作类型，当前版本仅支持输入"sum"。</td>
   <td>STRING</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>commTurn</td>
   <td>输入</td>
   <td>Host侧整型，通信数据切分数（总数据量/单次通信量），当前版本仅支持输入0。</td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>streamMode</td>
   <td>输入</td>
   <td>Host侧整型，acl流模式的枚举，当前只支持值1。</td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>y</td>
   <td>输出</td>
   <td>Device侧的aclTensorList，公式中的输出y，支持的最大长度为64个。</td>
   <td>FLOAT16、BFLOAT16（列表内张量数据类型）</td>
   <td>ND</td>
  </tr>
 </tbody></table>

## 约束说明

- 数据类型组合约束：x、weight、bias支持的数据类型组合为：
  -“x-FLOAT16、weight-FLOAT16、bias-FLOAT16”
  -“x-BFLOAT16、weight-BFLOAT16、bias-FLOAT32”
- 维度约束：
  - 当splitItem为0时，x支持2-6维，y支持2-6维；
  - 当splitItem为1/2/3时，x支持2维，y支持2维；
  - 无论splitItem取值如何，weight均支持2维。
- 设备数量约束：支持2、4、8卡部署。
- 张量维度大小约束：
  - x和weight中每一组tensor的最后一维大小都应小于65536（x的最后一维：transpose_x为false时指K轴，为true时指M轴；weight的最后一维：transpose_weight为false时指N轴，为true时指K轴）。
  - x和weight中每一组tensor的每一维大小在32字节对齐后都应小于int32最大值（2147483647）。

## 调用说明

| 调用方式  | 样例代码                                  | 说明                                                     |
| :--------: | :----------------------------------------: | :-------------------------------------------------------: |
| aclnn接口 | [test_grouped_mat_mul_all_reduce.cpp](tests/ut/op_kernel/test_grouped_mat_mul_all_reduce.cpp) | 通过[aclnnGroupedMatMulAllReduce](./docs/aclnnGroupedMatMulAllReduce.md)接口方式调用grouped_mat_mul_all_reduce算子。 |
