# ElasticReceivableInfoCollect


## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>昇腾910_95 AI处理器</term>                             |    ×     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term> |    ×     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品 </term>                             |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×     |
| <term>Atlas 200/300/500 推理产品</term>                      |    ×     |

## 功能说明

算子功能：在快速恢复检测流程中，收集并整理检测通信域内aclnnElasticReceivableTest发送的状态位，并将结果输出，供下一步检测流程判断全局联通情况。



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
   <th>输入/输出/属性</th>
   <th>描述</th>
   <th>数据类型</th>
   <th>数据格式</th>
  </tr></thead>
 <tbody>
  <tr>
   <td>group</td>
   <td>输入</td>
   <td>ep通信域名称，专家并行的通信域。</td>
   <td>STRING</td>
   <td>-</td>
  </tr>
  <tr>
   <td>worldSize</td>
   <td>输入</td>
   <td>通信域大小。</td>
   <td>INT64</td>
   <td>-</td>
  </tr>
  <tr>
  <tr>
   <td>y</td>
   <td>输出</td>
   <td>设备互联信息，Device侧的aclTensor，要求为一个2D的Tensor，shape为(worldSize, worldSize)，数据类型支持INT32，数据格式要求为ND，支持非连续的Tensor。</td>
   <td>INT32</td>
   <td>ND</td>
  </tr>
 </tbody></table>




## 约束说明

无

## 调用说明

| 调用方式  | 样例代码                                  | 说明                                                     |
| :--------: | :----------------------------------------: | :-------------------------------------------------------: |
| aclnn接口 | [test_aclnn_elastic_receivable_info_collect.cpp](./examples/test_aclnn_elastic_receivable_info_collect.cpp) | 通过[aclnnElasticReceivableInfoCollect](./docs/aclnnElasticReceivableInfoCollect.md)接口方式调用elastic_receivable_info_collect算子。 |