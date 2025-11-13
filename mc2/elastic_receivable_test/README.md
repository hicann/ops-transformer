# ElasticReceivableTest


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

算子功能：对一个通信域内的所有卡发送数据并写状态位，用于检测通信链路是否正常。



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
   <td>dstRank</td>
   <td>输入</td>
   <td>表示同一个通信域内目的server内的通信卡，Device侧的aclTensor，要求为一个1D的Tensor，shape为(rankNum,)。</td>
   <td>INT32</td>
   <td>ND</td>
  </tr>
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
   <td>rankNum</td>
   <td>输入</td>
   <td>需要发送对端server内的卡数。</td>
   <td>INT64</td>
   <td>-</td>
  </tr>
 </tbody></table>




## 约束说明

无

## 调用说明

| 调用方式  | 样例代码                                  | 说明                                                     |
| :--------: | :----------------------------------------: | :-------------------------------------------------------: |
| aclnn接口 | [test_aclnn_elastic_receivable_test.cpp](./examples/test_aclnn_elastic_receivable_test.cpp) | 通过[aclnnElasticReceivableTest](./docs/aclnnElasticReceivableTest.md)接口方式调用elastic_receivable_test算子。 |