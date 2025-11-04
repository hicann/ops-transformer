# MoeDistributeBufferReset


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

算子功能：在故障检测流程中，对EP通信域做数据区与状态区的清理。若当前机器为未被隔离机器，则对其进行通信域的重置操作，对有效的die进行数据区和状态区的清0，确保后续使用时通信域不会存在已被隔离机器的数据或状态信息。



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
   <td>elasticInfo</td>
   <td>输入</td>
   <td>有效rank掩码表，标识有效rank的tensor，要求为一个1D的Tensor，shape为(epWorldSize,)。</td>
   <td>INT32</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>groupEp</td>
   <td>属性</td>
   <td>标识EP通信域名称（专家并行通信域），字符串长度范围为[1, 128)。</td>
   <td>STRING</td>
   <td>-</td>
  </tr>
  <tr>
   <td>epWorldSize</td>
   <td>属性</td>
   <td>EP通信域大小。</td>
   <td>INT32</td>
   <td>-</td>
  </tr>
  <tr>
   <td>needSync</td>
   <td>属性</td>
   <td>是否需要全卡同步，0表示不需要，1表示需要。</td>
   <td>INT32</td>
   <td>-</td>
  </tr>
 </tbody></table>




## 约束说明

无

## 调用说明

| 调用方式  | 样例代码                                  | 说明                                                     |
| :--------: | :----------------------------------------: | :-------------------------------------------------------: |
| aclnn接口 | [test_aclnn_moe_distribute_buffer_reset.cpp](./examples/test_aclnn_moe_distribute_buffer_reset.cpp) | 通过[aclnnMoeDistributeBufferReset](./docs/aclnnMoeDistributeBufferReset.md)接口方式调用moe_distribute_buffer_reset算子。 |