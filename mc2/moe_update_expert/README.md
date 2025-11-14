# MoeUpdateExpert

## 产品支持情况

| 产品                                                         |  是否支持   |
| :----------------------------------------------------------- |:-------:|
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √    |
| <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term> |    ×    |

## 功能说明

**算子功能**：
* 为了解决负载不均衡的场景，该算子可以完成每个token的topK个专家逻辑专家号到物理卡号的映射。
* 支持根据阈值对token发送的topK个专家进行剪枝。

## 参数说明

<table style="undefined;table-layout: fixed; width: 1392px"> <colgroup>
 <col style="width: 120px">
 <col style="width: 120px">
 <col style="width: 160px">
 <col style="width: 150px">
 <col style="width: 80px">
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
   <td>expertIds</td>
   <td>输入</td>
   <td>每个token的topK个专家索引，Device侧的aclTensor，要求为2D Tensor，shape为 (BS, K)；支持非连续的Tensor。</td>
   <td>INT32、INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>eplbTable</td>
   <td>输入</td>
   <td>逻辑专家到物理专家的映射表，外部调用者需保证输入Tensor的值正确；world_size张卡，每张卡部署place_per_rank个路由专家实例，共world_size*place_per_rank个实例；每行第一列为行号对应逻辑专家部署的实例数count（取值范围[1, world_size]），每行[1, count]列为对应的实例编号（取值范围[0, world_size*place_per_rank)，有效实例编号不可重复）；Device侧的aclTensor，要求为2D Tensor，shape为 (moeExpertNum, F)；支持非连续的Tensor。</td>
   <td>INT32</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>expertScalesOptional</td>
   <td>输入</td>
   <td>每个token的topK个专家的scale权重，用户需保证scale在token内部按降序排列；可传有效数据或空指针，传有效数据时pruningThresholdOptional也需传有效数据；Device侧的aclTensor，要求为2D Tensor，shape为 (BS, K)；支持非连续的Tensor。</td>
   <td>FLOAT16、BFLOAT16、FLOAT</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>pruningThresholdOptional</td>
   <td>输入</td>
   <td>专家scale权重的最小阈值，当token对应topK专家scale小于阈值时，该token对该专家剪枝（不发送至该专家处理）；可传有效数据或空指针，传有效数据时expertScalesOptional也需传有效数据；Device侧的aclTensor，要求为1D或2D Tensor，shape为(K,)或(1, K)；支持非连续的Tensor。</td>
   <td>FLOAT</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>activeMaskOptional</td>
   <td>输入</td>
   <td>表示token是否参与通信；可传有效数据或空指针，传有效数据时expertScalesOptional、pruningThresholdOptional也必须传有效数据；true表示token参与通信且true需排在false前（例：{true, false, true}非法），传空指针表示所有token参与通信；Device侧的aclTensor，要求为1D Tensor，shape为 (BS，)；支持非连续的Tensor。</td>
   <td>BOOL</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>localRankId</td>
   <td>输入</td>
   <td>本卡Id；balanceMode为0时，取值范围为[0, worldSize)；同一个通信域中各卡的localRankId不重复。</td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>worldSize</td>
   <td>输入</td>
   <td>通信域Size；balanceMode为0时，取值范围为[2, 768]。</td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>balanceMode</td>
   <td>输入</td>
   <td>均衡规则，0表示按rank分发，1表示按token分发，默认值为0；取值范围为[0, 1]。</td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>balancedExpertIds</td>
   <td>输出</td>
   <td>映射后每个token的topK个专家所在物理专家的实例编号，Device侧的aclTensor，要求为2D Tensor，shape为（BS, K）；数据类型、数据格式与expertIds保持一致。</td>
   <td>INT32、INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>balancedActiveMask</td>
   <td>输出</td>
   <td>剪枝后均衡的activeMask，expertScalesOptional、pruningThresholdOptional传入有效数据时该输出有效；Device侧的aclTensor，要求为2D Tensor，shape为（BS, K）；支持非连续的Tensor。</td>
   <td>BOOL</td>
   <td>ND</td>
  </tr>
 </tbody>
</table>

## 约束说明

- aclnnMoeUpdateExpert接口必须与aclnnMoeDistributeDispatchV2及aclnnMoeDistributeCombineV2或aclnnMoeDistributeCombineAddRmsNorm接口配套使用，调用顺序为aclnnMoeUpdateExpert，aclnnMoeDistributeDispatchV2，aclnnMoeDistributeCombineV2或aclnnMoeDistributeCombineAddRmsNorm，具体参考调用示例。

- 调用接口过程中使用的worldSize、moeExpertNum参数取值所有卡需保持一致，网络中不同层中也需保持一致，且和aclnnMoeDistributeDispatchV2,aclnnMoeDistributeCombineV2或aclnnMoeDistributeCombineAddRmsNorm对应参数也保持一致。

- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：该场景下单卡包含双DIE（简称为“晶粒”或“裸片”），因此参数说明里的“本卡”均表示单DIE。

- 参数说明里shape格式说明：
    - BS：表示batch sequence size，即本卡最终输出的token数量，取值范围为0 < BS ≤ 512。
    - K：表示选取topK个专家，取值范围为0 < K ≤ 16同时满足0 < K ≤ moeExpertNum。
    - moeExpertNum：表示Moe专家数量，取值范围(0, 1024]。
    - F: 表示映射表的列数，取值范围[2, worldSize + 1]，第一列为各行号对应Moe专家部署的实例个数(值>0)，后F-1列为该Moe专家部署的物理卡号。
    - 所有卡部署的moe专家实例总数最多1024，即place_per_rank * world_size ≤ 1024。
    - 每张卡部署的实例数需相同。


## 调用说明

| 调用方式  | 样例代码                                  | 说明                                                     |
| :--------: | :----------------------------------------: | :-------------------------------------------------------: |
| aclnn接口 | [test_aclnn_moe_update_expert.cpp](./examples/test_aclnn_moe_update_expert.cpp) | 通过[aclnnMoeUpdateExpert](./docs/aclnnMoeUpdateExpert.md)接口方式调用moe_update_expert算子。 |


