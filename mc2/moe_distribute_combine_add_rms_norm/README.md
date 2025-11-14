# MoeDistributeCombineAddRmsNorm

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term> |    ×     |

## 功能说明

**算子功能**：当存在TP域通信时，先进行ReduceScatterV通信，再进行AllToAllV通信，最后将接收的数据整合（乘权重再相加）；当不存在TP域通信时，进行AllToAllV通信，最后将接收的数据整合（乘权重再相加），之后完成Add + RmsNorm融合。

**计算公式**：
$$
rsOut = ReduceScatterV(expandX)\\
ataOut = AllToAllV(rsOut)\\
combineOut = Sum(expertScales * ataOut + expertScales * sharedExpertX)\\
x = combineOut + residualX\\
y = \frac{x}{RMS(x)} * gamma,\quad\text{where}RMS(x) = \sqrt{\frac{1}{H}\sum_{i=1}^{H}x_{i}^{2}+normEps}
$$

注意该接口必须与aclnnMoeDistributeDispatchV2配套使用，相当于按MoeDistributeDispatchV2算子收集数据的路径原路返还。


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
   <td>expandX</td>
   <td>输入</td>
   <td>根据expertIds进行扩展过的token特征，Device侧的aclTensor，要求为2D Tensor，shape为 (max(tpWorldSize, 1) * A , H)；支持非连续的Tensor。</td>
   <td>BFLOAT16</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>expertIds</td>
   <td>输入</td>
   <td>每个token的topK个专家索引，Device侧的aclTensor，要求为2D Tensor，shape为 (Bs, K)；支持非连续的Tensor。</td>
   <td>INT32</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>assistInfoForCombine</td>
   <td>输入</td>
   <td>对应aclnnMoeDistributeDispatchV2中的assistInfoForCombineOut输出，Device侧的aclTensor，要求为1D Tensor，shape为 (A * 128, )；支持非连续的Tensor。</td>
   <td>INT32</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>epSendCounts</td>
   <td>输入</td>
   <td>对应aclnnMoeDistributeDispatchV2中的epRecvCounts输出，Device侧的aclTensor，要求为1D Tensor，shape为 (epWorldSize * max(tpWorldSize, 1) * localExpertNum, )；支持非连续的Tensor。</td>
   <td>INT32</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>expertScales</td>
   <td>输入</td>
   <td>每个token的topK个专家的权重，Device侧的aclTensor，要求为2D Tensor，shape为 (Bs, K)；支持非连续的Tensor。</td>
   <td>FLOAT32</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>residualX</td>
   <td>输入</td>
   <td>AddRmsNorm中Add的右矩阵，Device侧的aclTensor，要求为3D Tensor，shape为 (Bs，1，H)；支持非连续的Tensor。</td>
   <td>BFLOAT16</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>gamma</td>
   <td>输入</td>
   <td>RmsNorm中的gamma输入，Device侧的aclTensor，要求为1D Tensor，shape为 (H, )；支持非连续的Tensor。</td>
   <td>BFLOAT16</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>tpSendCountsOptional</td>
   <td>输入</td>
   <td>对应aclnnMoeDistributeDispatchV2中的tpRecvCounts输出，Device侧的aclTensor；有TP域通信需传参，无TP域通信传空指针；有TP域通信时为1D Tensor，shape为 (tpWorldSize, )；支持非连续的Tensor。</td>
   <td>INT32</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>xActiveMaskOptional</td>
   <td>输入</td>
   <td>表示token是否参与通信，Device侧的aclTensor，要求为1D或2D Tensor；1D时shape为 (BS, )，2D时shape为 (BS, K)；可传有效数据或空指针；1D时true表示token参与通信且true需排在false前（例：{true, false, true}非法），2D时true表示token对应expert_ids参与通信，token对应K个值全为false则不参与通信；默认所有token参与通信；各卡BS不一致时所有token需有效；支持非连续的Tensor。</td>
   <td>BOOL</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>activationScaleOptional</td>
   <td>输入</td>
   <td>Device侧的aclTensor，预留参数，当前版本不支持，传空指针即可。</td>
   <td>-</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>weightScaleOptional</td>
   <td>输入</td>
   <td>Device侧的aclTensor，预留参数，当前版本不支持，传空指针即可。</td>
   <td>-</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>groupListOptional</td>
   <td>输入</td>
   <td>Device侧的aclTensor，预留参数，当前版本不支持，传空指针即可。</td>
   <td>-</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>expandScalesOptional</td>
   <td>输入</td>
   <td>对应aclnnMoeDistributeDispatchV2中的expandScales输出，Device侧的aclTensor；预留参数，当前版本不支持，传空指针即可。</td>
   <td>-</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>sharedExpertXOptional</td>
   <td>输入</td>
   <td>表示共享专家计算后的Token，Device侧的aclTensor；要求为2D或3D Tensor，2D时shape为 (Bs, H)，3D时shape为 (Bs, 1, H)；数据类型需与expandX一致；可传或不传；支持非连续的Tensor。</td>
   <td>BFLOAT16</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>groupEp</td>
   <td>输入</td>
   <td>EP通信域名称（专家并行通信域），字符串长度范围为[1, 128)，不能和groupTp相同。</td>
   <td>STRING</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>epWorldSize</td>
   <td>输入</td>
   <td>EP通信域大小，取值支持(1, 768]。</td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>epRankId</td>
   <td>输入</td>
   <td>EP域本卡Id，取值范围[0, epWorldSize)，同一个EP通信域中各卡的epRankId不重复。</td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>moeExpertNum</td>
   <td>输入</td>
   <td>MoE专家数量，取值范围(0, 1024]，且满足moeExpertNum % (epWorldSize - sharedExpertRankNum) = 0。</td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>groupTp</td>
   <td>输入</td>
   <td>TP通信域名称（数据并行通信域），字符串长度范围为[1, 128)，不能和groupEp相同。</td>
   <td>STRING</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>tpWorldSize</td>
   <td>输入</td>
   <td>TP通信域大小，取值范围[0, 2]，0和1表示无TP域通信，有TP域通信时仅支持2。</td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>tpRankId</td>
   <td>输入</td>
   <td>TP域本卡Id，取值范围[0, 1]，同一个TP通信域中各卡的tpRankId不重复；无TP域通信时传0即可。</td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>expertShardType</td>
   <td>输入</td>
   <td>表示共享专家卡分布类型，当前仅支持传0，表示共享专家卡排在MoE专家卡前面。</td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>sharedExpertNum</td>
   <td>输入</td>
   <td>表示共享专家数量，当前版本不支持，传0即可。</td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>sharedExpertRankNum</td>
   <td>输入</td>
   <td>表示共享专家卡数量，当前版本不支持，仅支持传入0。</td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>globalBs</td>
   <td>输入</td>
   <td>EP域全局的batch size大小；各rank Bs一致时，globalBs = Bs * epWorldSize 或 0；各rank Bs不一致时，globalBs = maxBs * epWorldSize（maxBs为单卡Bs最大值）。</td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>outDtype</td>
   <td>输入</td>
   <td>用于指定输出x的数据类型，预留参数，当前版本不支持，传0即可。</td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>commQuantMode</td>
   <td>输入</td>
   <td>通信量化类型，当前版本不支持，传0即可。</td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>groupListType</td>
   <td>输入</td>
   <td>group List格式，预留参数，当前版本不支持，传0即可。</td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>commAlg</td>
   <td>输入</td>
   <td>表示通信亲和内存布局算法，string数据类型；预留字段，当前版本不支持，传入空指针即可。</td>
   <td>STRING</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>normEps</td>
   <td>输入</td>
   <td>用于防止AddRmsNorm除0错误，可取值为1e-6。</td>
   <td>FLOAT</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>yOut</td>
   <td>输出</td>
   <td>RmsNorm后的输出结果，Device侧的aclTensor，要求为3D Tensor，shape为（Bs，1，H）；数据类型、数据格式与residualX保持一致。</td>
   <td>BFLOAT16</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>rstdOut</td>
   <td>输出</td>
   <td>RmsNorm后的输出结果，Device侧的aclTensor，要求为3D Tensor，shape为（Bs，1，1）。</td>
   <td>FLOAT32</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>xOut</td>
   <td>输出</td>
   <td>Add后的输出结果，Device侧的aclTensor，要求为3D Tensor，shape为 (Bs, 1，H)；数据类型、数据格式与residualX保持一致。</td>
   <td>BFLOAT16</td>
   <td>ND</td>
  </tr>
 </tbody>
</table>

    

## 约束说明

- aclnnMoeDistributeDispatchV2接口与aclnnMoeDistributeCombineAddRmsNorm接口必须配套使用，具体参考调用示例。

- 调用接口过程中使用的groupEp、epWorldSize、moeExpertNum、groupTp、tpWorldSize、expertShardType、sharedExpertNum、sharedExpertRankNum、globalBs参数取值所有卡需保持一致，网络中不同层中也需保持一致，且和aclnnMoeDistributeDispatchV2对应参数也保持一致。

- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：该场景下单卡包含双DIE（简称为“晶粒”或“裸片”），因此参数说明里的“本卡”均表示单DIE。

- 参数说明里shape格式说明：
    - A：表示本卡需要分发的最大token数量，取值范围如下：
        - 对于共享专家，要满足A = Bs * epWorldSize * sharedExpertNum / sharedExpertRankNum。
        - 对于MoE专家，当globalBs为0时，要满足A >= Bs * epWorldSize * min(localExpertNum, K)；当globalBs非0时，要满足A >= globalBs * min(localExpertNum, K)。
    - H：表示hidden size隐藏层大小，取值范围为[1024, 8192]。
    - Bs：表示batch sequence size，即本卡最终输出的token数量，取值范围为0 < Bs ≤ 512。
    - K：表示选取topK个专家，取值范围为0 < K ≤ 16同时满足0 < K ≤ moeExpertNum。
    - localExpertNum：表示本卡专家数量。
        - 对于共享专家卡，localExpertNum = 1
        - 对于MoE专家卡，localExpertNum = moeExpertNum / (epWorldSize - sharedExpertRankNum)，localExpertNum > 1时，不支持TP域通信。
  
- HCCL_BUFFSIZE：
    调用本接口前需检查HCCL_BUFFSIZE环境变量取值是否合理，该环境变量表示单个通信域占用内存大小，单位MB，不配置时默认为200MB。要求 >= 2且满足1024 ^ 2 * (HCCL_BUFFSIZE - 2) / 2 >= (BS * 2 * (H + 128) * (epWorldSize * localExpertNum + K + 1)，localExpertNum需使用MoE专家卡的本卡专家数。

- 通信域使用约束：
    - 一个模型中的aclnnMoeDistributeCombineAddRmsNorm和aclnnMoeDistributeDispatchV2仅支持相同EP通信域，且该通信域中不允许有其他算子。
    - 一个模型中的aclnnMoeDistributeCombineAddRmsNorm和aclnnMoeDistributeDispatchV2仅支持相同TP通信域或都不支持TP通信域，有TP通信域时该通信域中不允许有其他算子。

## 调用说明

| 调用方式  | 样例代码                                  | 说明                                                     |
| :--------: | :----------------------------------------: | :-------------------------------------------------------: |
| aclnn接口 | [test_aclnn_moe_distribute_combine_add_rms_norm.cpp](./examples/test_aclnn_moe_distribute_combine_add_rms_norm.cpp) | 通过[aclnnMoeDistributeCombineAddRmsNorm](./docs/aclnnMoeDistributeCombineAddRmsNorm.md)接口方式调用moe_distribute_combine_add_rms_norm算子。 |


