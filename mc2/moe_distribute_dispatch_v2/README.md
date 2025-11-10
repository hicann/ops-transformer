# aclnnMoeDistributeDispatchV2

## 产品支持情况

| 产品                                                         |  是否支持   |
| :----------------------------------------------------------- |:-------:|
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √    |
| <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term> |    √    |
## 功能说明

算子功能：对token数据进行量化（可选），当存在TP域通信时，先进行EP（Expert Parallelism）域的AllToAllV通信，再进行TP（Tensor Parallelism）域的AllGatherV通信；当不存在TP域通信时，进行EP（Expert Parallelism）域的AllToAllV通信。
$$
agOut = AllGatherV(X)\\
expandXOut = AllToAllV(agOut)\\
$$

- <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>：该接口必须与aclnnMoeDistributeCombineV2配套使用。
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：该接口必须与aclnnMoeDistributeCombineV2或aclnnMoeDistributeCombineAddRmsNorm配套使用。

> 说明：
>
> aclnnMoeDistributeCombineV2、aclnnMoeDistributeCombineAddRmsNorm算子在后续文档中统称为CombineV2系列算子。

相较于aclnnMoeDistributeDispatch接口，该接口变更如下：

-   输出了更详细的token信息辅助CombineV2系列算子高效地进行全卡同步，因此原接口中shape为(Bs * K,)的expandIdx出参替换为shape为(A * 128,)的assistInfoForCombineOut参数；
-   新增commAlg入参，代替HCCL_INTRA_PCIE_ENABLE和HCCL_INTRA_ROCE_ENABLE环境变量。

详细说明请参考以下参数说明。

## 参数说明

<table style="undefined;table-layout: fixed; width: 1576px">
 <colgroup>
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
  </tr>
 </thead>
 <tbody>
  <tr>
   <td>x</td>
   <td>输入</td>
   <td>本卡发送的token数据，Device侧的aclTensor，要求为2D Tensor，shape为 (Bs, H)（Bs为batch size，H为hidden size（隐藏层大小））；支持非连续的Tensor。</td>
   <td>FLOAT16、BFLOAT16</td>
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
   <td>scalesOptional</td>
   <td>输入</td>
   <td>每个专家的量化平滑参数，Device侧的aclTensor，要求为2D Tensor，shape为 (sharedExpertNum + moeExpertNum, H)；非量化场景传空指针，动态量化可传有效数据或空指针；支持非连续的Tensor。</td>
   <td>FLOAT32</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>xActiveMaskOptional</td>
   <td>输入</td>
   <td>表示token是否参与通信，Device侧的aclTensor；支持非连续的Tensor；可传有效数据或空指针；1D时true需排在false前（例：{true, false, true}非法），2D时token对应K个值全为false则不参与通信；默认所有token参与通信；各卡BS不一致时所有token需有效。</td>
   <td>BOOL</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>expertScalesOptional</td>
   <td>输入</td>
   <td>每个token的topK个专家权重，Device侧的aclTensor；支持非连续的Tensor。</td>
   <td>FLOAT32</td>
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
   <td>EP通信域大小。</td>
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
   <td>MoE专家数量，满足moeExpertNum % (epWorldSize - sharedExpertRankNum) = 0。</td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>groupTp</td>
   <td>输入</td>
   <td>TP通信域名称（数据并行通信域）。</td>
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
   <td>表示共享专家数量（一个共享专家可复制部署到多个卡上）。</td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>sharedExpertRankNum</td>
   <td>输入</td>
   <td>表示共享专家卡数量，取值范围[0, epWorldSize)；为0时需满足sharedExpertNum为0或1，不为0时需满足sharedExpertRankNum % sharedExpertNum = 0。</td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>quantMode</td>
   <td>输入</td>
   <td>表示量化模式，支持0：非量化，2：动态量化。</td>
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
   <td>expertTokenNumsType</td>
   <td>输入</td>
   <td>输出expertTokenNums中值的语义类型，支持0：expertTokenNums中的输出为每个专家处理的token数的前缀和，1：expertTokenNums中的输出为每个专家处理的token数量。</td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>commAlg</td>
   <td>输入</td>
   <td>表示通信亲和内存布局算法，string数据类型。</td>
   <td>STRING</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>expandXOut</td>
   <td>输出</td>
   <td>根据expertIds进行扩展过的token特征，Device侧的aclTensor，要求为2D Tensor，shape为 (max(tpWorldSize, 1) * A, H)；支持非连续的Tensor。</td>
   <td>FLOAT16、BFLOAT16、INT8</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>dynamicScalesOut</td>
   <td>输出</td>
   <td>Device侧的aclTensor，要求为1D Tensor，shape为 (A, )；仅quantMode=2时有该输出；支持非连续的Tensor。</td>
   <td>FLOAT32</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>assistInfoForCombineOut</td>
   <td>输出</td>
   <td>表示给同一专家发送的token个数（对应CombineV2系列算子中的assistInfoForCombine），Device侧的aclTensor，要求为1D Tensor，shape为 (A*128, )；支持非连续的Tensor。</td>
   <td>INT32</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>expertTokenNumsOut</td>
   <td>输出</td>
   <td>表示每个专家收到的token个数，Device侧的aclTensor，要求为1D Tensor，shape为 (localExpertNum, )；支持非连续的Tensor。</td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>epRecvCountsOut</td>
   <td>输出</td>
   <td>从EP通信域各卡接收的token数（对应CombineV2系列算子中的epSendCounts），Device侧的aclTensor，要求为1D Tensor；支持非连续的Tensor。</td>
   <td>INT32</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>tpRecvCountsOut</td>
   <td>输出</td>
   <td>从TP通信域各卡接收的token数（对应CombineV2系列算子中的tpSendCounts），Device侧的aclTensor；有TP域通信则有该输出，无TP域通信则无该输出；有TP域通信时为1D Tensor；支持非连续的Tensor。</td>
   <td>INT32</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>expertScalesOut</td>
   <td>输出</td>
   <td>表示本卡输出token的权重（对应CombineV2系列算子中的expertScalesOptional），Device侧的aclTensor；支持非连续的Tensor。</td>
   <td>FLOAT32</td>
   <td>ND</td>
  </tr>
 </tbody>
</table>

## 约束说明

- aclnnMoeDistributeDispatchV2接口与CombineV2系列算子接口必须配套使用，具体参考调用示例。

- 在不同产品型号、不同通信算法或不同版本中，aclnnMoeDistributeDispatchV2的Tensor输出assistInfoForCombineOut、epRecvCounts、tpRecvCounts、expandScales中的元素值可能不同，使用时直接将上述Tensor传给aclnnMoeDistributeCombineV2对应参数即可，模型其他业务逻辑不应对其存在依赖。

- 调用接口过程中使用的groupEp、epWorldSize、moeExpertNum、groupTp、tpWorldSize、expertShardType、sharedExpertNum、sharedExpertRankNum、globalBs、commAlg参数，HCCL_BUFFSIZE取值所有卡需保持一致，网络中不同层中也需保持一致，且和CombineV2系列算子对应参数也保持一致。

- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：该场景下单卡包含双DIE（简称为“晶粒”或“裸片”），因此参数说明里的“本卡”均表示单DIE。

- 参数说明里shape格式说明：
    - A：表示本卡可能接收的最大token数量，取值范围如下：
        - 对于共享专家，要满足A = Bs * epWorldSize * sharedExpertNum / sharedExpertRankNum。
        - 对于MoE专家，当globalBs为0时，要满足A >= Bs * epWorldSize * min(localExpertNum, K)；当globalBs非0时，要满足A >= globalBs * min(localExpertNum, K)。
    - H：表示hidden size隐藏层大小。
        - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>：根据不同commAlg（见上文介绍）支持H不同的取值范围。
            - commAlg = "fullmesh": 取值范围(0, 7168]，且保证是32的整数倍。
            - commAlg = "hierarchy": 取值范围(0, 10 * 1024]，且保证是32的整数倍。
        - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：取值范围[1024, 8192]。
    - Bs：表示batch sequence size，即本卡最终输出的token数量。
        - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>：取值范围为0 < Bs ≤ 256。
        - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：取值范围为0 < Bs ≤ 512。
    - K：表示选取topK个专家，取值范围为0 < K ≤ 16同时满足0 < K ≤ moeExpertNum。
    - serverNum：表示服务器的节点数，取值只支持2、4、8。
    - localExpertNum：表示本卡专家数量。
        - 对于共享专家卡，localExpertNum = 1
        - 对于MoE专家卡，localExpertNum = moeExpertNum / (epWorldSize - sharedExpertRankNum)，localExpertNum > 1时，不支持TP域通信。

- HCCL_BUFFSIZE：
    调用本接口前需检查HCCL_BUFFSIZE环境变量取值是否合理，该环境变量表示单个通信域占用内存大小，单位MB，不配置时默认为200MB。
    - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>：
        - commAlg配置为""或nullptr：依照HCCL_INTRA_PCIE_ENABLE和HCCL_INTRA_ROCE_ENABLE环境变量配置，选择"fullmesh"或"hierarchy"公式。
        - commAlg配置为"fullmesh": 要求 >= 2 * (Bs * epWorldSize * min(localExpertNum, K) * H * sizeof(uint16) + 2MB)。
        - commAlg配置为"hierarchy": 要求 >= moeExpertNum * Bs * (H * sizeof(dtypeX) + 4 * ((K + 7) / 8 * 8) * sizeof(uint32)) + 4MB + 100MB，不要求moeExpertNum / (epWorldSize - sharedExpertRankNum) <= 24。
    - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：要求 >= 2且满足>= 2 * (localExpertNum * maxBs * epWorldSize * Align512(Align32(2 * H) + 64) + (K + sharedExpertNum) * maxBs * Align512(2 * H))，localExpertNum需使用MoE专家卡的本卡专家数，其中Align512(x) = ((x + 512 - 1) / 512) * 512，Align32(x) = ((x + 32 - 1) / 32) * 32。

- HCCL_INTRA_PCIE_ENABLE和HCCL_INTRA_ROCE_ENABLE：
    - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>：该环境变量不再推荐使用，建议commAlg配置"hierarchy"。

- 本文公式中的"/"表示整除。
- 通信域使用约束：
    - 一个模型中的CombineV2系列算子和aclnnMoeDistributeDispatchV2仅支持相同EP通信域，且该通信域中不允许有其他算子。
    - 一个模型中的CombineV2系列算子和aclnnMoeDistributeDispatchV2仅支持相同TP通信域或都不支持TP通信域，有TP通信域时该通信域中不允许有其他算子。

## 调用说明

| 调用方式  | 样例代码                                  | 说明                                                     |
| :--------: | :----------------------------------------: | :-------------------------------------------------------: |
| aclnn接口 | [test_aclnn_moe_distribute_dispatch_v2.cpp](./examples/test_aclnn_moe_distribute_dispatch_v2.cpp) | 通过[aclnnMoeDistributeDispatchV2](./docs/aclnnMoeDistributeDispatchV2.md)接口方式调用moe_distribute_dispatch_v2算子。 |
