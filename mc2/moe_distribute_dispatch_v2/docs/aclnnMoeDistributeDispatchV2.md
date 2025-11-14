# aclnnMoeDistributeDispatchV2

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term> |    √     |

## 功能说明

- 算子功能：对token数据进行量化（可选），当存在TP域通信时，先进行EP（Expert Parallelism）域的AllToAllV通信，再进行TP（Tensor Parallelism）域的AllGatherV通信；当不存在TP域通信时，进行EP（Expert Parallelism）域的AllToAllV通信。
- 计算公式：
$$
agOut = AllGatherV(X)\\
expandXOut = AllToAllV(agOut)
$$

- <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>：该接口必须与`aclnnMoeDistributeCombineV2`配套使用。
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：该接口必须与`aclnnMoeDistributeCombineV2`或`aclnnMoeDistributeCombineAddRmsNorm`配套使用。

> 说明：
> `aclnnMoeDistributeCombineV2`、`aclnnMoeDistributeCombineAddRmsNorm`算子在后续文档中统称为**CombineV2系列算子**。

### 接口变更说明
相较于`aclnnMoeDistributeDispatch`接口，该接口变更如下：
1. 输出了更详细的token信息辅助CombineV2系列算子高效地进行全卡同步，因此原接口中shape为`(Bs * K,)`的`expandIdx`出参替换为shape为`(A * 128,)`的`assistInfoForCombineOut`参数；
2. 新增`commAlg`入参，代替`HCCL_INTRA_PCIE_ENABLE`和`HCCL_INTRA_ROCE_ENABLE`环境变量。

详细说明请参考以下参数说明。

## 函数原型

每个算子分为两段式接口，必须先调用 “aclnnMoeDistributeDispatchV2GetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnMoeDistributeDispatchV2”接口执行计算。

```cpp
aclnnStatus aclnnMoeDistributeDispatchV2GetWorkspaceSize(
    const aclTensor* x,
    const aclTensor* expertIds,
    const aclTensor* scalesOptional,
    const aclTensor* xActiveMaskOptional,
    const aclTensor* expertScalesOptional,
    const char*      groupEp,
    int64_t          epWorldSize,
    int64_t          epRankId,
    int64_t          moeExpertNum,
    const char*      groupTp,
    int64_t          tpWorldSize,
    int64_t          tpRankId,
    int64_t          expertShardType,
    int64_t          sharedExpertNum,
    int64_t          sharedExpertRankNum,
    int64_t          quantMode,
    int64_t          globalBs,
    int64_t          expertTokenNumsType,
    const char*      commAlg,
    aclTensor*       expandXOut,
    aclTensor*       dynamicScalesOut,
    aclTensor*       assistInfoForCombineOut,
    aclTensor*       expertTokenNumsOut,
    aclTensor*       epRecvCountsOut,
    aclTensor*       tpRecvCountsOut,
    aclTensor*       expandScalesOut,
    uint64_t*        workspaceSize,
    aclOpExecutor**  executor)
```

```cpp
aclnnStatus aclnnMoeDistributeDispatchV2(
    void            *workspace,
    uint64_t        workspaceSize,
    aclOpExecutor   *executor,
    aclrtStream     stream)
```

## aclnnMoeDistributeDispatchV2GetWorkspaceSize

### 参数说明

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
   <th>输入/输出</th>
   <th>描述</th>
   <th>数据类型</th>
   <th>数据格式</th>
  </tr>
 </thead>
 <tbody>
  <tr>
   <td>x</td>
   <td>输入</td>
   <td>表示本卡发送的token数据，Device侧的aclTensor，要求为2D Tensor，shape为 (Bs, H)（Bs为batch size，H为隐藏层大小）；支持非连续的Tensor。</td>
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
   <td>表示token是否参与通信，Device侧的aclTensor；可传有效数据或空指针，默认所有token参与通信；各卡BS不一致时所有token需有效；支持非连续的Tensor。</td>
   <td>BOOL</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>expertScalesOptional</td>
   <td>输入</td>
   <td>每个token的topK个专家权重，Device侧的aclTensor；支持非连续的Tensor。</td>
   <td><term>Atlas A2系列</term>：FLOAT32</td>
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
   <td>TP通信域大小。</td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>tpRankId</td>
   <td>输入</td>
   <td>TP域本卡Id。</td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>expertShardType</td>
   <td>输入</td>
   <td>表示共享专家卡分布类型。</td>
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
   <td>表示共享专家卡数量。</td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>quantMode</td>
   <td>输入</td>
   <td>表示量化模式，支持0（非量化）、2（动态量化）。</td>
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
   <td>输出expertTokenNums中值的语义类型（0：每个专家处理token数的前缀和，1：每个专家处理的token数量）。</td>
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
   <td>Device侧的aclTensor，要求为1D Tensor，shape为 (A, )；支持非连续的Tensor；仅当quantMode=2时有输出。</td>
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
   <td>从TP通信域各卡接收的token数（对应CombineV2系列算子中的tpSendCounts），Device侧的aclTensor；有TP域通信时有输出，无TP域通信时无输出。</td>
   <td>INT32</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>expandScalesOut</td>
   <td>输出</td>
   <td>表示本卡输出token的权重（对应CombineV2系列算子中的expertScalesOptional），Device侧的aclTensor；支持非连续的Tensor。</td>
   <td>FLOAT32</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>workspaceSize</td>
   <td>输出</td>
   <td>返回需要在Device侧申请的workspace大小。</td>
   <td>UINT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>executor</td>
   <td>输出</td>
   <td>返回op执行器，包含了算子的计算流程。</td>
   <td>aclOpExecutor*</td>
   <td>ND</td>
  </tr>
 </tbody>
</table>

- <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>：
    - commAlg为"hierarchy"或HCCL_INTRA_PCIE_ENABLE=1且HCCL_INTRA_ROCE_ENABLE=0时，scalesOptional 需传nullptr。
    - xActiveMaskOptional 要求为1D Tensor，shape为(BS, )；true需排在false前（例：{true, false, true}非法）。
    - expertScalesOptional 要求为2D Tensor，shape为(Bs, K)。
    - epWorldSize 依commAlg取值，"fullmesh"支持16、32、64、128、256；"hierarchy"支持16、32、64。
    - moeExpertNum 取值范围(0, 512]，还需满足moeExpertNum / (epWorldSize - sharedExpertRankNum) <= 24。
    - groupTp 当前版本不支持，传空字符即可。
    - tpWorldSize、tpRankId、expertShardType、sharedExpertNum、sharedExpertRankNum 当前版本不支持，传0即可。
    - commAlg 支持nullptr、""、"fullmesh"、"hierarchy"；推荐配置"hierarchy"并搭配≥25.0.RC1.1版本驱动；nullptr和""依HCCL环境变量选择算法（不推荐）；"fullmesh"通过RDMA直传token；"hierarchy"经跨机、机内两次发送优化通信。
    - epRecvCountsOut 的shape为(moeExpertNum + 2globalBsK*serverNum,)（前moeExpertNum个为接收token数，剩余为通信前reduce相关信息）。
    - 当前不支持TP域通信。
    - expandScalesOut 要求为1D Tensor，shape为(A,)。

- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：
    - xActiveMaskOptional 要求为1D或2D Tensor（1D时shape为(BS, )，2D时shape为(BS, K)）；1D时true需排在false前，2D时token对应K个值全为false则不参与通信。
    - expertScalesOptional 当前版本不支持，传空指针即可。
    - epWorldSize 取值范围[2, 768]。
    - moeExpertNum 取值范围(0, 1024]。
    - groupTp 字符串长度范围为[1, 128)，不能和groupEp相同。
    - tpWorldSize 取值范围[0, 2]，0和1表示无TP域通信，有TP域通信时仅支持2。
    - tpRankId 取值范围[0, 1]，同一个TP通信域中各卡的tpRankId不重复；无TP域通信时传0即可。
    - expertShardType 当前仅支持传0，表示共享专家卡排在MoE专家卡前面。
    - sharedExpertNum 当前取值范围[0, 4]。
    - sharedExpertRankNum 取值范围[0, epWorldSize)；为0时需满足sharedExpertNum为0或1，不为0时需满足sharedExpertRankNum % sharedExpertNum = 0。
    - commAlg 当前版本不支持，传空指针即可。
    - epRecvCountsOut 的shape为(epWorldSize * max(tpWorldSize, 1) * localExpertNum,)。
    - 有TP域通信时为1D Tensor，tpRecvCountsOut 的shape为(tpWorldSize,)；支持非连续的Tensor。
    - expandScalesOut 当前版本不支持该输出。

### 返回值

返回aclnnStatus状态码，具体参见aclnn返回码。

第一段接口完成入参校验，出现以下场景时报错：

<table style="undefined;table-layout: fixed; width: 1180px"> <colgroup>
 <col style="width: 250px">
 <col style="width: 130px">
 <col style="width: 800px">
 </colgroup>
 <thead>
  <tr>
   <th>返回值</th>
   <th>错误码</th>
   <th>描述</th>
  </tr>
 </thead>
 <tbody>
  <tr>
   <td>ACLNN_ERR_PARAM_NULLPTR</td>
   <td>161001</td>
   <td>输入和输出的必选参数Tensor是空指针。</td>
  </tr>
  <tr>
   <td>ACLNN_ERR_PARAM_INVALID</td>
   <td>161002</td>
   <td>输入和输出的数据类型不在支持的范围内。</td>
  </tr>
  <tr>
   <td>ACLNN_ERR_INNER_TILING_ERROR</td>
   <td>561002</td>
   <td>1. 输入和输出的shape不在支持的范围内；<br>2. 参数的取值不在支持的范围。</td>
  </tr>
 </tbody>
</table>

## aclnnMoeDistributeDispatchV2

### 参数说明

<table style="undefined;table-layout: fixed; width: 1180px"> <colgroup>
 <col style="width: 250px">
 <col style="width: 130px">
 <col style="width: 800px">
 </colgroup>
 <thead>
  <tr>
   <th>参数名</th>
   <th>输入/输出</th>
   <th>描述</th>
  </tr>
 </thead>
 <tbody>
  <tr>
   <td>workspace</td>
   <td>输入</td>
   <td>在Device侧申请的workspace内存地址。</td>
  </tr>
  <tr>
   <td>workspaceSize</td>
   <td>输入</td>
   <td>在Device侧申请的workspace大小，由第一段接口<code>aclnnMoeDistributeDispatchV2GetWorkspaceSize</code>获取。</td>
  </tr>
  <tr>
   <td>executor</td>
   <td>输入</td>
   <td>op执行器，包含了算子计算流程。</td>
  </tr>
  <tr>
   <td>stream</td>
   <td>输入</td>
   <td>指定执行任务的Stream。</td>
  </tr>
 </tbody>
</table>

### 返回值

返回aclnnStatus状态码，具体参见aclnn返回码。

## 约束说明

1. `aclnnMoeDistributeDispatchV2`接口与CombineV2系列算子接口必须配套使用，具体参考调用示例。

2. 在不同产品型号、不同通信算法或不同版本中，`aclnnMoeDistributeDispatchV2`的Tensor输出`assistInfoForCombineOut`、`epRecvCountsOut`、`tpRecvCountsOut`、`expandScalesOut`中的元素值可能不同，使用时直接将上述Tensor传给CombineV2系列算子对应参数即可，模型其他业务逻辑不应对其存在依赖。

3. 调用接口过程中使用的`groupEp`、`epWorldSize`、`moeExpertNum`、`groupTp`、`tpWorldSize`、`expertShardType`、`sharedExpertNum`、`sharedExpertRankNum`、`globalBs`、`commAlg`参数及`HCCL_BUFFSIZE`取值所有卡需保持一致，网络中不同层中也需保持一致，且和CombineV2系列算子对应参数也保持一致。

4. <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：该场景下单卡包含双DIE（简称为“晶粒”或“裸片”），因此参数说明里的“本卡”均表示单DIE。

5. 参数说明里shape格式说明：
    - **A**：表示本卡可能接收的最大token数量，取值范围如下：
      - 对于共享专家，需满足 (A = Bs * epWorldSize * sharedExpertNum / sharedExpertRankNum)。
      - 对于MoE专家，当`globalBs`为0时，需满足 (A >= Bs * epWorldSize * min(localExpertNum, K))；当`globalBs`非0时，需满足 (A >= globalBs * min(localExpertNum, K))。
    - **H**：表示hidden size（隐藏层大小）：
      - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>：依commAlg取值，"fullmesh"支持(0, 7168]且为32的整数倍；"hierarchy"并且驱动版本≥25.0.RC1.1时支持(0, 10*1024]且为32的整数倍。
      - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：取值范围[1024, 8192]。
    - **Bs**：表示batch sequence size（本卡最终输出的token数量）：
      - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>：取值范围为 (0 < Bs ≤ 256)。
      - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：取值范围为 (0 < Bs ≤ 512)。
    - **K**：表示选取topK个专家，取值范围为 (0 < K ≤ 16) 且满足 (0 < K ≤ moeExpertNum)。
    - **serverNum**：表示服务器的节点数，取值仅支持2、4、8。
    - **localExpertNum**：表示本卡专家数量：
      - 对于共享专家卡，(localExpertNum = 1)。
      - 对于MoE专家卡，(localExpertNum = moeExpertNum / (epWorldSize - sharedExpertRankNum))；当(localExpertNum > 1)时，不支持TP域通信。

6. **HCCL_BUFFSIZE**：
   调用本接口前需检查`HCCL_BUFFSIZE`环境变量取值是否合理，该环境变量表示单个通信域占用内存大小，单位MB，不配置时默认为200MB：
   - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>：
     - commAlg为""或nullptr：依HCCL环境变量选择“fullmesh”或“hierarchy”公式。
     - commAlg为"fullmesh"：要求 (≥ 2 * (Bs * epWorldSize * min(localExpertNum, K) * H * sizeof(uint16) + 2MB))。
     - commAlg为"hierarchy"：要求 (≥ moeExpertNum * Bs * (H * sizeof(dtypeX) + 4 * ((K + 7) / 8 * 8) * sizeof(uint32)) + 4MB + 100MB)，不要求 (moeExpertNum / (epWorldSize - sharedExpertRankNum) ≤ 24)。
   - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：要求 (≥ 2) 且满足 (≥ 2 * (localExpertNum * maxBs * epWorldSize * Align512(Align32(2 * H) + 64) + (K + sharedExpertNum) * maxBs * Align512(2 * H)))（`localExpertNum`需使用MoE专家卡的本卡专家数；`Align512(x) = ((x + 512 - 1) / 512) * 512`；`Align32(x) = ((x + 32 - 1) / 32) * 32`）。

7. **HCCL_INTRA_PCIE_ENABLE和HCCL_INTRA_ROCE_ENABLE**：
   <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>：该环境变量不再推荐使用，建议通过`commAlg`配置为"hierarchy"。

8. 本文公式中的“/”表示整除。

9. 通信域使用约束：
   - 一个模型中的CombineV2系列算子和`aclnnMoeDistributeDispatchV2`仅支持相同EP通信域，且该通信域中不允许有其他算子。
   - 一个模型中的CombineV2系列算子和`aclnnMoeDistributeDispatchV2`仅支持相同TP通信域或都不支持TP通信域；有TP通信域时，该通信域中不允许有其他算子。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考编译与运行样例。本示例代码仅支持Atlas A3。

- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>
    ```Cpp
    #include <thread>
    #include <iostream>
    #include <string>
    #include <vector>
    #include "acl/acl.h"
    #include "hccl/hccl.h"
    #include "aclnnop/aclnn_moe_distribute_dispatch_v2.h"
    #include "aclnnop/aclnn_moe_distribute_combine_v2.h"
    #include <unistd.h>

    #define CHECK_RET(cond, return_expr) \
        do {                             \
            if (!(cond)) {               \
                return_expr;             \
            }                            \
        } while (0)

    #define LOG_PRINT(message, ...)         \
        do {                                \
            printf(message, ##__VA_ARGS__); \
        } while(0)

    struct Args {
        uint32_t rankId;
        uint32_t epRankId;
        uint32_t tpRankId;
        HcclComm hcclEpComm;
        HcclComm hcclTpComm;
        aclrtStream dispatchStream;
        aclrtStream combineStream;
        aclrtContext context;
    };

    constexpr uint32_t EP_WORLD_SIZE = 8;
    constexpr uint32_t TP_WORLD_SIZE = 2;
    constexpr uint32_t DEV_NUM = EP_WORLD_SIZE * TP_WORLD_SIZE;

    int64_t GetShapeSize(const std::vector<int64_t> &shape)
    {
        int64_t shape_size = 1;
        for (auto i : shape) {
            shape_size *= i;
        }
        return shape_size;
    }

    template<typename T>
    int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
        aclDataType dataType, aclTensor **tensor)
    {
        auto size = GetShapeSize(shape) * sizeof(T);
        auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc failed. ret: %d\n", ret); return ret);
        ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMemcpy failed. ret: %d\n", ret); return ret);
        std::vector<int64_t> strides(shape.size(), 1);
        for (int64_t i = shape.size() - 2; i >= 0; i--) {
            strides[i] = shape[i +1] * strides[i + 1];
        }
        *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
            shape.data(), shape.size(), *deviceAddr);
        return 0;
    }

    int LaunchOneProcessDispatchAndCombine(Args &args)
    {
        int ret = aclrtSetCurrentContext(args.context);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetCurrentContext failed, ret %d\n", ret); return ret);

        char hcomEpName[128] = {0};
        ret = HcclGetCommName(args.hcclEpComm, hcomEpName);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclGetEpCommName failed, ret %d\n", ret); return -1);
        char hcomTpName[128] = {0};
        ret = HcclGetCommName(args.hcclTpComm, hcomTpName);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclGetTpCommName failed, ret %d\n", ret); return -1);
        LOG_PRINT("[INFO] rank = %d, hcomEpName = %s, hcomTpName = %s, dispatchStream = %p, combineStream = %p, \
                  context = %p\n", args.rankId, hcomEpName, hcomTpName, args.dispatchStream, args.combineStream,                 \
                  args.context);

        int64_t Bs = 8;
        int64_t H = 7168;
        int64_t K = 3;
        int64_t expertShardType = 0;
        int64_t sharedExpertNum = 1;
        int64_t sharedExpertRankNum = 1;
        int64_t moeExpertNum = 7;
        int64_t quantMode = 0;
        int64_t globalBs = Bs * EP_WORLD_SIZE;
        int64_t expertTokenNumsType = 1;
        int64_t outDtype = 0;
        int64_t commQuantMode = 0;
        int64_t groupList_type = 1;
        int64_t localExpertNum;
        int64_t A;
        if (args.epRankId < sharedExpertRankNum) {
            localExpertNum = 1;
            A = globalBs / sharedExpertRankNum;
        } else {
            localExpertNum = moeExpertNum / (EP_WORLD_SIZE - sharedExpertRankNum);
            A = globalBs * (localExpertNum < K ? localExpertNum : K);
        }

        void *xDeviceAddr = nullptr;
        void *expertIdsDeviceAddr = nullptr;
        void *scalesDeviceAddr = nullptr;
        void *expertScalesDeviceAddr = nullptr;
        void *expandXDeviceAddr = nullptr;
        void *dynamicScalesDeviceAddr = nullptr;
        void *expandIdxDeviceAddr = nullptr;
        void *expertTokenNumsDeviceAddr = nullptr;
        void *epRecvCountsDeviceAddr = nullptr;
        void *tpRecvCountsDeviceAddr = nullptr;
        void *expandScalesDeviceAddr = nullptr;

        aclTensor *x = nullptr;
        aclTensor *expertIds = nullptr;
        aclTensor *scales = nullptr;
        aclTensor *expertScales = nullptr;
        aclTensor *expandX = nullptr;
        aclTensor *dynamicScales = nullptr;
        aclTensor *expandIdx = nullptr;
        aclTensor *expertTokenNums = nullptr;
        aclTensor *epRecvCounts = nullptr;
        aclTensor *tpRecvCounts = nullptr;
        aclTensor *expandScales = nullptr;

        std::vector<int64_t> xShape{Bs, H};
        std::vector<int64_t> expertIdsShape{Bs, K};
        std::vector<int64_t> scalesShape{moeExpertNum + 1, H};
        std::vector<int64_t> expertScalesShape{Bs, K};
        std::vector<int64_t> expandXShape{TP_WORLD_SIZE * A, H};
        std::vector<int64_t> dynamicScalesShape{TP_WORLD_SIZE * A};
        std::vector<int64_t> expandIdxShape{A * 128};
        std::vector<int64_t> expertTokenNumsShape{localExpertNum};
        std::vector<int64_t> epRecvCountsShape{TP_WORLD_SIZE * localExpertNum * EP_WORLD_SIZE};
        std::vector<int64_t> tpRecvCountsShape{TP_WORLD_SIZE};
        std::vector<int64_t> expandScalesShape{A};

        int64_t xShapeSize = GetShapeSize(xShape);
        int64_t expertIdsShapeSize = GetShapeSize(expertIdsShape);
        int64_t scalesShapeSize = GetShapeSize(scalesShape);
        int64_t expertScalesShapeSize = GetShapeSize(expertScalesShape);
        int64_t expandXShapeSize = GetShapeSize(expandXShape);
        int64_t dynamicScalesShapeSize = GetShapeSize(dynamicScalesShape);
        int64_t expandIdxShapeSize = GetShapeSize(expandIdxShape);
        int64_t expertTokenNumsShapeSize = GetShapeSize(expertTokenNumsShape);
        int64_t epRecvCountsShapeSize = GetShapeSize(epRecvCountsShape);
        int64_t tpRecvCountsShapeSize = GetShapeSize(tpRecvCountsShape);
        int64_t expandScalesShapeSize = GetShapeSize(expandScalesShape);

        std::vector<int16_t> xHostData(xShapeSize, 1);
        std::vector<int32_t> expertIdsHostData;
        for (int32_t token_id = 0; token_id < expertIdsShape[0]; token_id++) {
            for (int32_t k_id = 0; k_id < expertIdsShape[1]; k_id++) {
                expertIdsHostData.push_back(k_id);
            }
        }

        std::vector<float> scalesHostData(scalesShapeSize, 0.1);
        std::vector<float> expertScalesHostData(expertScalesShapeSize, 0.1);
        std::vector<int16_t> expandXHostData(expandXShapeSize, 0);
        std::vector<float> dynamicScalesHostData(dynamicScalesShapeSize, 0);
        std::vector<int32_t> expandIdxHostData(expandIdxShapeSize, 0);
        std::vector<int64_t> expertTokenNumsHostData(expertTokenNumsShapeSize, 0);
        std::vector<int32_t> epRecvCountsHostData(epRecvCountsShapeSize, 0);
        std::vector<int32_t> tpRecvCountsHostData(tpRecvCountsShapeSize, 0);
        std::vector<float> expandScalesHostData(expandScalesShapeSize, 0);

        ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_BF16, &x);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(expertIdsHostData, expertIdsShape, &expertIdsDeviceAddr, aclDataType::ACL_INT32, &expertIds);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(scalesHostData, scalesShape, &scalesDeviceAddr, aclDataType::ACL_FLOAT, &scales);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(expertScalesHostData, expertScalesShape, &expertScalesDeviceAddr, aclDataType::ACL_FLOAT, &expertScales);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(expandXHostData, expandXShape, &expandXDeviceAddr, (quantMode > 0) ? aclDataType::ACL_INT8 : aclDataType::ACL_BF16, &expandX);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(dynamicScalesHostData, dynamicScalesShape, &dynamicScalesDeviceAddr, aclDataType::ACL_FLOAT, &dynamicScales);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
         ret = CreateAclTensor(expandIdxHostData, expandIdxShape, &expandIdxDeviceAddr, aclDataType::ACL_INT32, &expandIdx);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(expertTokenNumsHostData, expertTokenNumsShape, &expertTokenNumsDeviceAddr, aclDataType::ACL_INT64, &expertTokenNums);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(epRecvCountsHostData, epRecvCountsShape, &epRecvCountsDeviceAddr, aclDataType::ACL_INT32, &epRecvCounts);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(tpRecvCountsHostData, tpRecvCountsShape, &tpRecvCountsDeviceAddr, aclDataType::ACL_INT32, &tpRecvCounts);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(expandScalesHostData, expandScalesShape, &expandScalesDeviceAddr, aclDataType::ACL_FLOAT, &expandScales);
        CHECK_RET(ret == ACL_SUCCESS, return ret);

        uint64_t dispatchWorkspaceSize = 0;
        aclOpExecutor *dispatchExecutor = nullptr;
        void *dispatchWorkspaceAddr = nullptr;

        uint64_t combineWorkspaceSize = 0;
        aclOpExecutor *combineExecutor = nullptr;
        void *combineWorkspaceAddr = nullptr;

        /**************************************** 调用dispatch ********************************************/

        ret = aclnnMoeDistributeDispatchV2GetWorkspaceSize(x, expertIds, (quantMode > 0 ? scales : nullptr), nullptr,
                expertScales, hcomEpName, EP_WORLD_SIZE, args.epRankId, moeExpertNum, hcomTpName, TP_WORLD_SIZE,
                args.tpRankId, expertShardType, sharedExpertNum,sharedExpertRankNum, quantMode, globalBs,
                expertTokenNumsType, nullptr, expandX, dynamicScales, expandIdx, expertTokenNums, epRecvCounts,
                tpRecvCounts, expandScales, &dispatchWorkspaceSize, &dispatchExecutor);

        CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("[ERROR] aclnnMoeDistributeDispatchV2GetWorkspaceSize failed. ret = %d \n", ret); return ret);

        if (dispatchWorkspaceSize > 0) {
            ret = aclrtMalloc(&dispatchWorkspaceAddr, dispatchWorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc workspace failed. ret = %d \n", ret); return ret);
        }
        // 调用第二阶段接口
        ret = aclnnMoeDistributeDispatchV2(dispatchWorkspaceAddr, dispatchWorkspaceSize,
                                           dispatchExecutor, args.dispatchStream);
        ret = aclrtSynchronizeStreamWithTimeout(args.dispatchStream, 10000);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclnnMoeDistributeDispatchV2 failed. ret = %d \n", ret);  \
            return ret);

        /**************************************** 调用combine ********************************************/
        // 调用第一阶段接口
        ret = aclnnMoeDistributeCombineV2GetWorkspaceSize(expandX, expertIds,
                                                         expandIdx, epRecvCounts,
                                                         expertScales, tpRecvCounts,
                                                         nullptr, nullptr, nullptr,
                                                         nullptr, nullptr, nullptr,
                                                         hcomEpName, EP_WORLD_SIZE, args.epRankId, moeExpertNum,
                                                         hcomTpName, TP_WORLD_SIZE, args.tpRankId, expertShardType,
                                                         sharedExpertNum, sharedExpertRankNum, globalBs, outDtype,
                                                         commQuantMode, groupList_type, nullptr, x,
                                                         &combineWorkspaceSize, &combineExecutor);
        CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("[ERROR] aclnnMoeDistributeCombineV2GetWorkspaceSize failed. ret = %d \n", ret); return ret);
        // 根据第一阶段接口计算出的workspaceSize申请device内存
        if (combineWorkspaceSize > 0) {
            ret = aclrtMalloc(&combineWorkspaceAddr, combineWorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc workspace failed. ret = %d \n", ret); return ret);
        }

        // 调用第二阶段接口
        ret = aclnnMoeDistributeCombineV2(combineWorkspaceAddr, combineWorkspaceSize, combineExecutor, args.combineStream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclnnMoeDistributeCombineV2 failed. ret = %d \n", ret);
            return ret);
        // （固定写法）同步等待任务执行结束
        ret = aclrtSynchronizeStreamWithTimeout(args.combineStream, 10000);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSynchronizeStreamWithTimeout failed. ret = %d \n", ret);
            return ret);
        LOG_PRINT("[INFO] device_%d aclnnMoeDistributeDispatchV2 and aclnnMoeDistributeCombineV2                      \
                   execute successfully.\n", args.rankId);

        // 释放device资源
        if (dispatchWorkspaceSize > 0) {
            aclrtFree(dispatchWorkspaceAddr);
        }
        if (combineWorkspaceSize > 0) {
            aclrtFree(combineWorkspaceAddr);
        }
        if (x != nullptr) {
            aclDestroyTensor(x);
        }
        if (expertIds != nullptr) {
            aclDestroyTensor(expertIds);
        }
        if (scales != nullptr) {
            aclDestroyTensor(scales);
        }
        if (expertScales != nullptr) {
            aclDestroyTensor(expertScales);
        }
        if (expandX != nullptr) {
            aclDestroyTensor(expandX);
        }
        if (dynamicScales != nullptr) {
            aclDestroyTensor(dynamicScales);
        }
        if (expandIdx != nullptr) {
            aclDestroyTensor(expandIdx);
        }
        if (expertTokenNums != nullptr) {
            aclDestroyTensor(expertTokenNums);
        }
        if (epRecvCounts != nullptr) {
            aclDestroyTensor(epRecvCounts);
        }
        if (tpRecvCounts != nullptr) {
            aclDestroyTensor(tpRecvCounts);
        }
        if (expandScales != nullptr) {
            aclDestroyTensor(expandScales);
        }
        if (xDeviceAddr != nullptr) {
            aclrtFree(xDeviceAddr);
        }
        if (expertIdsDeviceAddr != nullptr) {
            aclrtFree(expertIdsDeviceAddr);
        }
        if (scalesDeviceAddr != nullptr) {
            aclrtFree(scalesDeviceAddr);
        }
        if (expertScalesDeviceAddr != nullptr) {
            aclrtFree(expertScalesDeviceAddr);
        }
        if (expandXDeviceAddr != nullptr) {
            aclrtFree(expandXDeviceAddr);
        }
        if (dynamicScalesDeviceAddr != nullptr) {
            aclrtFree(dynamicScalesDeviceAddr);
        }
        if (expandIdxDeviceAddr != nullptr) {
            aclrtFree(expandIdxDeviceAddr);
        }
        if (expertTokenNumsDeviceAddr != nullptr) {
            aclrtFree(expertTokenNumsDeviceAddr);
        }
        if (epRecvCountsDeviceAddr != nullptr) {
            aclrtFree(epRecvCountsDeviceAddr);
        }
        if (expandScalesDeviceAddr != nullptr) {
            aclrtFree(expandScalesDeviceAddr);
        }
        if (tpRecvCountsDeviceAddr != nullptr) {
            aclrtFree(tpRecvCountsDeviceAddr);
        }

        HcclCommDestroy(args.hcclEpComm);
        HcclCommDestroy(args.hcclTpComm);
        aclrtDestroyStream(args.dispatchStream);
        aclrtDestroyStream(args.combineStream);
        aclrtDestroyContext(args.context);
        aclrtResetDevice(args.rankId);

        return 0;
    }

    int main(int argc, char *argv[])
    {
        int ret = aclInit(nullptr);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtInit failed, ret = %d\n", ret); return ret);

        aclrtStream dispatchStream[DEV_NUM];
        aclrtStream combineStream[DEV_NUM];
        aclrtContext context[DEV_NUM];
        for (uint32_t rankId = 0; rankId < DEV_NUM; rankId++) {
            ret = aclrtSetDevice(rankId);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetDevice failed, ret = %d\n", ret); return ret);
            ret = aclrtCreateContext(&context[rankId], rankId);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateContext failed, ret = %d\n", ret); return ret);
            ret = aclrtCreateStream(&dispatchStream[rankId]);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateStream failed, ret = %d\n", ret); return ret);
            ret = aclrtCreateStream(&combineStream[rankId]);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateStream failed, ret = %d\n", ret); return ret);
        }

        int32_t devicesEp[TP_WORLD_SIZE][EP_WORLD_SIZE];
        for (int32_t tpId = 0; tpId < TP_WORLD_SIZE; tpId++) {
            for (int32_t epId = 0; epId < EP_WORLD_SIZE; epId++) {
                devicesEp[tpId][epId] = epId * TP_WORLD_SIZE + tpId;
            }
        }

        HcclComm commsEp[TP_WORLD_SIZE][EP_WORLD_SIZE];
        for (int32_t tpId = 0; tpId < TP_WORLD_SIZE; tpId++) {
            ret = HcclCommInitAll(EP_WORLD_SIZE, devicesEp[tpId], commsEp[tpId]);
            CHECK_RET(ret == ACL_SUCCESS,
                      LOG_PRINT("[ERROR] HcclCommInitAll ep %d failed, ret %d\n", tpId, ret); return ret);
        }

        int32_t devicesTp[EP_WORLD_SIZE][TP_WORLD_SIZE];
        for (int32_t epId = 0; epId < EP_WORLD_SIZE; epId++) {
            for (int32_t tpId = 0; tpId < TP_WORLD_SIZE; tpId++) {
                devicesTp[epId][tpId] = epId * TP_WORLD_SIZE + tpId;
            }
        }

        HcclComm commsTp[EP_WORLD_SIZE][TP_WORLD_SIZE];
        for (int32_t epId = 0; epId < EP_WORLD_SIZE; epId++) {
            ret = HcclCommInitAll(TP_WORLD_SIZE, devicesTp[epId], commsTp[epId]);
            CHECK_RET(ret == ACL_SUCCESS,
                      LOG_PRINT("[ERROR] HcclCommInitAll tp %d failed, ret %d\n", epId, ret); return ret);
        }

        Args args[DEV_NUM];
        std::vector<std::unique_ptr<std::thread>> threads(DEV_NUM);
        for (uint32_t rankId = 0; rankId < DEV_NUM; rankId++) {
            uint32_t epRankId = rankId / TP_WORLD_SIZE;
            uint32_t tpRankId = rankId % TP_WORLD_SIZE;

            args[rankId].rankId = rankId;
            args[rankId].epRankId = epRankId;
            args[rankId].tpRankId = tpRankId;
            args[rankId].hcclEpComm = commsEp[tpRankId][epRankId];
            args[rankId].hcclTpComm = commsTp[epRankId][tpRankId];
            args[rankId].dispatchStream = dispatchStream[rankId];
            args[rankId].combineStream = combineStream[rankId];
            args[rankId].context = context[rankId];
            threads[rankId].reset(new(std::nothrow) std::thread(&LaunchOneProcessDispatchAndCombine, std::ref(args[rankId])));
        }

        for(uint32_t rankId = 0; rankId < DEV_NUM; rankId++) {
            threads[rankId]->join();
        }

        aclFinalize();
        LOG_PRINT("[INFO] aclFinalize success\n");

        return 0;
    }
    ```