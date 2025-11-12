# aclnnMoeDistributeCombineV3

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term> |    √     |

## 功能说明

- 算子功能：当存在TP域通信时，先进行ReduceScatterV通信，再进行AlltoAllV通信，最后将接收的数据整合（乘权重再相加）；当不存在TP域通信时，进行AlltoAllV通信，最后将接收的数据整合（乘权重再相加）。
- 计算公式：
$$
rsOut = ReduceScatterV(expandX)\\
ataOut = AllToAllV(rsOut)\\
xOut = Sum(expertScales * ataOut + expertScales * sharedExpertX)
$$

> 注意：该接口必须与`aclnnMoeDistributeDispatchV3`配套使用，相当于按`aclnnMoeDistributeDispatchV3`接口收集数据的路径原路返还。

相较于`aclnnMoeDistributeCombineV2`接口，该接口变更如下：
- 新增支持动态缩容场景：支持在创建通信域后，剔除故障卡，算子可正常执行（无需重新编译），通过传入`elasticInfoOptional`参数使能该特性。
- 新增支持特殊专家场景：
  - **zeroExpertNum≠0**：通过传入大于0的`zeroExpertNum`参数使能。
  $$Moe(oriXOptional) = 0$$
  - **copyExpertNum≠0**：通过传入大于0的`copyExpertNum`参数使能，且需传入有效的`oriXOptional`参数。
  $$Moe(oriXOptional) = oriXOptional$$
  - **constExpertNum≠0**：通过传入大于0的`constExpertNum`参数使能，且需传入有效的`oriXOptional`、`constExpertAlpha1Optional`、`constExpertAlpha2Optional`、`constExpertVOptional`参数。
  $$Moe(oriXOptional) = constExpertAlpha1Optional * oriXOptional + constExpertAlpha2Optional * constExpertVOptional$$


## 函数原型

每个算子分为两段式接口，必须先调用 “aclnnMoeDistributeCombineV3GetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnMoeDistributeCombineV3”接口执行计算。

```cpp
aclnnStatus aclnnMoeDistributeCombineV3GetWorkspaceSize(
    const aclTensor* expandX,
    const aclTensor* expertIds,
    const aclTensor* assistInfoForCombine,
    const aclTensor* epSendCounts,
    const aclTensor* expertScales,
    const aclTensor* tpSendCountsOptional,
    const aclTensor* xActiveMaskOptional,
    const aclTensor* activationScaleOptional,
    const aclTensor* weightScaleOptional,
    const aclTensor* groupListOptional,
    const aclTensor* expandScalesOptional,
    const aclTensor* sharedExpertXOptional,
    const aclTensor* elasticInfoOptional,
    const aclTensor* oriXOptional,
    const aclTensor* constExpertAlpha1Optional,
    const aclTensor* constExpertAlpha2Optional,
    const aclTensor* constExpertVOptional,
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
    int64_t          globalBs,
    int64_t          outDtype,
    int64_t          commQuantMode,
    int64_t          groupListType,
    const char*      commAlg,
    int64_t          zeroExpertNum,
    int64_t          copyExpertNum,
    int64_t          constExpertNum,
    aclTensor*       xOut,
    uint64_t*        workspaceSize,
    aclOpExecutor**  executor)
```

```cpp
aclnnStatus aclnnMoeDistributeCombineV3(
    void*           workspace,
    uint64_t        workspaceSize,
    aclOpExecutor*  executor,
    aclrtStream     stream)
```

## aclnnMoeDistributeCombineV3GetWorkspaceSize

### 参数说明

<table style="undefined;table-layout: fixed; width: 1576px">
<colgroup>
 <col style="width: 170px">
 <col style="width: 170px">
 <col style="width: 800px">
 <col style="width: 400px">
 <col style="width: 200px">
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
   <td>expandX</td>
   <td>输入</td>
   <td>根据expertIds进行扩展过的token特征，Device侧的aclTensor，要求为一个2D的Tensor，shape为<code>(max(tpWorldSize, 1) * A , H)</code>。<br> <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>：不支持共享专家场景。 </td>
   <td>FLOAT16、BFLOAT16</td>
   <td>ND（支持非连续Tensor）</td>
  </tr>
  <tr>
   <td>expertIds</td>
   <td>输入</td>
   <td>每个token的topK个专家索引，Device侧的aclTensor，要求为一个2D的Tensor，shape为<code>(Bs, K)</code>。</td>
   <td>INT32</td>
   <td>ND（支持非连续Tensor）</td>
  </tr>
  <tr>
   <td>assistInfoForCombine</td>
   <td>输入</td>
   <td>对应<code>aclnnMoeDistributeDispatchV3</code>的<code>assistInfoForCombineOut</code>输出，Device侧的aclTensor，要求是一个1D的Tensor，要求shape为<code>(A * 128, )</code>，</td>
   <td>INT32</td>
   <td>ND（支持非连续Tensor）</td>
  </tr>
  <tr>
   <td>epSendCounts</td>
   <td>输入</td>
   <td>对应<code>aclnnMoeDistributeDispatchV3</code>的<code>epRecvCounts</code>输出，Device侧的aclTensor,要求是一个1D的Tensor。<br><term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>：要求shape为<code>(moeExpertNum + 2 * globalBs * K * serverNum, )</code>，前<code>moeExpertNum</code>个数表示从EP通信域各卡接收的token数，<code>2 * globalBs * K * serverNum</code>存储了机间机内做通信前combine可以提前做reduce的token个数和token在通信区中的偏移，globalBs传入0时在此处应当按照<code>Bs * epWorldSize</code>计算。<br><term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：要求shape为 <code>(epWorldSize * max(tpWorldSize, 1) * localExpertNum, )</code>。</td>
   <td>INT32</td>
   <td>ND（支持非连续Tensor）</td>
  </tr>
  <tr>
   <td>expertScales</td>
   <td>输入</td>
   <td>每个token的topK个专家的权重，Device侧的aclTensor，要求是一个2D的Tensor，shape为 <code>(Bs, K)</code>。</td>
   <td>FLOAT32</td>
   <td>ND（支持非连续Tensor）</td>
  </tr>
  <tr>
   <td>tpSendCountsOptional</td>
   <td>输入</td>
   <td>对应<code>aclnnMoeDistributeDispatchV3</code>的<code>tpRecvCounts</code>输出，有TP域通信时传参，否则传空指针：<br><term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>：当前不支持TP域通信。<br><term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：当有TP域通信时，要求是一个1D的Tensor，shape为 <code>(tpWorldSize, )</code>。
   <td>INT32</td>
   <td>ND（支持非连续Tensor）</td>
  </tr>
  <tr>
   <td>xActiveMaskOptional</td>
   <td>输入</td>
   <td>标识token是否参与通信,，Device侧的aclTensor：<br><term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>：要求为1D Tensor，shape为\(BS, \)；true需排在false前（例：{true, false, true}非法）。<br><term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：要求是一个1D或者2D Tensor。当输入为1D时，shape为<code>(Bs, )</code>; 当输入为2D时，shape为<code>(Bs, K)</code>；可选择传入有效数据或传入空指针。当输入为1D时，参数为true表示对应的token参与通信，true必须排到false之前，例：{true, false, true} 为非法输入；当输入为2D时，参数为true表示当前token对应的expert_ids参与通信。若当前token对应的K个BOOL值全为false，表示当前token不会参与通信。默认所有token都会参与通信。当每张卡的Bs数量不一致时，所有token必须全部有效。</td>
   <td>BOOL</td>
   <td>ND（支持非连续Tensor）</td>
  </tr>
  <tr>
   <td>activationScaleOptional</td>
   <td>输入</td>
   <td>预留参数，当前版本不支持，传空指针。</td>
   <td>-</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>weightScaleOptional</td>
   <td>输入</td>
   <td>预留参数，当前版本不支持，传空指针。</td>
   <td>-</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>groupListOptional</td>
   <td>输入</td>
   <td>预留参数，当前版本不支持，传空指针。</td>
   <td>-</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>expandScalesOptional</td>
   <td>输入</td>
   <td>对应<code>aclnnMoeDistributeDispatchV3</code>的<code>expandScales</code>输出：<br><term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>：要求是一个1D的Tensor，shape为 <code>(A, )</code>。<br><term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：预留参数，当前版本不支持，传空指针即可。</td>
   <td>FLOAT32</td>
   <td>ND（支持非连续Tensor）</td>
  </tr>
  <tr>
   <td>sharedExpertXOptional</td>
   <td>输入</td>
   <td>表示共享专家计算后的token，Device侧的aclTensor：<br><term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>：预留参数，当前版本不支持，传空指针即可。<br><term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：要求是一个2D或3D的Tensor，当Tensor为2D时，shape为<code>(Bs, H)</code>；当Tensor为3D时，前两位的乘积需等于Bs，第三维需等于H。数据类型需跟<code>expandX</code>保持一致。可传/可不传，传入时，<code>sharedExpertRankNum</code>需为0。</td>
   <td>FLOAT16、BFLOAT16</td>
   <td>ND（支持非连续Tensor）</td>
  </tr>
  <tr>
   <td>elasticInfoOptional</td>
   <td>输入</td>
   <td>EP通信域动态缩容信息：<br><term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>：当前版本不支持，传空指针即可。<br><term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：可选择传入有效数据或填空指针，传入空指针时表示不使能动态缩容功能；当传入有效数据时，要求是一个1D的Tensor，shape为 <code>(4 + 2 * epWorldSize, )</code>。Tensor中的前四个数字分别表示（是否缩容，缩容后实际rank数，缩容后共享专家使用的rank数，缩容后moe专家的个数），后2 * epWorldSize表示2个rank映射表，缩容后本卡中因部分rank异常而从EP通信域中剔除，第一个Table的映射关系为<code>Table1[epRankId]=localEpRankId或-1</code>，localEpRankId表示新EP通信域中的rank Index，-1表示epRankId这张卡从通信域中被剔除，第二个Table映射关系为<code>Table2[localEpRankId] = epRankId</code>。</td>
   <td>INT32</td>
   <td>ND（支持非连续Tensor）</td>
  </tr>
  <tr>
   <td>oriXOptional</td>
   <td>输入</td>
   <td>Device侧的aclTensor，表示未经过FFN（Feed-Forward Neural network）的token数据，在使能copyExpert或使能constExpert的场景下需要本输入数据。：<br><term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>：预留参数，当前版本不支持，传空指针即可。<br><term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：可选择传入有效数据或填空指针，当<code>copyExpertNum</code>不为0或<code>constExpertNum</code>不为0时必须传入有效输入；当传入有效数据时，要求是一个2D的Tensor，shape为 <code>(Bs, H)</code>，数据类型需跟expandX保持一致。</td>
   <td>FLOAT16、BFLOAT16</td>
   <td>ND（支持非连续Tensor）</td>
  </tr>
  <tr>
   <td>constExpertAlpha1Optional</td>
   <td>输入</td>
   <td>Device侧的aclTensor，在使能constExpert的场景下需要输入的计算系数：<br><term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>：预留参数，当前版本不支持，传空指针即可。<br><term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：可选择传入有效数据或填空指针，当constExpertNum不为0时必须传入有效输入；当传入有效数据时，要求是一个1D的Tensor，shape为<code>(constExpertNum, )</code>，数据类型需跟expandX保持一致。</td>
   <td>FLOAT16、BFLOAT16</td>
   <td>ND（支持非连续Tensor）</td>
  </tr>
  <tr>
   <td>constExpertAlpha2Optional</td>
   <td>输入</td>
   <td>Device侧的aclTensor，在使能constExpert的场景下需要输入的计算系数：<br><term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>：预留参数，当前版本不支持，传空指针即可。<br><term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：可选择传入有效数据或填空指针，当constExpertNum不为0时必须传入有效输入；当传入有效数据时，要求是一个1D的Tensor，shape为<code>(constExpertNum, )</code>，数据类型需跟expandX保持一致。</td>
   <td>FLOAT16、BFLOAT16</td>
   <td>ND（支持非连续Tensor）</td>
  </tr>
  <tr>
   <td>constExpertVOptional</td>
   <td>输入</td>
   <td>Device侧的aclTensor，在使能constExpert的场景下需要输入的计算系数：<br><term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>：预留参数，当前版本不支持，传空指针即可。<br><term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：可选择传入有效数据或填空指针，当constExpertNum不为0时必须传入有效输入；当传入有效数据时，要求是一个2D的Tensor，shape为 <code>(constExpertNum, H)</code>，数据类型需跟expandX保持一致。</td>
   <td>FLOAT16、BFLOAT16</td>
   <td>ND（支持非连续Tensor）</td>
  </tr>
  <tr>
   <td>groupEp</td>
   <td>输入</td>
   <td>专家并行的EP通信域名称，字符串长度范围为[1, 128)，不能和groupTp相同</td>
   <td>STRING</td>
   <td>-</td>
  </tr>
  <tr>
   <td>epWorldSize</td>
   <td>输入</td>
   <td>EP通信域大小：<br><term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>：依commAlg取值，"fullmesh"支持16、32、64、128、256；"hierarchy"支持16、32、64。<br><term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：取值支持[2, 768]。</td>
   <td>INT64</td>
   <td>-</td>
  </tr>
  <tr>
   <td>epRankId</td>
   <td>输入</td>
   <td>EP域本卡ID，取值范围[0, epWorldSize)，同一个EP通信域中各卡的epRankId不重复。</td>
   <td>INT64</td>
   <td>-</td>
  </tr>
  <tr>
   <td>moeExpertNum</td>
   <td>输入</td>
   <td>MoE专家数量，满足 <code>moeExpertNum % (epWorldSize - sharedExpertRankNum) = 0</code>：<br><term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>: 取值范围(0, 512], 还需满足<code>moeExpertNum / (epWorldSize - sharedExpertRankNum) <= 24</code>。<br><term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：取值范围(0, 1024]。
   <td>INT64</td>
   <td>-</td>
  </tr>
  <tr>
   <td>groupTp</td>
   <td>输入</td>
   <td>TP通信域名称（数据并行）：<br><term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>：当前版本不支持，传空字符即可。<br><term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：字符串长度范围为[1, 128)，不能和groupEp相同。</td>
   <td>STRING</td>
   <td>-</td>
  </tr>
  <tr>
   <td>tpWorldSize</td>
   <td>输入</td>
   <td>TP通信域大小：<br><term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>：当前版本不支持，传0即可。<br>><term>Atlas A3 训练系列产品/Atlas A3 推理系列产品：取值范围[0, 2]，0和1表示无TP域通信，有TP域通信时仅支持2。</term></td>
   <td>INT64</td>
   <td>-</td>
  </tr>
  <tr>
   <td>tpRankId</td>
   <td>输入</td>
   <td>TP域本卡ID：<br><term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>：当前版本不支持，传0即可。<br><term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：取值范围[0, 1]，同一个TP通信域中各卡的tpRankId不重复。无TP域通信时，传0即可。</td>
   <td>INT64</td>
   <td>-</td>
  </tr>
  <tr>
   <td>expertShardType</td>
   <td>输入</td>
   <td>共享专家卡分布类型:<br><term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>：当前版本不支持，传0即可。<br><term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：当前仅支持传0，表示共享专家卡排在MoE专家卡前面。</td>
   <td>INT64</td>
   <td>-</td>
  </tr>
  <tr>
   <td>sharedExpertNum</td>
   <td>输入</td>
   <td>共享专家卡分布类型：<br><term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>：当前版本不支持，传0即可。<br><term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：当前取值范围[0, 4]。</td>
   <td>INT64</td>
   <td>-</td>
  </tr>
  <tr>
   <td>sharedExpertRankNum</td>
   <td>输入</td>
   <td>共享专家卡数量：<br><term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>：当前版本不支持，传0即可.<br><term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：当前取值范围[0, epWorldSize)，为0时需满足sharedExpertNum为0或1，不为0时需满足<code>sharedExpertRankNum % sharedExpertNum = 0</code>。
   </td>
   <td>INT64</td>
   <td>-</td>
  </tr>
  <tr>
   <td>globalBs</td>
   <td>输入</td>
   <td>EP域全局batch size：<br>当每个rank的Bs数一致时，<code>globalBs = Bs * epWorldSize </code>或 0 <br>当每个rank的Bs数不一致时，<code>globalBs = maxBs * epWorldSize</code>，其中maxBs表示单卡Bs最大值。
   </td>
   <td>INT64</td>
   <td>-</td>
  </tr>
  <tr>
   <td>outDtype</td>
   <td>输入</td>
   <td>预留参数，指定输出数据类型，当前版本不支持，传0。</td>
   <td>INT64</td>
   <td>-</td>
  </tr>
  <tr>
   <td>commQuantMode</td>
   <td>输入</td>
   <td>通信量化类型:取值范围0或者2，0表示通信时不进行量化，2表示通信时进行int8量化<br><term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>：2仅当commAlg配置为"hierarchy"或HCCL_INTRA_PCIE_ENABLE为1且HCCL_INTRA_ROCE_ENABLE为0且驱动版本不低于25.0.RC1.1时支持。
   <br><term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：int8量化当且仅当tpWorldSize < 2时可使能。</td>
   <td>INT64</td>
   <td>-</td>
  </tr>
  <tr>
   <td>groupListType</td>
   <td>输入</td>
   <td>预留参数，group List格式，当前版本不支持，传0。</td>
   <td>INT64</td>
   <td>-</td>
  </tr>
  <tr>
   <td>commAlg</td>
   <td>输入</td>
   <td>通信亲和内存布局算法：<ul><li><term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>：当前版本支持nullptr，""，"fullmesh"，"hierarchy"四种输入方式。推荐配置"hierarchy"并搭配25.0.RC1.1及以上版本驱动使用。<ul><li>nullptr和"": 仅在此场景下，HCCL_INTRA_PCIE_ENABLE和HCCL_INTRA_ROCE_ENABLE配置生效。当HCCL_INTRA_PCIE_ENABLE=1&& HCCL_INTRA_ROCE_ENABLE=0时，调用"hierarchy"算法，否则调用"fullmesh"算法。不推荐使用该方式;</li><li>"fullmesh": token数据直接通过RDMA方式发回目标卡;</li><li>"hierarchy": token数据经过机内、跨机两次发送，先在server内将同一个token数据汇总求和，再跨机发送，以减少跨机数据量。</ul></li><li><term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：当前版本不支持，传空指针即可。</td>
   <td>STRING</td>
   <td>-</td>
  </tr>
  <tr>
   <td>zeroExpertNum</td>
   <td>输入</td>
   <td>零专家数量：<br><term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>：当前版本不支持，传0即可。<br><term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：取值范围:[0, MAX_INT32)，MAX_INT32 = 2^31 - 1, 合法的零专家的ID的值是[<code>moeExpertNum<code>, <code>moeExpertNum + zeroExpertNum</code>)。
   <td>INT64</td>
   <td>-</td>
  </tr>
  <tr>
   <td>copyExpertNum</td>
   <td>输入</td>
   <td>copy专家数量：<br><term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>：当前版本不支持，传0即可。<br><term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：取值范围:[0, MAX_INT32)，MAX_INT32 = 2^31 - 1, 合法的拷贝专家的ID的值是[<code>moeExpertNum + zeroExpertNum</code>, </code>moeExpertNum + zeroExpertNum + copyExpertNum</code>)。</td>
   <td>INT64</td>
   <td>-</td>
  </tr>
  <tr>
   <td>constExpertNum</td>
   <td>输入</td>
   <td>常量专家数量：<br><term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>：当前版本不支持，传0即可；<br><term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：取值范围:[0, MAX_INT32)，MAX_INT32 = 2^31 - 1, 合法的常量专家的ID的值是[<code>moeExpertNum + zeroExpertNum + copyExpertNum<code>, <code>moeExpertNum + zeroExpertNum + copyExpertNum + constExpertNum<code>).
   <td>INT64</td>
   <td>-</td>
  </tr>
  <tr>
   <td>xOut</td>
   <td>输出</td>
   <td>处理后的token，2D Tensor，shape为 <code>(Bs, H)</code>，数据类型/格式与expandX一致。</td>
   <td>FLOAT16、BFLOAT16</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>workspaceSize</td>
   <td>输出</td>
   <td>返回Device侧需申请的workspace大小。</td>
   <td>UINT64</td>
   <td>-</td>
  </tr>
  <tr>
   <td>executor</td>
   <td>输出</td>
   <td>返回包含算子计算流程的op执行器。</td>
   <td>aclOpExecutor*</td>
   <td>-</td>
  </tr>
 </tbody>
</table>

### 返回值

第一段接口完成入参校验，出现以下场景时报错：

<table style="undefined;table-layout: fixed; width: 1576px">
<colgroup>
 <col style="width: 170px">
 <col style="width: 170px">
 <col style="width: 400px">
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
   <td>输入和输出的shape不在支持的范围内；<br>参数的取值不在支持的范围。</td>
  </tr>
 </tbody>
</table>

## aclnnMoeDistributeCombineV3

### 参数说明

<table style="undefined;table-layout: fixed; width: 1576px">
<colgroup>
 <col style="width: 170px">
 <col style="width: 170px">
 <col style="width: 800px">
 </colgroup>
 <thead>
  <tr>
   <th>参数名</th>
   <th>输入/输出</th>
   <th>描述</th>
   <th>数据类型</th>
  </tr>
 </thead>
 <tbody>
  <tr>
   <td>workspace</td>
   <td>输入</td>
   <td>在Device侧申请的workspace内存地址。</td>
   <td>void*</td>
  </tr>
  <tr>
   <td>workspaceSize</td>
   <td>输入</td>
   <td>在Device侧申请的workspace大小，由第一段接口<code>aclnnMoeDistributeCombineV3GetWorkspaceSize</code>获取。</td>
   <td>uint64_t</td>
  </tr>
  <tr>
   <td>executor</td>
   <td>输入</td>
   <td>op执行器，包含了算子计算流程。</td>
   <td>aclOpExecutor*</td>
  </tr>
  <tr>
   <td>stream</td>
   <td>输入</td>
   <td>指定执行任务的Stream。</td>
   <td>aclOpStream*</td>
  </tr>
 </tbody>
</table>

### 返回值

返回aclnnStatus状态码，具体参见aclnn返回码。

## 约束说明

- **接口配套约束**：
  - `aclnnMoeDistributeDispatchV3`与`aclnnMoeDistributeCombineV3`必须配套使用，前者输出的`assistInfoForCombineOut`、`epRecvCountsOut`、`tpRecvCountsOut`、`expandScalesOut`需直接传入后者对应参数，业务逻辑不可依赖这些Tensor的具体值。

- **参数一致性约束**：
  - 所有卡的`groupEp`、`epWorldSize`、`moeExpertNum`、`groupTp`、`tpWorldSize`、`expertShardType`、`sharedExpertNum`、`sharedExpertRankNum`、`globalBs`、`commAlg`参数及`HCCL_BUFFSIZE`取值需保持一致，且与`aclnnMoeDistributeDispatchV3`对应参数一致。
  - 动态缩容后的部署信息通过`elasticInfoOptional`参数传递给算子，无需修改其他参数，动态缩容后，MOE专家卡上的本卡部署MOE专家数需与缩容前保持一致，不支持缩容后无MOE专家卡。

- **产品特定约束**：
  - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：单卡包含双DIE（晶粒/裸片），参数说明中的“本卡”均指单DIE。
  - 动态缩容功能不支持在TP并行场景下使能，即仅在 `tpWorldSize` 取值为 1 时生效。

- **Shape变量约束**：
  | 变量         | 定义与取值范围                                                                 |
  | :----------- | :----------------------------------------------------------------------------- |
  | A            | 本卡需分发的最大token数,取值范围如下: <ul><li>不使能动态缩容场景时：<ul><li>对于共享专家，要满足<code>A = Bs * epWorldSize \* sharedExpertNum / sharedExpertRankNum</code>。</li><li>对于MoE专家，当globalBs为0时，要满足<code>A >= Bs * epWorldSize * min(localExpertNum, K)</code>；当globalBs非0时，要满足<code>A >= globalBs * min(localExpertNum, K)</code>。</li></ul></li><li>使能动态缩容场景时：<ul><li>当globalBs为0时，<code>A >= max(Bs * epWorldSize \* sharedExpertNum / sharedExpertRankNum, Bs * epWorldSize * min(localExpertNum, K))</code>；</li><li>当globalBs非0时，<code>A >= max(Bs * epWorldSize \* sharedExpertNum / sharedExpertRankNum, globalBs * min(localExpertNum, K))</code>；</li></ul></li><ul>
  | H            |表示hidden size隐藏层大小:<ul><li> <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>：依commAlg取值，"fullmesh"支持(0, 7168]且为32的整数倍；"hierarchy"并且驱动版本≥25.0.RC1.1时支持(0, 10*1024]且为32的整数倍；</li><li><term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：[1024, 8192]。 |
  | Bs           | 本卡最终输出token数:<ul><li> <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>：0 < Bs ≤256；</li><li><term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：0 < Bs ≤512。 |
  | K            |表示选取topK个专家:<br> 0 < K ≤16，且0 < K ≤ <code>moeExpertNum+zeroExpertNum+copyExpertNum+constExpertNum</code>。 |
  | serverNum    | 服务器节点数:<br>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件：仅该场景的shape使用了该变量,仅支持2、4、8。
  | localExpertNum | 本卡专家数。<ul><li>对于共享专家卡，localExpertNum = 1；</li><li>对于MoE专家卡，localExpertNum = <code>moeExpertNum/(epWorldSize-sharedExpertRankNum)</code>，localExpertNum > 1时不支持TP通信。 |

- **环境变量约束**：
  - **HCCL_BUFFSIZE**：调用本接口前需检查HCCL_BUFFSIZE环境变量取值是否合理，该环境变量表示单个通信域占用内存大小，单位MB，不配置时默认为200MB。
      - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>：
        - commAlg配置为""或nullptr：依照HCCL_INTRA_PCIE_ENABLE和HCCL_INTRA_ROCE_ENABLE环境变量配置，选择"fullmesh"或"hierarchy"公式。
        - commAlg配置为"fullmesh": 要求 <code>= 2 \* (Bs \* epWorldSize \* min(localExpertNum, K) \* H \* sizeof(uint16) + 2MB)</code>。
        - commAlg配置为"hierarchy": 要求 <code>= moeExpertNum \* Bs \* (H \* sizeof(dtypeX) + 4 \* ((K + 7) / 8 \* 8) \* sizeof(uint32)) + 4MB + 100MB</code>，不要求<code>moeExpertNum / (epWorldSize - sharedExpertRankNum) <= 24</code>。
      - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：要求 <code>>= 2且满足>= 2 \* (localExpertNum \* maxBs \* epWorldSize \* Align512(Align32(2 \* H) + 44) + (K + sharedExpertNum) \* maxBs \* Align512(2 \* H))</code>，<code>localExpertNum</code>需使用MoE专家卡的本卡专家数，其中<code>Align512(x) = ((x + 512 - 1) / 512) \* 512，Align32(x) = ((x + 32 - 1) / 32) \* 32</code>。
  - **HCCL_INTRA_PCIE_ENABLE/HCCL_INTRA_ROCE_ENABLE**：
    - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>：该环境变量不再推荐使用，建议commAlg配置"hierarchy"。

- **通信域使用约束**：
  - 一个模型中，`aclnnMoeDistributeCombineV3`与`aclnnMoeDistributeDispatchV3`仅支持相同EP通信域，且该域不允许有其他算子。
  - 若使用TP通信域，两者需使用相同TP域或都不使用，TP域不允许有其他算子。

- **其他约束**：
  - 公式中的“/”表示整除。
  - <code>moeExpertNum + zeroExpertNum + copyExpertNum + constExpertNum < MAX_INT32</code>。

## 调用示例


<term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term> ：类似下文<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>调用示例，其中V3接口相较于V2接口新增的场景参数按上述参数说明传值即可。

<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：示例代码如下，仅供参考，调起aclnnMoeDistributeCombineV3和aclnnMoeDistributeDispatchV3接口。本示例代码仅支持Atlas A3。




- 文件准备：
  1.新建combineDemo目录，按照下方指导在combineDemo下新建aclnnCombineDemo.cpp，buildCombine.sh，文件并参考如下代码修改。

  2.安装cann包，并根据下方指导编译运行combineDemo。

-  编译脚本
    ```bash
    #!/bin/bash
    cann_path="/path/to/cann_env" # 更改cann包环境的路径
    g++ "aclnnCombineDemo.cpp" -o combineDemo -I"$cann_path/latest/include/" -I"$cann_path/latest/include/aclnnop/" \
                        -L="$cann_path/latest/lib64/" -lascendcl -lnnopbase -lopapi -lop_common -lpthread -lhccl
    ```
- 编译与运行：

    ```bash
    # source cann环境
    source /path/to/cann_env/latest/bin/setenv.bash

    # 编译aclnnCombineDemo.cpp
    bash buildCombine.sh

    ./combineDemo
    ```

- 示例代码如下，仅供参考
    ```Cpp
    #include <thread>
    #include <iostream>
    #include <string>
    #include <vector>
    #include <unordered_set>
    #include "acl/acl.h"
    #include "hccl/hccl.h"
    #include "aclnnop/aclnn_moe_distribute_dispatch_v3.h"
    #include "aclnnop/aclnn_moe_distribute_combine_v3.h"

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
    constexpr uint32_t TP_WORLD_SIZE = 1;
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
        int64_t zeroExpertNum = 1;
        int64_t copyExpertNum = 1;
        int64_t constExpertNum = 1;
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
        void *residualXDeviceAddr = nullptr;
        void *sharedExpertXDeviceAddr = nullptr;

        //动态缩容和零专家场景输入
        void *elasticInfoDeviceAddr = nullptr;
        void *oriXDeviceAddr = nullptr;
        void *constExpertAlpha1DeviceAddr = nullptr;
        void *constExpertAlpha2DeviceAddr = nullptr;
        void *constExpertVDeviceAddr = nullptr;

        void *xOutDeviceAddr = nullptr;

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
        aclTensor *residualX = nullptr;
        aclTensor *sharedExpertX = nullptr;


        aclTensor *elasticInfo = nullptr;
        aclTensor *oriX = nullptr;
        aclTensor *constExpertAlpha1 = nullptr;
        aclTensor *constExpertAlpha2 = nullptr;
        aclTensor *constExpertV = nullptr;

        aclTensor *xOut = nullptr;

        //定义当前场景下各变量维度
        std::vector<int64_t> xShape{Bs, H};
        std::vector<int64_t> expertIdsShape{Bs, K};
        std::vector<int64_t> scalesShape{moeExpertNum + 1, H};
        std::vector<int64_t> expertScalesShape{Bs, K};

        std::vector<int64_t> expandXShape{TP_WORLD_SIZE * A, H};
        std::vector<int64_t> dynamicScalesShape{TP_WORLD_SIZE * A};
        std::vector<int64_t> expandIdxShape{A * 128};
        std::vector<int64_t> expertTokenNumsShape{localExpertNum};
        std::vector<int64_t> epRecvCountsShape{TP_WORLD_SIZE * localExpertNum * EP_WORLD_SIZE};
        std::vector<int64_t> tpRecvCountsShape{TP_WORLD_SIZE * localExpertNum};
        std::vector<int64_t> expandScalesShape{A};
        std::vector<int64_t> sharedExpertXShape{Bs, 1, H};


        std::vector<int64_t> elasticInfoShape{4 + EP_WORLD_SIZE * 2};
        std::vector<int64_t> oriXShape{Bs, H};
        std::vector<int64_t> constExpertAlpha1Shape{constExpertNum};
        std::vector<int64_t> constExpertAlpha2Shape{constExpertNum};
        std::vector<int64_t> constExpertVShape{constExpertNum, H};

        std::vector<int64_t> xOutShape{Bs, H};

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
        int64_t sharedExpertXShapeSize = GetShapeSize(sharedExpertXShape);

        int64_t elasticInfoSize = GetShapeSize(elasticInfoShape);
        int64_t oriXSize = GetShapeSize(oriXShape);
        int64_t constExpertAlpha1Size = GetShapeSize(constExpertAlpha1Shape);
        int64_t constExpertAlpha2Size = GetShapeSize(constExpertAlpha2Shape);
        int64_t constExpertVSize = GetShapeSize(constExpertVShape);

        int64_t xOutShapeSize = GetShapeSize(xOutShape);

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
        std::vector<int16_t> sharedExpertXHostData(sharedExpertXShapeSize, 1);

        int32_t isElastic = 1;
        int32_t rankNumAfterElastic = 4;
        int32_t sharedExpertRankNumAfterElastic = sharedExpertRankNum;
        int32_t moeExpertNumAfterElastic = rankNumAfterElastic - sharedExpertRankNumAfterElastic;
        std::unordered_set<int16_t> availableRank{
            0, 1, /*2, 3, 4, 5,*/ 6, 7
        };
        std::vector<int32_t> elasticInfoHostData{
            isElastic, rankNumAfterElastic, sharedExpertRankNumAfterElastic, moeExpertNumAfterElastic,
            0, 1, -1, -1, -1, -1, 2, 3,
            0, 1, 6, 7, -1, -1, -1, -1
        };
        std::vector<int16_t> oriXHostData(oriXSize, 1);
        std::vector<int16_t> constExpertAlpha1HostData(constExpertAlpha1Size, 0);
        std::vector<int16_t> constExpertAlpha2HostData(constExpertAlpha2Size, 0);
        std::vector<int16_t> constExpertVHostData(constExpertVSize, 0);

        std::vector<int16_t> xOutHostData(xOutShapeSize, 0);


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
        ret = CreateAclTensor(sharedExpertXHostData, sharedExpertXShape, &sharedExpertXDeviceAddr, aclDataType::ACL_BF16, &sharedExpertX);
        CHECK_RET(ret == ACL_SUCCESS, return ret);

        ret = CreateAclTensor(elasticInfoHostData, elasticInfoShape, &elasticInfoDeviceAddr, aclDataType::ACL_INT32, &elasticInfo);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(oriXHostData, oriXShape, &oriXDeviceAddr, aclDataType::ACL_BF16, &oriX);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(constExpertAlpha1HostData, constExpertAlpha1Shape, &constExpertAlpha1DeviceAddr, aclDataType::ACL_BF16, &constExpertAlpha1);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(constExpertAlpha2HostData, constExpertAlpha2Shape, &constExpertAlpha2DeviceAddr, aclDataType::ACL_BF16, &constExpertAlpha2);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(constExpertVHostData, constExpertVShape, &constExpertVDeviceAddr, aclDataType::ACL_BF16, &constExpertV);
        CHECK_RET(ret == ACL_SUCCESS, return ret);

        ret = CreateAclTensor(xOutHostData, xOutShape, &xOutDeviceAddr, aclDataType::ACL_BF16, &xOut);
        CHECK_RET(ret == ACL_SUCCESS, return ret);



        uint64_t dispatchWorkspaceSize = 0;
        aclOpExecutor *dispatchExecutor = nullptr;
        void *dispatchWorkspaceAddr = nullptr;

        uint64_t combineWorkspaceSize = 0;
        aclOpExecutor *combineExecutor = nullptr;
        void *combineWorkspaceAddr = nullptr;
        /**************************************** 调用dispatch warm up********************************************/
        // 模拟动态缩容场景，需要先运行一遍正常情况建立通信域；调用第一阶段接口
        ret = aclnnMoeDistributeDispatchV3GetWorkspaceSize(x, expertIds, (quantMode > 0 ? scales : nullptr), nullptr,
                expertScales, nullptr, hcomEpName, EP_WORLD_SIZE, args.epRankId, moeExpertNum, hcomTpName, TP_WORLD_SIZE,
                args.tpRankId, expertShardType, sharedExpertNum,sharedExpertRankNum, quantMode, globalBs,
                expertTokenNumsType, nullptr, zeroExpertNum, copyExpertNum, constExpertNum, expandX, dynamicScales, expandIdx, expertTokenNums, epRecvCounts,
                tpRecvCounts, expandScales, &dispatchWorkspaceSize, &dispatchExecutor);

        CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("[ERROR] warm up aclnnMoeDistributeDispatchV3GetWorkspaceSize failed. ret = %d \n", ret); return ret);

        if (dispatchWorkspaceSize > 0) {
            ret = aclrtMalloc(&dispatchWorkspaceAddr, dispatchWorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] warm up aclrtMalloc workspace failed. ret = %d \n", ret); return ret);
        }
        // 调用第二阶段接口
        ret = aclnnMoeDistributeDispatchV3(dispatchWorkspaceAddr, dispatchWorkspaceSize,
                                            dispatchExecutor, args.dispatchStream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] warm up aclnnMoeDistributeDispatchV3 failed. ret = %d \n", ret);  \
                return ret);
        ret = aclrtSynchronizeStreamWithTimeout(args.dispatchStream, 10000);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] warm up aclrtSynchronizeStreamWithTimeout failed. ret = %d \n", ret);  \
            return ret);

        /**************************************** 调用dispatch ********************************************/
        if (availableRank.find(args.rankId) != availableRank.end()) {
            // 调用第一阶段接口
        ret = aclnnMoeDistributeDispatchV3GetWorkspaceSize(x, expertIds, (quantMode > 0 ? scales : nullptr), nullptr,
                expertScales, elasticInfo, hcomEpName, EP_WORLD_SIZE, args.epRankId, moeExpertNum, hcomTpName, TP_WORLD_SIZE,
                args.tpRankId, expertShardType, sharedExpertNum,sharedExpertRankNum, quantMode, globalBs,
                expertTokenNumsType, nullptr, zeroExpertNum, copyExpertNum, constExpertNum, expandX, dynamicScales, expandIdx, expertTokenNums, epRecvCounts,
                tpRecvCounts, expandScales, &dispatchWorkspaceSize, &dispatchExecutor);

        CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("[ERROR] aclnnMoeDistributeDispatchV3GetWorkspaceSize failed. ret = %d \n", ret); return ret);

        if (dispatchWorkspaceSize > 0) {
            ret = aclrtMalloc(&dispatchWorkspaceAddr, dispatchWorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc workspace failed. ret = %d \n", ret); return ret);
        }
        // 调用第二阶段接口
        ret = aclnnMoeDistributeDispatchV3(dispatchWorkspaceAddr, dispatchWorkspaceSize,
                                            dispatchExecutor, args.dispatchStream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclnnMoeDistributeDispatchV3 failed. ret = %d \n", ret);  \
                return ret);
        ret = aclrtSynchronizeStreamWithTimeout(args.dispatchStream, 10000);
                    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] dispatch aclrtSynchronizeStreamWithTimeout failed. ret = %d \n", ret);  \
                return ret);
        }
        /**************************************** 调用combine ********************************************/
        // 调用第一阶段接口
        if (availableRank.find(args.rankId) != availableRank.end()) {
        ret = aclnnMoeDistributeCombineV3GetWorkspaceSize(expandX, expertIds,
                                                            expandIdx, epRecvCounts,
                                                            expertScales, tpRecvCounts,
                                                            nullptr, nullptr, nullptr,
                                                            nullptr, nullptr, nullptr,
                                                            elasticInfo, oriX, constExpertAlpha1, constExpertAlpha2, constExpertV,
                                                            hcomEpName, EP_WORLD_SIZE, args.epRankId, moeExpertNum,
                                                            hcomTpName, TP_WORLD_SIZE, args.tpRankId, expertShardType,
                                                            sharedExpertNum, sharedExpertRankNum, globalBs, outDtype,
                                                            commQuantMode, groupList_type, nullptr, zeroExpertNum, copyExpertNum, constExpertNum, xOut,
                                                            &combineWorkspaceSize, &combineExecutor);
        CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("[ERROR] aclnnMoeDistributeCombineV3GetWorkspaceSize failed. ret = %d \n", ret); return ret);
        // 根据第一阶段接口计算出的workspaceSize申请device内存
        if (combineWorkspaceSize > 0) {
            ret = aclrtMalloc(&combineWorkspaceAddr, combineWorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc workspace failed. ret = %d \n", ret); return ret);
        }

        // 调用第二阶段接口
        ret = aclnnMoeDistributeCombineV3(combineWorkspaceAddr, combineWorkspaceSize, combineExecutor, args.combineStream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclnnMoeDistributeCombineV3 failed. ret = %d \n", ret);
            return ret);
        // （固定写法）同步等待任务执行结束
        ret = aclrtSynchronizeStreamWithTimeout(args.combineStream, 10000);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSynchronizeStreamWithTimeout failed. ret = %d \n", ret);
            return ret);
        LOG_PRINT("[INFO] device_%d aclnnMoeDistributeDispatchV3 and aclnnMoeDistributeCombineV3                      \
                    execute successfully.\n", args.rankId);
        }
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
        if (residualX != nullptr) {
            aclDestroyTensor(residualX);
        }
        if (sharedExpertX != nullptr) {
            aclDestroyTensor(sharedExpertX);
        }
        if (elasticInfo != nullptr) {
            aclDestroyTensor(elasticInfo);
        }
        if (oriX != nullptr) {
            aclDestroyTensor(oriX);
        }
        if (constExpertAlpha1 != nullptr) {
            aclDestroyTensor(constExpertAlpha1);
        }
        if (constExpertAlpha2 != nullptr) {
            aclDestroyTensor(constExpertAlpha2);
        }
        if (constExpertV != nullptr) {
            aclDestroyTensor(constExpertV);
        }

        if (xOut != nullptr) {
            aclDestroyTensor(xOut);
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
        if (sharedExpertXDeviceAddr != nullptr) {
            aclrtFree(sharedExpertXDeviceAddr);
        }

        if (elasticInfoDeviceAddr != nullptr) {
            aclrtFree(elasticInfoDeviceAddr);
        }
        if (oriXDeviceAddr != nullptr) {
            aclrtFree(oriXDeviceAddr);
        }
        if (constExpertAlpha1DeviceAddr != nullptr) {
            aclrtFree(constExpertAlpha1DeviceAddr);
        }
        if (constExpertAlpha2DeviceAddr != nullptr) {
            aclrtFree(constExpertAlpha2DeviceAddr);
        }
        if (constExpertVDeviceAddr != nullptr) {
            aclrtFree(constExpertVDeviceAddr);
        }

        if (xOutDeviceAddr != nullptr) {
            aclrtFree(xOutDeviceAddr);
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
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclInit failed, ret = %d\n", ret); return ret);

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