# MoeDistributeDispatchV2

## 产品支持情况

| 产品                                                         |  是否支持   |
| :----------------------------------------------------------- |:-------:|
| <term>Ascend 950DT</term>                             |    √    |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>       |    √    |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √    |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×    |
| <term>Atlas 推理系列产品</term>                               |    ×    |
| <term>Atlas 训练系列产品</term>                              |    ×    |

## 功能说明

算子功能：对token数据进行量化（可选），进行EP（Expert Parallelism）域的AllToAllV通信。TP（Tensor Parallelism）域通信当前版本不再支持，TP相关参数为预留参数。

- 情形1：如果quantMode=0（非量化场景）：

$$
allToAllXOut = AllToAllV(X)\\
expandXOut = AllToAllV(X)
$$

- 情形2：如果quantMode=1（静态量化场景）：

$$
xFp32 = CastToFp32(X) \times scales \\
quantOut = Cast(xFp32, dstType) \\
allToAllXOut = AllToAllV(quantOut)\\
expandXOut = AllToAllV(quantOut)
$$

- 情形3：如果quantMode=2（pertoken动态量化场景）：

$$
xFp32 = CastToFp32(X) \times scales \\
dynamicScales = dstTypeMax/Max(Abs(xFp32)) \\
quantOut = CastToInt8(xFp32 \times dynamicScales) \\
allToAllXOut = AllToAllV(quantOut) \\
allToAllDynamicScalesOut = AllToAllV(1.0/dynamicScales) \\
expandXOut = AllToAllV(quantOut) \\
dynamicScalesOut = allToAllDynamicScalesOut
$$

- 情形4：如果quantMode=3（pertile动态量化场景）：

$$
xFp32 = CastToFp32(X) \times scales \\
dynamicScales = dstTypeMax/Max(Abs(xFp32)) \\
quantOut = CastToInt8(xFp32 \times dynamicScales) \\
allToAllXOut = AllToAllV(quantOut) \\
allToAllDynamicScalesOut = AllToAllV(1.0/dynamicScales) \\
expandXOut = AllToAllV(quantOut) \\
dynamicScalesOut = allToAllDynamicScalesOut
$$

- 情形5：如果quantMode=4（mx量化场景）：

$$
sharedExp = Floor(log_2(max(x))) - emax \\
dynamicScales = 2^{sharedExp} \\
quantOut = CastToFp8(X / dynamicScales) \\
allToAllXOut = AllToAllV(quantOut) \\
allToAllDynamicScalesOut = AllToAllV(1.0 / dynamicScales) \\
expandXOut = AllToAllV(quantOut) \\
dynamicScalesOut = allToAllDynamicScalesOut
\end{cases}
$$

其中，$emax$表示该类型最大正规数对应的指数部分的值。

- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：该算子必须与`MoeDistributeCombineV2`配套使用。
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、Ascend 950DT：该算子必须与`MoeDistributeCombineV2`或`MoeDistributeCombineAddRmsNorm`配套使用。

> 说明：MoeDistributeCombineV2、MoeDistributeCombineAddRmsNorm算子在后续文档中统称为CombineV2系列算子。
・
相较于`MoeDistributeDispatch`算子，该算子变更如下：

- 输出了更详细的token信息辅助`CombineV2`系列算子高效地进行全卡同步，因此原算子中shape为(`BS` * `K`,)的`expandIdx`出参替换为shape为(`A` * 128,)的`assistInfoForCombineOut`参数；
- 新增`commAlg`入参，代替`HCCL_INTRA_PCIE_ENABLE`和`HCCL_INTRA_ROCE_ENABLE`环境变量。

详细说明请参考以下参数说明。

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
<td>x</td>
    <td>输入</td>
    <td>本卡发送的token数据。</td>
    <td>FLOAT16、BFLOAT16、FLOAT8_E5M2、FLOAT8_E4M3FN、HIFLOAT8、FLOAT4_E2M1、FLOAT4_E1M2</td>
    <td>ND</td>
   </tr>
   <tr>
    <td>expertIds</td>
   <td>输入</td>
   <td>每个token的topK个专家索引。</td>
   <td>INT32</td>
   <td>ND</td>
  </tr>
  <tr>
<td>scalesOptional</td>
    <td>可选输入</td>
    <td>每个专家的量化平滑参数。</td>
    <td>FLOAT32、FLOAT8_E8M0</td>
    <td>ND</td>
   </tr>
  <tr>
   <td>xActiveMaskOptional</td>
   <td>可选输入</td>
   <td>表示token是否参与通信，可传有效数据或空指针；1D时true需排在false前（例：{true, false, true}非法），2D时token对应K个值全为false则不参与通信；默认所有token参与通信；各卡BS不一致时所有token需有效。</td>
   <td>BOOL</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>expertScalesOptional</td>
   <td>可选输入</td>
   <td>每个token的topK个专家权重。</td>
   <td>FLOAT32</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>elasticInfoOptional</td>
   <td>可选输入</td>
   <td>EP通信域动态缩容信息。当某些通信卡因异常而从通信域中剔除，实际参与通信的卡数可从本参数中获取。</td>
   <td>FLOAT32</td>
   <td>ND</td>
  </tr>
  <tr>
    <td>performanceInfoOptional</td>
    <td>可选输入</td>
    <td>表示本卡等待各卡数据的通信时间，单位为us（微秒）。单次算子调用各卡通信耗时会累加到该Tensor上，算子内部不进行自动清零，因此用户每次启用此Tensor开始记录耗时前需对Tensor清零。</td>
    <td>INT64</td>
    <td>ND</td>
  </tr>
  <tr>
   <td>groupEp</td>
   <td>属性</td>
    <td>EP通信域名称（专家并行通信域），字符串长度范围为[1, 128)。</td>
   <td>STRING</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>epWorldSize</td>
   <td>属性</td>
   <td>EP通信域大小。</td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>epRankId</td>
   <td>属性</td>
   <td>EP域本卡ID，取值范围[0, epWorldSize)，同一个EP通信域中各卡的epRankId不重复。</td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>moeExpertNum</td>
   <td>属性</td>
   <td>MoE专家数量，满足moeExpertNum % (epWorldSize - sharedExpertRankNum) = 0。</td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>groupTp</td>
   <td>可选属性</td>
   <td><li>TP通信域名称（预留参数，当前版本不支持）。</li><li>默认值为""。</li></td>
   <td>STRING</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>tpWorldSize</td>
   <td>可选属性</td>
   <td><li>TP通信域大小（预留参数，当前版本不支持，仅支持传0或1）。</li><li>默认值为0。</li></td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>tpRankId</td>
   <td>可选属性</td>
   <td><li>TP域本卡ID（预留参数，当前版本不支持，仅支持传0）。</li><li>默认值为0。</li></td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>expertShardType</td>
   <td>可选属性</td>
   <td><li>表示共享专家卡分布类型，当前仅支持传0，表示共享专家卡排在MoE专家卡前面。</li><li>默认值为0。</li></td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>sharedExpertNum</td>
   <td>可选属性</td>
   <td><li>表示共享专家数量（一个共享专家可复制部署到多个卡上）。</li><li>默认值为1。</li></td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>sharedExpertRankNum</td>
   <td>可选属性</td>
   <td><li>表示共享专家卡数量，取值范围[0, epWorldSize)；为0时需满足sharedExpertNum为0或1，不为0时需满足sharedExpertRankNum % sharedExpertNum = 0。</li><li>默认值为0。</li></td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>quantMode</td>
   <td>可选属性</td>
   <td><li>表示量化模式。默认值为0。</li></td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>globalBS</td>
   <td>可选属性</td>
   <td><li>EP域全局的batch size容量；各rank BS一致时，globalBS = BS * epWorldSize或0；各rank BS不一致时，globalBS = maxBS * epWorldSize（maxBS为业务配置的单rank BS容量上限，且不小于任意rank的实际BS）。各rank传入的globalBS必须一致。</li><li>默认值为0；各rank BS不一致时不能传0。</li></td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>expertTokenNumsType</td>
   <td>可选属性</td>
   <td><li>输出expertTokenNums中值的语义类型，支持0：expertTokenNums中的输出为每个专家处理的token数的前缀和，1：expertTokenNums中的输出为每个专家处理的token数量。</li><li>默认值为1。</li></td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>commAlg</td>
   <td>可选属性</td>
   <td><li>表示通信亲和内存布局算法。</li><li>默认值为""。</li></td>
   <td>STRING</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>zeroExpertNum</td>
   <td>可选属性</td>
   <td><li>零专家数量。</li><li>默认值为0。</li></td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>copyExpertNum</td>
   <td>可选属性</td>
   <td><li>copy专家数量。</li><li>默认值为0。</li></td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>constExpertNum</td>
   <td>可选属性</td>
   <td><li>常量专家数量。</li><li>默认值为0。</li></td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
    <td>expandXOut</td>
    <td>输出</td>
    <td>根据expertIds进行扩展过的token特征。</td>
    <td>FLOAT16、BFLOAT16、INT8、FLOAT8_E4M3FN、FLOAT8_E5M2、HIFLOAT8、FLOAT4_E2M1、FLOAT4_E1M2</td>
    <td>ND</td>
   </tr>
   <tr>
    <td>dynamicScalesOut</td>
    <td>输出</td>
    <td>量化场景下，表示本卡输出Token的量化系数。</td>
    <td>FLOAT32、FLOAT8_E8M0</td>
    <td>ND</td>
   </tr>
  <tr>
   <td>assistInfoForCombineOut</td>
   <td>输出</td>
   <td>表示给同一专家发送的token个数（对应CombineV2系列算子中的assistInfoForCombine）。</td>
   <td>INT32</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>expertTokenNumsOut</td>
   <td>输出</td>
   <td>表示每个专家收到的token个数。</td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>epRecvCountsOut</td>
   <td>输出</td>
   <td>从EP通信域各卡接收的token数（对应CombineV2系列算子中的epSendCounts）。</td>
   <td>INT32</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>tpRecvCountsOut</td>
   <td>输出</td>
   <td>从TP通信域各卡接收的token数（预留输出，当前版本不支持，传空指针即可），无TP域通信则无该输出。</td>
   <td>INT32</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>expandScalesOut</td>
   <td>输出</td>
   <td>表示本卡输出token的权重（对应CombineV2系列算子中的expandScalesOptional）。</td>
   <td>FLOAT32</td>
   <td>ND</td>
  </tr>
 </tbody>
</table>

* <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：
    * 不支持共享专家场景，不支持`expertShardType`、`sharedExpertNum`、`sharedExpertRankNum`属性。
    * 仅支持EP域，无TP域，不支持`groupTp`、`tpWorldSize`、`tpRankId`属性，且`tpRecvCounts`输出无有效内容。
    * 不支持`elasticInfoOptional`。
    * 当`commAlg` = "hierarchy"，`expandScalesOut`内容有效。
    * 不支持常量专家场景，不支持`constExpertNum`，使用默认值即可。
* <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>  ：
    * commAlg支持""，"fullmesh_v1"，"fullmesh_v2", "hierarchy"三种输入方式。""：默认值，不开启fullmesh_v2模板；"fullmesh_v1"：不开启fullmesh_v2模板；"fullmesh_v2"：开启fullmesh_v2模板，不支持在各卡`BS`不一致、输入xActiveMask和特殊专家场景下开启；"hierarchy": 开启跨超模板，支持各rank `BS`不一致，仅支持共享专家为0的场景，且不支持二维mask、特殊专家、performanceInfo场景。可变BS场景需为所有rank配置相同且非0的`globalBS`容量，并且不传`xActiveMaskOptional`。
    * `expertScalesOptional`当`commAlg`="hierarchy"场景时，要求为2D Tensor，shape为(BS, K)；当`commAlg`=""、"fullmesh_v1"、"fullmesh_v2"场景时，可传有效Tensor或空指针，有效Tensor为2D Tensor，shape为(BS, K)，传空指针时`expertScalesOptional`无效，不支持空Tensor。
    * epWorldSize取值范围[2, 768]；当commAlg="hierarchy"场景时，取值范围为[16, 256]，且为16的整数倍。
    * moeExpertNum取值范围(0, 1024]；当commAlg="hierarchy"场景时，取值范围为(0, 512]。
    * `expandScalesOut`当`commAlg`="hierarchy"场景时，要求为1D Tensor，shape为(A,)；当`commAlg`=""、"fullmesh_v1"、"fullmesh_v2"场景时暂不支持该输出。
* <term>Ascend 950DT</term>：
    * 仅支持EP域，无TP域，不支持`groupTp`、`tpWorldSize`、`tpRankId`属性，且`tpRecvCounts`输出无有效内容。
    * `expertScalesOptional`可传有效Tensor或空指针。传有效Tensor时使能expertScales专家权重功能，有效Tensor为2D Tensor，shape为(BS, K)；传空指针时不使能expertScales专家权重功能。不支持空Tensor。
    * 使能expertScales专家权重功能时，`expandScalesOut`输出有效；不使能时，该输出无效。

## 约束说明

- `MoeDistributeDispatchV2`与`CombineV2`系列算子必须配套使用，具体参考调用示例。

- 算子通信域各节点的驱动版本应当相同。

- 在不同产品型号、不同通信算法或不同版本中，`MoeDistributeDispatchV2`的Tensor输出`assistInfoForCombineOut`、`epRecvCountsOut`、`tpRecvCountsOut`、`expandScalesOut`中的元素值可能不同，使用时直接将上述Tensor传给`CombineV2`系列算子对应参数即可，模型其他业务逻辑不应对其存在依赖。

- 调用算子过程中使用的`groupEp`、`epWorldSize`、`moeExpertNum`、`groupTp`、`tpWorldSize`、`expertShardType`、`sharedExpertNum`、`sharedExpertRankNum`、`globalBS`、`commAlg`参数，`HCCL_BUFFSIZE`取值所有卡需保持一致，网络中不同层中也需保持一致，且和`CombineV2`系列算子对应参数也保持一致。

- 参数说明里shape格式说明：
    - `A`：表示本卡可能接收的最大token数量，取值范围如下：
        - 对于共享专家，要满足`A` = `BS` *`epWorldSize`* `sharedExpertNum` / `sharedExpertRankNum`。
        - 对于MoE专家，当`globalBS`为0时，要满足`A` >= `BS` *`epWorldSize`* min(`localExpertNum`, `K`)；当`globalBS`非0时，要满足`A` >= `globalBS` * min(`localExpertNum`, `K`)。
    - `K`：表示选取topK个专家，取值范围为0 < `K` ≤ 16同时满足0 < `K` ≤ `moeExpertNum` + `zeroExpertNum` + `copyExpertNum` + `constExpertNum`。
    - `localExpertNum`：表示本卡专家数量。
        - 对于共享专家卡，`localExpertNum` = 1
        - 对于MoE专家卡，`localExpertNum` = `moeExpertNum` / (`epWorldSize` - `sharedExpertRankNum`)，当前版本不支持TP域通信。

- 属性约束：
    - `zeroExpertNum`：取值范围：[0, MAX_INT32)，MAX_INT32 = 2^31 - 1,合法的零专家的ID的值是[`moeExpertNum`, `moeExpertNum` + `zeroExpertNum`)。
    - `copyExpertNum`：取值范围：[0, MAX_INT32)，MAX_INT32 = 2^31 - 1,合法的copy专家的ID的值是[`moeExpertNum` + `zeroExpertNum`, `moeExpertNum` + `zeroExpertNum` + `copyExpertNum`)。
    - `constExpertNum`：取值范围：[0, MAX_INT32)，MAX_INT32 = 2^31 - 1,合法的常量专家的ID的值是[`moeExpertNum` + `zeroExpertNum` + `copyExpertNum`, `moeExpertNum` + `zeroExpertNum` + `copyExpertNum` + `constExpertNum`)。

- 本文公式中的"/"表示整除。
- 通信域使用约束：
    - 一个模型中的`CombineV2`系列算子和`MoeDistributeDispatchV2`仅支持相同EP通信域，且该通信域中不允许有其他算子。
    - 当前不支持TP域通信。

- 通信方式约束：
    - <term>Ascend 950DT</term>：仅支持UB Memory通信。

- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：
    - 参数约束：
        - `commAlg`：当前版本支持nullptr，""，"fullmesh"，"hierarchy"四种输入方式，若配置"hierarchy"，建议搭配搭配25.0.RC1.1及以上版本驱动使用。
            - nullptr和""：仅在此场景下，`HCCL_INTRA_PCIE_ENABLE`和`HCCL_INTRA_ROCE_ENABLE`配置生效。当`HCCL_INTRA_PCIE_ENABLE`=1&&`HCCL_INTRA_ROCE_ENABLE`=0时，调用"hierarchy"算法，否则调用"fullmesh"算法。不推荐使用该方式。
            - "fullmesh"：token数据直接通过RDMA方式发往topk个目标专家所在的卡。
            - "hierarchy"：token数据经过跨机、机内两次发送，仅不同server同号卡之间使用RDMA通信，server内使用HCCS通信。
        - `epWorldSize`：依commAlg取值，"fullmesh"支持2、3、4、5、6、7、8、16、32、64、128、192、256、384；"hierarchy"支持16、32、64。
        - `moeExpertNum`：依commAlg取值，"fullmesh"支持(0, 1024]，"hierarchy"支持(0, 512]。
        - `epRecvCountsOut`：要求shape为(`moeExpertNum` + 2 * `globalBS` * `K` * `serverNum`,)，前`moeExpertNum`个数表示从EP通信域各卡接收的token数，2 * `globalBS` * `K` * `serverNum`存储了机间机内做通信前combine可以提前做reduce的token个数和token在通信区中的偏移，`globalBS`传入0时在此处应当按照`BS` * `epWorldSize`计算。
        - `performanceInfoOptional`：可选择传入有效数据或填空指针，传入空指针时表示不开启记录通信耗时功能；当传入有效数据时，要求是一个1D的Tensor，shape为(ep\_world\_size,)，数据类型支持int64；数据格式要求为ND。
    - `HCCL_INTRA_PCIE_ENABLE`和`HCCL_INTRA_ROCE_ENABLE`：不推荐使用该环境变量控制通信算法，原`HCCL_INTRA_PCIE_ENABLE`=1&&`HCCL_INTRA_ROCE_ENABLE`=0场景，下文均通过`commAlg` = "hierarchy"替代，默认场景使用`commAlg` = "fullmesh"替代。
    - `commAlg`配置"hierarchy"时，不支持`scalesOptional`、`xActiveMaskOptional`、`oriXOptional`、`zeroExpertNum`、`copyExpertNum`。
    - `quantMode`支持0（非量化）、2（pertoken动态量化）。
    - 参数说明里shape格式说明：
        - `H`：表示hidden size隐藏层大小，取值范围(0, 10240]且为32的整数倍。
        - `BS`：表示batch sequence size，即本卡最终输出的token数量。
            - `commAlg` = "fullmesh"：取值范围(0, 256]。
            - `commAlg` = "hierarchy"：取值范围(0, 512]。
    - `HCCL_BUFFSIZE`：调用本算子前需检查`HCCL_BUFFSIZE`环境变量取值是否合理，该环境变量表示单个通信域占用内存大小，单位MB，不配置时默认为200MB。
        - `commAlg` = "fullmesh"：要求 >= (`BS` *`epWorldSize`* min(`localExpertNum`, `K`) *`H`* 4B + 4MB)。
        - `commAlg` = "hierarchy"：要求 >= (`moeExpertNum` + `epWorldSize` / 4) *Align512(`maxBS`* (`H` *2 + 16* Align8(`K`))) *1B + 8MB，其中Align8(x) = ((x + 8 - 1) / 8)* 8，Align512(x) = ((x + 512 - 1) / 512) * 512。
    - 组网约束：多机场景仅支持交换机组网，不支持双机直连组网。

- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>  ：
    - 该场景下单卡包含双DIE（简称为“晶粒”或“裸片”），因此参数说明里的“本卡”均表示单DIE。
    - 参数约束：
        - `elasticInfoOptional`：当前版本不支持，传空指针即可。
        - `epWorldSize`：取值范围[2, 768]。
        - `moeExpertNum`：取值范围(0, 1024]。
        - `groupTp`：预留参数，TP域通信不再支持。
        - `sharedExpertNum`：取值支持[0, 4]。
        - `commAlg`：当前版本仅支持""，"fullmesh_v1"，"fullmesh_v2"，"hierarchy"三种输入方式。
            - ""：默认值，开启fullmesh_v1模板。
            - "fullmesh_v1"：开启fullmesh_v1模板。
            - "fullmesh_v2"：开启fullmesh_v2模板，不支持在各卡`BS`不一致、输入xActiveMask和特殊专家场景下开启。
            - "hierarchy": 开启ROCE分层直驱能力，需要根据不同的逻辑超节点设置环境变量`HCCL_LOGIC_SUPERPOD_ID`，例如两机分别设为`export HCCL_LOGIC_SUPERPOD_ID=0`和`export HCCL_LOGIC_SUPERPOD_ID=1`。
        - `epRecvCountsOut`：要求shape为(`epWorldSize` * `localExpertNum`,)。
        - `performanceInfoOptional`：预留参数，当前版本不支持，传空指针即可。
    - 参数说明里shape格式说明：
        - `H`：表示hidden size隐藏层大小，取值范围[1024, 8192]。
        - `BS`：表示batch sequence size，即本卡最终输出的token数量。
            - `commAlg` = "fullmesh_v2"或"hierarchy"：取值范围(0, 256]。
            - `commAlg` = "fullmesh_v1"或""：取值范围(0, 512]。
    - `HCCL_BUFFSIZE`：调用本算子前需检查`HCCL_BUFFSIZE`环境变量取值是否合理，该环境变量表示单个通信域占用内存大小，单位MB，不配置时默认为200MB。要求 >= 2且满足>= 2 *(`localExpertNum`* `maxBS` *`epWorldSize`* Align512(Align32(2 *`H`) + 64) + (`K` + `sharedExpertNum`)* `maxBS` *Align512(2* `H`))，`localExpertNum`需使用MoE专家卡的本卡专家数，其中Align512(x) = ((x + 512 - 1) / 512) *512，Align32(x) = ((x + 32 - 1) / 32)* 32。

- <term>Ascend 950DT</term>：
    - 参数约束：
        - `elasticInfoOptional`：当前版本不支持，传空指针即可。
        - `epWorldSize`：取值范围[2, 1024]。
        - `moeExpertNum`：取值范围(0, 1024]。
        - `sharedExpertNum`：取值支持[0, 4]。
        - `commAlg`：当前版本仅支持""，"fullmesh_v1"，"fullmesh_v2"三种输入方式。
            - ""：默认值，开启fullmesh_v1模板。
            - "fullmesh_v1"：开启fullmesh_v1模板。
            - "fullmesh_v2"：开启fullmesh_v2模板，不支持在各卡`BS`不一致、输入xActiveMask和特殊专家场景下开启。
        - `epRecvCountsOut`：要求shape为(`epWorldSize` * `localExpertNum`,)。
        - `performanceInfoOptional`：预留参数，当前版本不支持，传空指针即可。
        - `expertShardType`当前仅支持传0，表示共享专家卡排在MoE专家卡前面。
        - `quantMode`支持0（非量化）、1（静态量化）、2（pertoken动态量化）、3（pergroup动态量化）、4（mx动态量化）。
    - 参数说明里shape格式说明：
        - `H`：表示hidden size隐藏层大小，取值范围[1024, 8192]。
        - `BS`：表示batch sequence size，即本卡最终输出的token数量，取值范围(0, 512]。
    - `HCCL_BUFFSIZE`：调用本算子前需检查`HCCL_BUFFSIZE`环境变量取值是否合理，该环境变量表示单个通信域占用内存大小，单位MB，不配置时默认为200MB。要求 >= 2且满足>= 2 *(`localExpertNum`* `maxBS` *`epWorldSize`* Align512(Align32(2 *`H`) + 64) + (`K` + `sharedExpertNum`)* `maxBS` *Align512(2* `H`))，`localExpertNum`需使用MoE专家卡的本卡专家数，其中Align512(x) = ((x + 512 - 1) / 512) *512，Align32(x) = ((x + 32 - 1) / 32)* 32。

## 调用说明

| 调用方式  | 样例代码                                  | 说明                                                     |
| :--------: | :----------------------------------------: | :-------------------------------------------------------: |
| aclnn接口 | [test_aclnn_moe_distribute_dispatch_v2.cpp](./examples/test_aclnn_moe_distribute_dispatch_v2.cpp) | 通过[aclnnMoeDistributeDispatchV2](./docs/aclnnMoeDistributeDispatchV2.md)接口方式调用moe_distribute_dispatch_v2算子。 |
