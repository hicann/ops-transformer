# MoeDistributeDispatch


## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term> |    √     |




## 功能说明

算子功能：对Token数据进行量化（可选），当存在TP域通信时，先进行EP（Expert Parallelism）域的AllToAllV通信，再进行TP（Tensor Parallelism）域的AllGatherV通信；当不存在TP域通信时，进行EP（Expert Parallelism）域的AllToAllV通信。

- 不存在TP域通信时：
$$
expandXOut = AllToAllV(X)\\
$$

- 存在TP域通信时：
$$
ataOut = AllToAllV(X)\\
expandXOut = AllGatherV(ataOut)\\
$$

注意该算子必须与`MoeDistributeCombine`配套使用。


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
<td>FLOAT16、BFLOAT16</td>
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
<td>scales</td>
<td>可选输入</td>
<td>每个专家的平滑权重、融合量化平滑权重的量化系数或量化系数。</td>
<td>FLOAT32</td>
<td>ND</td>
</tr>
<tr>
<td>xActiveMask</td>
<td>可选输入</td>
<td>预留参数，当前版本不支持，传空指针即可。</td>
<td>-</td>
<td>ND</td>
</tr>
<tr>
<td>expertScales</td>
<td>可选输入</td>
<td>每个Token的topK个专家权重。</td>
<td>FLOAT32</td>
<td>ND</td>
</tr>
<tr>
<td>groupEp</td>
<td>属性</td>
<td>EP通信域名称（专家并行通信域），字符串长度范围为[1, 128)，不能和groupTp相同。</td>
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
<td>EP域本卡Id，取值范围[0, epWorldSize)，同一个EP通信域中各卡的epRankId不重复。</td>
<td>INT64</td>
<td>ND</td>
</tr>
<tr>
<td>moeExpertNum</td>
<td>属性</td>
<td>MoE专家数量，取值范围(0, 512]，且满足moeExpertNum % (epWorldSize - sharedExpertRankNum) = 0。</td>
<td>INT64</td>
<td>ND</td>
</tr>
<tr>
<td>groupTp</td>
<td>可选属性</td>
<td><li>TP通信域名称（数据并行通信域）。</li><li>默认值为""。</li></td>
<td>STRING</td>
<td>ND</td>
</tr>
<tr>
<td>tpWorldSize</td>
<td>可选属性</td>
<td><li>TP通信域大小。</li><li>默认值为0。</li></td>
<td>INT64</td>
<td>ND</td>
</tr>
<tr>
<td>tpRankId</td>
<td>可选属性</td>
<td><li>TP域本卡Id。</li><li>默认值为0。</li></td>
<td>INT64</td>
<td>ND</td>
</tr>
<tr>
<td>expertShardType</td>
<td>可选属性</td>
<td><li>表示共享专家卡分布类型，当前仅支持传0，表示共享专家卡排在MoE专家卡前面，默认值为0。</li><li>默认值为0。</li></td>
<td>INT64</td>
<td>ND</td>
</tr>
<tr>
<td>sharedExpertNum</td>
<td>可选属性</td>
<td><li>表示共享专家数量（一个共享专家可复制部署到多个卡上），当前取值范围[0, 1]，0表示无共享专家，1表示一个共享专家，当前版本仅支持传1。</li><li>默认值为1。</li></td>
<td>INT64</td>
<td>ND</td>
</tr>
<tr>
<td>sharedExpertRankNum</td>
<td>可选属性</td>
<td><li>表示共享专家卡数量，当前取值范围[0, epWorldSize)，不为0时需满足epWorldSize % sharedExpertRankNum = 0。</li><li>默认值为0。</li></td>
<td>INT64</td>
<td>ND</td>
</tr>
<tr>
<td>quantMode</td>
<td>可选属性</td>
<td><li>表示量化模式，支持0：非量化，2：pertoken动态量化。</li><li>默认值为0。</li></td>
<td>INT64</td>
<td>ND</td>
</tr>
<tr>
<td>globalBs</td>
<td>可选属性</td>
<td><li>EP域全局的batch size大小，各rank Bs一致时，globalBs = Bs * epWorldSize 或 0；各rank Bs不一致时，globalBs = maxBs * epWorldSize（maxBs为单rank BS最大值）。</li><li>默认值为0。</li></td>
<td>INT64</td>
<td>ND</td>
</tr>
<tr>
<td>expertTokenNumsType</td>
<td>可选属性</td>
<td><li>输出expertTokenNums中值的语义类型，支持0：expertTokenNums中的输出为每个专家处理的token数的前缀和，1：expertTokenNums中的输出为每个专家处理的token数量。</li><li>默认值为0。</li></td>
<td>INT64</td>
<td>ND</td>
</tr>
<tr>
<td>expandX</td>
<td>输出</td>
<td>根据expertIds进行扩展过的token特征。</td>
<td>FLOAT16、BFLOAT16、INT8</td>
<td>ND</td>
</tr>
<tr>
<td>dynamicScales</td>
<td>输出</td>
<td>量化场景下，表示本卡输出Token的量化系数。</td>
<td>FLOAT32</td>
<td>ND</td>
</tr>
<tr>
<td>expandIdx</td>
<td>输出</td>
<td>表示给同一专家发送的token个数（对应算子MoeDistributeCombine中的expandIdx）。</td>
<td>INT32</td>
<td>ND</td>
</tr>
<tr>
<td>expertTokenNums</td>
<td>输出</td>
<td>表示每个专家收到的token个数。</td>
<td>INT64</td>
<td>ND</td>
</tr>
<tr>
<td>epRecvCounts</td>
<td>输出</td>
<td>从EP通信域各卡接收的token数（对应MoeDistributeCombine中的epSendCounts）。</td>
<td>INT32</td>
<td>ND</td>
</tr>
<tr>
<td>tpRecvCounts</td>
<td>输出</td>
<td>从TP通信域各卡接收的token数（对应MoeDistributeCombine中的tpSendCounts）；若有TP域通信则有该输出，若无TP域通信则无该输出。</td>
<td>INT32</td>
<td>ND</td>
</tr>
<tr>
<td>expandScales</td>
<td>输出</td>
<td>表示本卡输出Token的权重（对应MoeDistributeCombine中的expandScales）。</td>
<td>FLOAT32</td>
<td>ND</td>
</tr>
</tbody>
</table>

* <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>：
    * 不支持共享专家场景，不支持`expertShardType`、`sharedExpertNum`、`sharedExpertRankNum`属性。
    * 仅支持EP域，无TP域，不支持`groupTp`、`tpWorldSize`、`tpRankId`属性，`tpRecvCounts`为无效内容。
    * 仅设置环境变量`HCCL_INTRA_PCIE_ENABLE` = 1和`HCCL_INTRA_ROCE_ENABLE` = 0时，`expandScales`内容有效。

* <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：
    * 不支持`expandScales`。

## 约束说明

- `MoeDistributeDispatch`算子与`MoeDistributeCombine`算子必须配套使用，具体参考调用示例。

- 在不同产品型号、不同通信算法或不同版本中，`MoeDistributeDispatch`的Tensor输出`expandIdx`、`epRecvCounts`、`tpRecvCounts`、`expandScales`中的元素值可能不同，使用时直接将上述Tensor传给`MoeDistributeCombine`对应参数即可，模型其他业务逻辑不应对其存在依赖。

- 调用算子过程中使用的`groupEp`、`epWorldSize`、`moeExpertNum`、`groupTp`、`tpWorldSize`、`expertShardType`、`sharedExpertNum`、`sharedExpertRankNum`、`globalBs`属性取值所有卡需保持一致，网络中不同层中也需保持一致，且和`MoeDistributeCombine`对应参数也保持一致。

- 参数说明里shape格式说明：
    - `A`：表示本卡可能接收的最大token数量，取值范围如下：
        - 对于共享专家，要满足`A` = `BS` * `epWorldSize` * `sharedExpertNum` / `sharedExpertRankNum。`
        - 对于MoE专家，当`globalBs`为0时，要满足`A` >= `BS` * `epWorldSize` * min(`localExpertNum`, `K`)；当`globalBs`非0时，要满足`A` >= `globalBs` * min(`localExpertNum`, `K`)。
    - `localExpertNum`：表示本卡专家数量。
        - 对于共享专家卡，`localExpertNum` = 1
        - 对于MoE专家卡，`localExpertNum` = `moeExpertNum` / (`epWorldSize` - `sharedExpertRankNum`)`，localExpertNum` > 1时，不支持TP域通信。

- 本文公式中的"/"表示整除。

- 通信域使用约束：
    - 一个模型中的`MoeDistributeCombine`和`MoeDistributeDispatch`仅支持相同EP通信域，且该通信域中不允许有其他算子。
    - 一个模型中的`MoeDistributeCombine`和`MoeDistributeDispatch`仅支持相同TP通信域或都不支持TP通信域，有TP通信域时该通信域中不允许有其他算子。


- <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>：
    - 参数说明里shape格式说明：
        - `H`：表示hidden size隐藏层大小，取值范围(0, 7168]，且保证是32的整数倍。
        - `BS`：表示batch sequence size，即本卡最终输出的token数量，取值范围为[1, 256]。
        - `K`：表示选取topK个专家，需满足0 < `K` ≤ moeExpertNum，取值范围为[1, 16]。
    - `HCCL_BUFFSIZE`：调用本算子前需检查`HCCL_BUFFSIZE`环境变量取值是否合理，该环境变量表示单个通信域占用内存大小，单位MB，不配置时默认为200MB，要求 >= (`BS` * `epWorldSize` * min(`localExpertNum`, `K`) * `H` * 4B + 4MB)。
    - `HCCL_INTRA_PCIE_ENABLE`和`HCCL_INTRA_ROCE_ENABLE`：设置环境变量`HCCL_INTRA_PCIE_ENABLE` = 1和`HCCL_INTRA_ROCE_ENABLE` = 0可以减少跨机通信数据量，可能提升算子性能。 此时，要求`HCCL_BUFFSIZE` >= `moeExpertNum` * `BS` * (`H` * 2 + 16 * Align8(`K`))B + 104MB。并且，对于入参`moeExpertNum`，只要求`moeExpertNum` % `epWorldSize` = 0，不要求`moeExpertNum` / `epWorldSize` <= 24，但不支持`scales`特性，其中Align8(x) = ((x + 8 - 1) / 8) * 8。
    - `epWorldSize`：取值支持[16, 256]，且保证是8的整数倍。
        - `HCCL_INTRA_PCIE_ENABLE` = 1和`HCCL_INTRA_ROCE_ENABLE` = 0：最大值为64。
    - `quantMode`相关约束：
        - `quantMode`取值为2时，表示pertoken动态量化场景，`expandX`的数据类型支持`INT8`。
            - 输入`scales`可传入空指针。
            - 若输入`scales`传入有效数据时，其shape为 (`moeExpertNum`, `H`)。

- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：
    - 该场景下单卡包含双DIE（简称为“晶粒”或“裸片”），因此参数说明里的“本卡”均表示单DIE。
    - 参数说明里shape格式说明：
        - `H`：表示hidden size隐藏层大小，取值为7168。
        - `BS`：表示batch sequence size，即本卡最终输出的token数量，取值范围为[1, 512]。
        - `K`：表示选取topK个专家，需满足0 < `K` ≤ moeExpertNum，取值范围为[1, 8]。
    - `HCCL_BUFFSIZE`：调用本算子前需检查`HCCL_BUFFSIZE`环境变量取值是否合理，该环境变量表示单个通信域占用内存大小，单位MB，不配置时默认为200MB，要求 >= 2且满足1024 ^ 2 * (`HCCL_BUFFSIZE` - 2) / 2 >= `BS` * 2 * (`H` + 128) * (`epWorldSize` * `localExpertNum` + `K` + 1)，`localExpertNum`需使用MoE专家卡的本卡专家数。
    - `epWorldSize`：取值支持8、16、32、64、128、144、256、288。
    - `quantMode`相关约束：
        - `quantMode`取值为2时，表示pertoken动态量化场景，`expandX`的数据类型支持`INT8`。
            - 输入`scales`可传入空指针。
            - 若输入`scales`传入有效数据且存在共享专家卡时，其shape为 (`sharedExpertNum` + `moeExpertNum`, `H`)。
            - 若输入`scales`传入有效数据且不存在共享专家卡时，其shape为 (`moeExpertNum`, `H`)。

## 调用说明

| 调用方式  | 样例代码                                  | 说明                                                     |
| :--------: | :----------------------------------------: | :-------------------------------------------------------: |
| aclnn接口 | [test_aclnn_moe_distribute_dispatch.cpp](./examples/test_aclnn_moe_distribute_dispatch.cpp) | 通过[aclnnMoeDistributeDispatch](./docs/aclnnMoeDistributeDispatch.md)接口方式调用moe_distribute_dispatch算子。 |