# aclnnQuantGroupedMatMulAlltoAllvV2

[📄 查看源码](https://gitcode.com/cann/ops-transformer/tree/master/mc2/grouped_mat_mul_allto_allv)

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950DT</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：不支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：不支持
<!-- end id3 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2 推理产品</term>：不支持
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas 推理系列产品</term>：不支持
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas 训练系列产品</term>：不支持
<!-- end id6 -->

## 功能说明

- 接口功能：
    - 完成量化的路由专家GroupedMatMul、Unpermute、AlltoAllv融合并实现与共享专家MatMul并行融合，先计算后通信，支持Pertensor-Pertensor、Mx[量化模式](../../../docs/zh/context/quant_mode_introduction.md)。

- 计算公式：
    - 路由专家：
        $$
        gmmY = (gmmX @ gmmWeight) * gmmXScale * gmmWeightScale \\
        unpermuteOut = Unpermute(gmmY) \\
        y = AlltoAllv(unpermuteOut)
        $$

    - 共享专家：

        $$
        mmY = (mmX @  mmWeight) * mmXScaleOptional * mmWeightScaleOptional
        $$

相较于`aclnnQuantGroupedMatMulAlltoAllv`接口，该接口变更如下：

- 新增`commMode`参数，用户根据该参数指定芯片使用的通信引擎。

  <!-- npu="950" id7 -->
  - <term>Ascend 950DT</term>：支持`ai_cpu`和`ccu`。

  <!-- end id7 -->

## 函数原型

该算子分为两段式接口，必须先调用`aclnnQuantGroupedMatMulAlltoAllvV2GetWorkspaceSize`接口获取入参并根据计算流程计算所需workspace大小以及包含了算子计算流程的执行器，再调用`aclnnQuantGroupedMatMulAlltoAllvV2`接口执行计算。

```cpp
aclnnStatus aclnnQuantGroupedMatMulAlltoAllvV2GetWorkspaceSize(
    const aclTensor*   gmmX,
    const aclTensor*   gmmWeight,
    const aclTensor*   gmmXScale,
    const aclTensor*   gmmWeightScale,
    const aclTensor*   sendCountsTensorOptional,
    const aclTensor*   recvCountsTensorOptional,
    const aclTensor*   mmXOptional,
    const aclTensor*   mmWeightOptional,
    const aclTensor*   mmXScaleOptional,
    const aclTensor*   mmWeightScaleOptional,
    const aclTensor*   commQuantScaleOptional,
    int64_t            gmmXQuantMode,
    int64_t            gmmWeightQuantMode,
    int64_t            mmXQuantMode,
    int64_t            mmWeightQuantMode,
    int64_t            commQuantMode,
    int64_t            commQuantDtypeOptional,
    int64_t            groupSize,
    const char*        group,
    const char*        commMode,
    int64_t            epWorldSize,
    const aclIntArray* sendCounts,
    const aclIntArray* recvCounts,
    bool               transGmmWeight,
    bool               transMmWeight,
    const aclTensor*   y,
    const aclTensor*   mmYOptional,
    uint64_t*          workspaceSize,
    aclOpExecutor**    executor)
```

```cpp
aclnnStatus aclnnQuantGroupedMatMulAlltoAllvV2(
    void*           workspace,
    uint64_t        workspaceSize,
    aclOpExecutor*  executor,
    aclrtStream     stream)
```

## aclnnQuantGroupedMatMulAlltoAllvV2GetWorkspaceSize

- **参数说明**

    <table style="undefined;table-layout: fixed; width: 1200px">
        <colgroup>
            <col style="width: 188px">
            <col style="width: 85px">
            <col style="width: 220px">
            <col style="width: 280px">
            <col style="width: 200px">
            <col style="width: 90px">
            <col style="width: 100px">
            <col style="width: 80px">
        </colgroup>
        <thead>
            <tr>
                <th>参数名</th>
                <th>输入/输出</th>
                <th>描述</th>
                <th>使用说明</th>
                <th>数据类型</th>
                <th>数据格式</th>
                <th>维度(shape)</th>
                <th>非连续Tensor</th>
            </tr>
        </thead>
        <tbody>
            <!-- gmmX -->
            <tr>
                <td>gmmX</td>
                <td>输入</td>
                <td>公式中的输入gmmX。</td>
                <td>shape (A, H1)。</td>
                <td>HIFLOAT8、FLOAT8_E4M3FN、FLOAT8_E5M2、FLOAT4_E2M1</td>
                <td>ND</td>
                <td>2</td>
                <td>x</td>
            </tr>
            <!-- gmmWeight -->
            <tr>
                <td>gmmWeight</td>
                <td>输入</td>
                <td>公式中的输入gmmWeight。</td>
                <td>shape (e, H1, N1)。e为每卡部署的专家数，H1为hidden size，N1为路由专家FFN中间维度。</td>
                <td>HIFLOAT8、FLOAT8_E4M3FN、FLOAT8_E5M2、FLOAT4_E2M1</td>
                <td>ND</td>
                <td>3</td>
                <td>x</td>
            </tr>
            <!-- gmmXScale -->
            <tr>
                <td>gmmXScale</td>
                <td>输入</td>
                <td>gmmX的量化系数。</td>
                <td><li>pertensor量化：shape (1)。</li><li>mx量化：shape (A, ceil(H1/64), 2)</li></td>
                <td>FLOAT32、FLOAT8_E8M0</td>
                <td>ND</td>
                <td><li>pertensor量化：1</li><li>mx量化：3</li></td>
                <td>x</td>
            </tr>
            <!-- gmmWeightScale -->
            <tr>
                <td>gmmWeightScale</td>
                <td>输入</td>
                <td>gmmWeight的量化系数。</td>
                <td><li>pertensor量化：shape (1)。</li><li>mx量化：shape (e, N1, ceil(H1/64), 2)，weight转置时为(e, ceil(H1/64), N1, 2)</li></td>
                <td>FLOAT32、FLOAT8_E8M0</td>
                <td>ND</td>
                <td><li>pertensor量化：1</li><li>mx量化：3</li></td>
                <td>x</td>
            </tr>
            <!-- sendCountsTensorOptional -->
            <tr>
                <td>sendCountsTensorOptional</td>
                <td>输入</td>
                <td>AlltoAllv使用的send count。</td>
                <td>当前仅支持空。shape (e * ep, )。e为每卡部署的专家个数，ep为ep域大小。</td>
                <td>INT64</td>
                <td>ND</td>
                <td>1</td>
                <td>x</td>
            </tr>
            <!-- recvCountsTensorOptional -->
            <tr>
                <td>recvCountsTensorOptional</td>
                <td>输入</td>
                <td>AlltoAllv使用的recv count。</td>
                <td>当前仅支持空。shape (e * ep, )。e为每卡部署的专家个数，ep为ep域大小。</td>
                <td>INT64</td>
                <td>ND</td>
                <td>1</td>
                <td>x</td>
            </tr>
            <!-- mmXOptional -->
            <tr>
                <td>mmXOptional</td>
                <td>输入</td>
                <td>公式中的输入mmX。</td>
                <td>shape (bs, H2)。bs为每卡部署的专家个数，H2为hidden size。</td>
                <td>HIFLOAT8、FLOAT8_E4M3FN、FLOAT8_E5M2、FLOAT4_E2M1</td>
                <td>ND</td>
                <td>2</td>
                <td>x</td>
            </tr>
            <!-- mmWeightOptional -->
            <tr>
                <td>mmWeightOptional</td>
                <td>输入</td>
                <td>公式中的输入mmWeight。</td>
                <td>shape (H2, N2)。H2为hidden size，N2为共享专家FFN的中间层维度。</td>
                <td>HIFLOAT8、FLOAT8_E4M3FN、FLOAT8_E5M2、FLOAT4_E2M1</td>
                <td>ND</td>
                <td>2</td>
                <td>x</td>
            </tr>
            <!-- mmXScaleOptional -->
            <tr>
                <td>mmXScaleOptional</td>
                <td>输入</td>
                <td>mmX的量化系数。</td>
                <td><li>pertensor量化：shape (1)。</li><li>mx量化：shape (BS, ceil(H2/64), 2)</li></td>
                <td>FLOAT32、FLOAT8_E8M0</td>
                <td>ND</td>
                <td><li>pertensor量化：1</li><li>mx量化：3</li></td>
                <td>x</td>
            </tr>
            <!-- mmWeightScaleOptional -->
            <tr>
                <td>mmWeightScaleOptional</td>
                <td>输入</td>
                <td>mmWeight的量化系数。</td>
                <td><li>pertensor量化：shape(1)。</li><li>mx量化: shape (N2, ceil(H2/64), 2)，weight转置时为(ceil(H2/64), N2, 2)</li></td>
                <td>FLOAT32、FLOAT8_E8M0</td>
                <td>ND</td>
                <td><li>pertensor量化：1</li><li>mx量化：3</li></td>
                <td>x</td>
            </tr>
            <!-- commQuantScaleOptional -->
            <tr>
                <td>commQuantScaleOptional</td>
                <td>输入</td>
                <td>低比特通信量化系数。</td>
                <td>预留参数，当前仅支持空。</td>
                <td>FLOAT32</td>
                <td>ND</td>
                <td>1</td>
                <td>x</td>
            </tr>
            <!-- gmmXQuantMode -->
            <tr>
                <td>gmmXQuantMode</td>
                <td>输入</td>
                <td>gmmX的量化模式。</td>
                <td>必须传入量化模式，当前支持1 （pertensor量化）和6（mx量化）。</td>
                <td>INT64</td>
                <td>-</td>
                <td>1</td>
                <td>x</td>
            </tr>
            <!-- gmmWeightQuantMode -->
            <tr>
                <td>gmmWeightQuantMode</td>
                <td>输入</td>
                <td>gmmWeight的量化模式。</td>
                <td>必须传入量化模式，当前支持1 （pertensor量化）和6（mx量化）。</td>
                <td>INT64</td>
                <td>-</td>
                <td>1</td>
                <td>x</td>
            </tr>
            <!-- mmXQuantMode -->
            <tr>
                <td>mmXQuantMode</td>
                <td>输入</td>
                <td>mmX的量化模式。</td>
                <td>mmX非空，则必须传入量化模式，当前支持1 （pertensor量化）和6（mx量化）。</td>
                <td>INT64</td>
                <td>-</td>
                <td>1</td>
                <td>x</td>
            </tr>
            <!-- mmWeightQuantMode -->
            <tr>
                <td>mmWeightQuantMode</td>
                <td>输入</td>
                <td>mmWeight的量化模式。</td>
                <td>mmWeight不为空，则必须传入量化模式，当前支持1 （pertensor量化）和6（mx量化）。</td>
                <td>INT64</td>
                <td>-</td>
                <td>1</td>
                <td>x</td>
            </tr>
            <!-- commQuantMode -->
            <tr>
                <td>commQuantMode</td>
                <td>输入</td>
                <td>低比特通信量化模式。</td>
                <td>当前低比特功能预留，必须传入0，表示不量化。</td>
                <td>INT64</td>
                <td>-</td>
                <td>1</td>
                <td>x</td>
            </tr>
            <!-- commQuantDtypeOptional -->
            <tr>
                <td>commQuantDtypeOptional</td>
                <td>输入</td>
                <td>低比特通信的数据类型。</td>
                <td>当前低比特功能预留，必须传入 -1。</td>
                <td>INT64</td>
                <td>-</td>
                <td>1</td>
                <td>x</td>
            </tr>
            <!-- groupSize（新增）-->
            <tr>
                <td>groupSize</td>
                <td>输入</td>
                <td>PerGroup量化分组大小。</td>
                <td>用于Matmul计算三个方向上的量化分组大小，预留参数，仅支持配置为0，取值不生效。groupSize输入由3个方向的groupSizeM，groupSizeN，groupSizeK三个值拼接组成，每个值占16位，共占用int64_t类型groupSize的低48位（高16位无效），计算公式为：groupSize = groupSizeK | groupSizeN << 16 | groupSizeM << 32。</td>
                <td>INT64</td>
                <td>-</td>
                <td>-</td>
                <td>-</td>
            </tr>
            <!-- group -->
            <tr>
                <td>group</td>
                <td>输入</td>
                <td>通信域标识。</td>
                <td>字符串长度需大于0，小于128。</td>
                <td>char*</td>
                <td>-</td>
                <td>-</td>
                <td>-</td>
            </tr>
            <!-- commMode -->
            <tr>
                <td>commMode</td>
                <td>输入</td>
                <td>指定当前通信类型。</td>
                <td>支持输入"ai_cpu"和"ccu"。</td>
                <td>char*</td>
                <td>-</td>
                <td>-</td>
                <td>-</td>
            </tr>
            <!-- epWorldSize -->
            <tr>
                <td>epWorldSize</td>
                <td>输入</td>
                <td>通信域大小。</td>
                <td>支持2/4/8/16/32/64/128/256。</td>
                <td>INT64</td>
                <td>-</td>
                <td>-</td>
                <td>-</td>
            </tr>
            <!-- sendCounts -->
            <tr>
                <td>sendCounts</td>
                <td>输入</td>
                <td>AlltoAllv使用的send count。表示其他Rank向当前rank上各expert发送的token数量。</td>
                <td>支持的维度为e * ep。按<code>sendCounts[fromRank][expertId]</code>一维展开，例如e=3时顺序为<code>e0,e1,e2,e0,e1,e2, ...</code></td>
                <td>aclIntArray*（元素类型INT64）</td>
                <td>ND</td>
                <td>-</td>
                <td>-</td>
            </tr>
            <!-- recvCounts -->
            <tr>
                <td>recvCounts</td>
                <td>输入</td>
                <td>AlltoAllv使用的recv count。表示AlltoAllv后本卡需要接收到的token数量。</td>
                <td>支持的维度为e * ep。按<code>recvCounts[fromRank][expertId]</code>一维展开，例如e=3时顺序为<code>e0,e1,e2,e0,e1,e2, ...</code></td>
                <td>aclIntArray*（元素类型INT64）</td>
                <td>ND</td>
                <td>-</td>
                <td>-</td>
            </tr>
            <!-- transGmmWeight -->
            <tr>
                <td>transGmmWeight</td>
                <td>输入</td>
                <td>gmm的右矩阵是否转置。</td>
                <td>必须传入，无默认值。</td>
                <td>BOOL</td>
                <td>ND</td>
                <td>-</td>
                <td>-</td>
            </tr>
            <!-- transMmWeight -->
            <tr>
                <td>transMmWeight</td>
                <td>输入</td>
                <td>mm的右矩阵是否转置。</td>
                <td>必须传入，无默认值。</td>
                <td>BOOL</td>
                <td>ND</td>
                <td>-</td>
                <td>-</td>
            </tr>
            <!-- y -->
            <tr>
                <td>y</td>
                <td>输出</td>
                <td>grouped matmul计算输出。</td>
                <td>不支持空Tensor。shape (BSK, N1)。</td>
                <td>FLOAT16、BFLOAT16</td>
                <td>ND</td>
                <td>2</td>
                <td>x</td>
            </tr>
            <!-- mmYOptional -->
            <tr>
                <td>mmYOptional</td>
                <td>输出</td>
                <td>matmul计算输出。</td>
                <td>shape (bs, N1)。</td>
                <td>FLOAT16、BFLOAT16</td>
                <td>ND</td>
                <td>2</td>
                <td>x</td>
            </tr>
            <!-- workspaceSize -->
            <tr>
                <td>workspaceSize</td>
                <td>输出</td>
                <td>返回需要在Device侧申请的workspace大小。</td>
                <td>-</td>
                <td>UINT64</td>
                <td>ND</td>
                <td>-</td>
                <td>-</td>
            </tr>
            <!-- executor -->
            <tr>
                <td>executor</td>
                <td>输出</td>
                <td>返回op执行器，包含了算子计算流程。</td>
                <td>-</td>
                <td>aclOpExecutor*</td>
                <td>ND</td>
                <td>-</td>
                <td>-</td>
            </tr>
        </tbody>
    </table>

  gmmXQuantMode、gmmWeightQuantMode、mmXQuantMode、mmWeightQuantMode、commQuantMode的枚举值跟[量化模式](../../../docs/zh/context/quant_mode_introduction.md)关系如下:
  * 0: 非量化
  * 1: pertensor
  * 2: perchannel
  * 3: pertoken
  * 4: pergroup
  * 5: perblock
  * 6: mx量化
  * 7: pertoken动态量化

- **返回值**

    返回aclnnStatus状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

    第一阶段接口完成入参校验，出现以下场景报错：

    <table style="undefined;table-layout: fixed; width: 1149px"><colgroup>
    <col style="width: 282px">
    <col style="width: 120px">
    <col style="width: 747px">
    </colgroup>
    <thead>
    <tr>
    <th>返回值</th>
    <th>错误码</th>
    <th>描述</th>
    </tr></thead>
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
    </tbody></table>

## aclnnQuantGroupedMatMulAlltoAllvV2

- **参数说明**

    <table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
    <col style="width: 168px">
    <col style="width: 128px">
    <col style="width: 854px">
    </colgroup>
    <thead>
    <tr>
    <th>参数名</th>
    <th>输入/输出</th>
    <th>描述</th>
    </tr></thead>
    <tbody>
    <tr>
    <td>workspace</td>
    <td>输入</td>
    <td>在Device侧申请的workspace内存地址。</td>
    </tr>
    <tr>
    <td>workspaceSize</td>
    <td>输入</td>
    <td>在Device侧申请的workspace大小，由第一段接口<code>aclnnQuantGroupedMatMulAlltoAllvV2GetWorkspaceSize</code>获取。</td>
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
    </tbody></table>

- **返回值**

    返回aclnnStatus状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

## 约束说明

- 确定性计算：
  - aclnnQuantGroupedMatMulAlltoAllvV2默认确定性实现。
- 通信引擎约束：

  <!-- npu="950" id8 -->
  - Ascend 950DT：支持CCU通信和AI_CPU通信，CCU仅支持单机UB域内互联，AI_CPU可支持跨机UB域内互联。
  <!-- end id8 -->

- e * epWorldSize最大支持256，e表示单卡上的专家数量，最大支持到32，epWorldSize支持2/4/8/16/32/64/128/256;
- gmmX的shape(A, H1)，A为sendCounts之和，H1取值范围(0, 65536);
- gmmWeight的shape(e, H1, N1)，N1取值范围(0, 65536);
- y的shape为(BSK, N1)，第一维其中K的范围[2, 8]，BSK为recvCounts之和;
- mmX是共享专家的左矩阵，shape为(BS, H2)，H2的取值范围(0, 12288]；
- mmWeight是共享专家的右矩阵，shape为(H2， N2)，N2的取值范围(0, 65536)；
- sendCounts为发送到其他卡的token数，数组大小为e * epWorldSize;
- recvCounts为从其他卡接收到的token数，数组大小为e * epWorldSize;
- 路由专家和共享专家量化Scale、Mode等均为必选；
- 低比特通信Mode为必选参数，DType和Scale为可选，当Mode为非0时需要提供DType和Scale；
- 参数说明里shape使用的变量：
  - BSK：本卡接收的token数，是recvCounts参数累加之和，取值范围(0, 52428800)。
  - H1：表示路由专家hidden size隐藏层大小，取值范围(0, 65536)。
  - H2：表示共享专家hidden size隐藏层大小，取值范围(0, 12288]。
  - e：表示单卡上专家个数，0<e<=32，e * epWorldSize最大支持256。
  - N1：表示路由专家FFN的中间层维度，取值范围(0, 65536)。
  - N2：表示共享专家FFN的中间层维度，取值范围(0, 65536)。
  - BS：batch sequence size。
  - K：表示选取TopK个专家，K的范围[2, 8]。
  - A：本卡发送的token数，是sendCounts参数累加之和。
  - ep通信域内所有卡的A参数的累加和等于所有卡上的BSK参数的累加和。
  - mx量化且gmmX与gmmWeight为FLOAT4_E2M1时，H1和H2必须为偶数且不能为2，同时transGmmWeight和transMmWeight为false情况下，N1和N2必须为偶数。
  - gmmWeight和gmmWeightScale的转置状态必须保持一致：同时转置或同时不转置。mmWeight和mmWeightScale同样需要保持转置状态一致。
  - groupSize:
    - 仅当gmmXScale/gmmWeightScale/mmXScale/mmWeightScale输入都是2维及以上数据时，groupSize取值有效，其他场景需传入0。
    - groupSize值支持公式推导：传入的groupSize内部会按如下公式分解得到groupSizeM、groupSizeN、groupSizeK，当其中有1个或多个为0，会根据gmmX/gmmWeight/mmX/mmWeight/gmmXScale/gmmWeightScale/mmXScale/mmWeightScale输入shape重新设置groupSizeM、groupSizeN、groupSizeK用于计算。设置原理：如果groupSizeM=0，表示m方向量化分组值由接口推导，推导公式为groupSizeM = m / scaleM（需保证m能被scaleM整除），其中m与gmmX/mmX shape中的m一致，scaleM与gmmXScale/mmXScale shape中的m一致；如果groupSizeK=0，表示k方向量化分组值由接口推导，推导公式为groupSizeK = k / scaleK（需保证k能被scaleK整除），其中k与gmmX/mmX shape中的k一致，scaleK与gmmXScale/mmXScale shape中的k一致；如果groupSizeN=0，表示n方向量化分组值由接口推导，推导公式为groupSizeN = n / scaleN（需保证n能被scaleN整除），其中n与gmmWeight/mmWeight shape中的n一致，scaleN与gmmWeightScale/mmWeightScale shape中的n一致。
    $$
    groupSize = groupSizeK | groupSizeN << 16 | groupSizeM << 32
    $$
    - 如果满足重新设置条件，当gmmXScale/gmmWeightScale/mmXScale/mmWeightScale输入是2维及以上时，且数据类型都为FLOAT8_E8M0时，[groupSizeM，groupSizeN，groupSizeK]取值组合会推导为[1, 1, 32]，对应groupSize的值为4295032864。

- 量化参数约束：
  - 当前版本支持pertensor量化、mx量化。
- 类型约束
  - pertensor量化

    | gmmX | gmmWeight | gmmXScale | gmmWeightScale | mmXScale | mmWeightScale | y |
    | :------: | :------: | :------: | :------: | :------: | :------: | :------: |
    | HIFLOAT8 | HIFLOAT8 | FLOAT32 | FLOAT32 | FLOAT32 | FLOAT32 | FLOAT16/BFLOAT16 |

  - mx量化

    | gmmX | gmmWeight | gmmXScale | gmmWeightScale | mmXScale | mmWeightScale | y |
    | :------: | :------: | :------: | :------: | :------: | :------: | :------: |
    | FLOAT8_E4M3FN | FLOAT8_E4M3FN | FLOAT8_E8M0 | FLOAT8_E8M0 | FLOAT8_E8M0 | FLOAT8_E8M0 | FLOAT16/BFLOAT16 |
    | FLOAT8_E4M3FN | FLOAT8_E5M2 | FLOAT8_E8M0 | FLOAT8_E8M0 | FLOAT8_E8M0 | FLOAT8_E8M0 | FLOAT16/BFLOAT16 |
    | FLOAT8_E5M2 | FLOAT8_E4M3FN | FLOAT8_E8M0 | FLOAT8_E8M0 | FLOAT8_E8M0 | FLOAT8_E8M0 | FLOAT16/BFLOAT16 |
    | FLOAT8_E5M2 | FLOAT8_E5M2 | FLOAT8_E8M0 | FLOAT8_E8M0 | FLOAT8_E8M0 | FLOAT8_E8M0 | FLOAT16/BFLOAT16 |
    | FLOAT4_E2M1 | FLOAT4_E2M1 | FLOAT8_E8M0 | FLOAT8_E8M0 | FLOAT8_E8M0 | FLOAT8_E8M0 | FLOAT16/BFLOAT16 |

  - mmX类型与gmmX类型保持一致，mmWeight类型与gmmWeight类型保持一致，mmY类型与y类型保持一致。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考编译与运行样例。

说明：本示例代码调用了部分HCCL集合通信库接口：HcclGetCommName、HcclCommInitAll、HcclCommDestroy,请参考[<<HCCL API (C)>>](https://hiascend.com/document/redirect/CannCommunityHcclCppApi)。

<!-- npu="950" id9 -->
- <term>Ascend 950DT</term>：

    ```Cpp
    #include <thread>
    #include <iostream>
    #include <string>
    #include <vector>
    #include "acl/acl.h"
    #include "hccl/hccl.h"
    #include "aclnnop/aclnn_quant_grouped_mat_mul_allto_allv_v2.h"

    #define CHECK_RET(cond, return_expr) \
        do {                             \
            if (!(cond)) {               \
                return_expr;             \
            }                            \
        } while (0)

    #define LOG_PRINT(message, ...)          \
        do {                                 \
            printf(message, ##__VA_ARGS__);  \
        } while (0)

    int64_t GetShapeSize(const std::vector<int64_t> &shape)
    {
        int64_t shape_size = 1;
        for (auto i : shape) {
            shape_size *= i;
        }
        return shape_size;
    }

    template <typename T>
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
            strides[i] = shape[i + 1] * strides[i + 1];
        }
        *tensor = aclCreateTensor(shape.data(),
            shape.size(),
            dataType,
            strides.data(),
            0,
            aclFormat::ACL_FORMAT_ND,
            shape.data(),
            shape.size(),
            *deviceAddr);
        return 0;
    }

    struct Args {
        int rankId;
        HcclComm hcclComm;
        aclrtStream stream;
        aclrtContext context;
    };

    // shape基本信息
    constexpr int64_t EP_WORLD_SIZE = 8;
    constexpr int64_t BS = 4096;
    constexpr int64_t K = 2;
    constexpr int64_t H = 7168;
    constexpr int64_t e = 4;
    constexpr int64_t N1 = 4096;
    constexpr int64_t N2 = 4096;
    constexpr int64_t A = BS * K;

    std::vector<int16_t> pGmmyData(BS *K *N1, 0);
    std::vector<int16_t> pmmYData(BS *N2, 0);

    int LaunchOneThreadAlltoAllvGmm(Args &args)
    {
        int ret = aclrtSetCurrentContext(args.context);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetCurrentContext failed. ret: %d\n", ret); return ret);
        char hcomName[128] = {0};
        ret = HcclGetCommName(args.hcclComm, hcomName);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclGetEpCommName failed. ret: %d\n", ret); return -1);

        std::vector<int64_t> gmmXShape = {A, H};
        std::vector<int64_t> gmmWShape = {e, H, N1};
        std::vector<int64_t> gmmXScaleShape = {1};
        std::vector<int64_t> gmmWScaleShape = {1};
        std::vector<int64_t> yShape = {BS * K, N1};
        std::vector<int64_t> mmXShape = {BS, H};
        std::vector<int64_t> mmWShape = {H, N2};
        std::vector<int64_t> mmXScaleShape = {1};
        std::vector<int64_t> mmWScaleShape = {1};
        std::vector<int64_t> mmYShape = {BS, N2};

        std::vector<int64_t> sendCountsShape = {EP_WORLD_SIZE * e};
        std::vector<int64_t> recvCountsShape = {EP_WORLD_SIZE * e};

        std::vector<int64_t> sendCountsList(EP_WORLD_SIZE * e, A / (EP_WORLD_SIZE * e));
        std::vector<int64_t> recvCountsList(EP_WORLD_SIZE * e, A / (EP_WORLD_SIZE * e));

        void *gmmXDeviceAddr = nullptr;
        void *gmmWDeviceAddr = nullptr;
        void *gmmXScaleDeviceAddr = nullptr;
        void *gmmWScaleDeviceAddr = nullptr;
        void *yDeviceAddr = nullptr;
        void *mmXDeviceAddr = nullptr;
        void *mmWDeviceAddr = nullptr;
        void *mmXScaleDeviceAddr = nullptr;
        void *mmWScaleDeviceAddr = nullptr;
        void *mmYDeviceAddr = nullptr;

        aclTensor *gmmX = nullptr;
        aclTensor *gmmW = nullptr;
        aclTensor *gmmXScale = nullptr;
        aclTensor *gmmWScale = nullptr;
        aclTensor *y = nullptr;

        aclTensor *mmX = nullptr;
        aclTensor *mmW = nullptr;
        aclTensor *mmXScale = nullptr;
        aclTensor *mmWScale = nullptr;
        aclTensor *mmY = nullptr;

        aclTensor *sendCountsTensor = nullptr;
        aclTensor *recvCountsTensor = nullptr;
        aclTensor *commQuantScaleOptional = nullptr;

        int64_t gmmXQuantMode = 1;
        int64_t gmmWQuantMode = 1;
        int64_t mmXQuantMode = 1;
        int64_t mmWQuantMode = 1;
        int64_t commQuantMode = 0;
        int64_t commQuantDtypeOptional = -1;
        int64_t groupSize = 0;

        uint64_t workspaceSize = 0;
        aclOpExecutor *executor = nullptr;
        void *workspaceAddr = nullptr;

        long long gmmXShapeSize = GetShapeSize(gmmXShape);
        long long gmmWShapeSize = GetShapeSize(gmmWShape);
        long long gmmXScaleShapeSize = GetShapeSize(gmmXScaleShape);
        long long gmmWScaleShapeSize = GetShapeSize(gmmWScaleShape);
        long long yShapeSize = GetShapeSize(yShape);

        long long mmXShapeSize = GetShapeSize(mmXShape);
        long long mmWShapeSize = GetShapeSize(mmWShape);
        long long mmXScaleShapeSize = GetShapeSize(mmXScaleShape);
        long long mmWScaleShapeSize = GetShapeSize(mmWScaleShape);
        long long mmYShapeSize = GetShapeSize(mmYShape);

        std::vector<uint8_t> gmmXHostData(gmmXShapeSize, (args.rankId + 1) * 1024);  // HIFLOAT8
        std::vector<uint8_t> gmmWHostData(gmmWShapeSize, (args.rankId + 1) * 512);
        std::vector<float> gmmXScaleHostData(gmmXScaleShapeSize, 1);
        std::vector<float> gmmWScaleHostData(gmmWScaleShapeSize, 1);
        std::vector<int16_t> yHostData(yShapeSize, 65535);

        std::vector<uint8_t> mmXHostData(mmXShapeSize, (args.rankId + 1) * 1024);  // HIFLOAT8
        std::vector<uint8_t> mmWHostData(mmWShapeSize, (args.rankId + 1) * 512);
        std::vector<float> mmXScaleHostData(mmXScaleShapeSize, 1);
        std::vector<float> mmWScaleHostData(mmWScaleShapeSize, 1);
        std::vector<int16_t> mmYHostData(mmYShapeSize, 0);

        ret = CreateAclTensor(gmmXHostData, gmmXShape, &gmmXDeviceAddr, aclDataType::ACL_HIFLOAT8, &gmmX);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(gmmWHostData, gmmWShape, &gmmWDeviceAddr, aclDataType::ACL_HIFLOAT8, &gmmW);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(gmmXScaleHostData, gmmXScaleShape, &gmmXScaleDeviceAddr, aclDataType::ACL_FLOAT, &gmmXScale);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(gmmWScaleHostData, gmmWScaleShape, &gmmWScaleDeviceAddr, aclDataType::ACL_FLOAT, &gmmWScale);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(yHostData, yShape, &yDeviceAddr, aclDataType::ACL_FLOAT16, &y);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(mmXHostData, mmXShape, &mmXDeviceAddr, aclDataType::ACL_HIFLOAT8, &mmX);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(mmWHostData, mmWShape, &mmWDeviceAddr, aclDataType::ACL_HIFLOAT8, &mmW);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(mmXScaleHostData, mmXScaleShape, &mmXScaleDeviceAddr, aclDataType::ACL_FLOAT, &mmXScale);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(mmWScaleHostData, mmWScaleShape, &mmWScaleDeviceAddr, aclDataType::ACL_FLOAT, &mmWScale);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(mmYHostData, mmYShape, &mmYDeviceAddr, aclDataType::ACL_FLOAT16, &mmY);
        CHECK_RET(ret == ACL_SUCCESS, return ret);

        aclIntArray *sendCounts = aclCreateIntArray(sendCountsList.data(), sendCountsList.size());
        aclIntArray *recvCounts = aclCreateIntArray(recvCountsList.data(), recvCountsList.size());

        // 调用第一阶段接口
        ret = aclnnQuantGroupedMatMulAlltoAllvV2GetWorkspaceSize(
            gmmX,
            gmmW,
            gmmXScale,
            gmmWScale,
            sendCountsTensor,
            recvCountsTensor,
            mmX,
            mmW,
            mmXScale,
            mmWScale,
            commQuantScaleOptional,
            gmmXQuantMode,
            gmmWQuantMode,
            mmXQuantMode,
            mmWQuantMode,
            commQuantMode,
            commQuantDtypeOptional,
            groupSize,
            hcomName,
            "ai_cpu",
            EP_WORLD_SIZE,
            sendCounts,
            recvCounts,
            false,
            false,
            y,
            mmY,
            &workspaceSize,
            &executor);
        CHECK_RET(
            ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclnnQuantGroupedMatMulAlltoAllvV2GetWorkspaceSize failed. ret = %d \n", ret);
            return ret);

        if (workspaceSize > 0) {
            ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc workspace failed. ret = %d \n", ret); return ret);
        }
        // 调用第二阶段接口
        ret = aclnnQuantGroupedMatMulAlltoAllvV2(workspaceAddr, workspaceSize, executor, args.stream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclnnQuantGroupedMatMulAlltoAllvV2 failed. ret = %d \n", ret);
                return ret);
        //（固定写法）同步等待任务执行结束
        ret = aclrtSynchronizeStreamWithTimeout(args.stream, 10000000);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSynchronizeStreamWithTimeout failed. ret = %d \n", ret);
                return ret);
        LOG_PRINT("[INFO] device_%d aclnnQuantGroupedMatMulAlltoAllvV2 execute successfully.\n", args.rankId);
        // 释放device资源，需要根据具体API的接口定义修改
        if (args.rankId == 0) {
            size_t size = A * N1 * sizeof(int16_t);
            aclrtMemcpy(pGmmyData.data(), size, yDeviceAddr, size, ACL_MEMCPY_DEVICE_TO_HOST);
        }
        if (gmmX != nullptr) {
            aclDestroyTensor(gmmX);
        }
        if (gmmW != nullptr) {
            aclDestroyTensor(gmmW);
        }
        if (gmmXScale != nullptr) {
            aclDestroyTensor(gmmXScale);
        }
        if (gmmWScale != nullptr) {
            aclDestroyTensor(gmmWScale);
        }
        if (y != nullptr) {
            aclDestroyTensor(y);
        }
        if (mmX != nullptr) {
            aclDestroyTensor(mmX);
        }
        if (mmW != nullptr) {
            aclDestroyTensor(mmW);
        }
        if (mmXScale != nullptr) {
            aclDestroyTensor(mmXScale);
        }
        if (mmWScale != nullptr) {
            aclDestroyTensor(mmWScale);
        }
        if (mmY != nullptr) {
            aclDestroyTensor(mmY);
        }
        if (gmmXDeviceAddr != nullptr) {
            aclrtFree(gmmXDeviceAddr);
        }
        if (gmmWDeviceAddr != nullptr) {
            aclrtFree(gmmWDeviceAddr);
        }
        if (gmmXScaleDeviceAddr != nullptr) {
            aclrtFree(gmmXScaleDeviceAddr);
        }
        if (gmmWScaleDeviceAddr != nullptr) {
            aclrtFree(gmmWScaleDeviceAddr);
        }
        if (yDeviceAddr != nullptr) {
            aclrtFree(yDeviceAddr);
        }
        if (mmXDeviceAddr != nullptr) {
            aclrtFree(mmXDeviceAddr);
        }
        if (mmWDeviceAddr != nullptr) {
            aclrtFree(mmWDeviceAddr);
        }
        if (mmXScaleDeviceAddr != nullptr) {
            aclrtFree(mmXScaleDeviceAddr);
        }
        if (mmWScaleDeviceAddr != nullptr) {
            aclrtFree(mmWScaleDeviceAddr);
        }
        if (mmYDeviceAddr != nullptr) {
            aclrtFree(mmYDeviceAddr);
        }
        if (workspaceSize > 0) {
            aclrtFree(workspaceAddr);
        }
        HcclCommDestroy(args.hcclComm);
        aclrtDestroyStream(args.stream);
        aclrtDestroyContext(args.context);
        aclrtResetDevice(args.rankId);
        return 0;
    }

    int main(int argc, char *argv[])
    {
        int ret = aclInit(nullptr);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclInit failed. ret = %d \n", ret); return ret);
        aclrtStream stream[EP_WORLD_SIZE];
        aclrtContext context[EP_WORLD_SIZE];
        for (uint32_t rankId = 0; rankId < EP_WORLD_SIZE; rankId++) {
            ret = aclrtSetDevice(rankId);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetDevice failed. ret = %d \n", ret); return ret);
            ret = aclrtCreateContext(&context[rankId], rankId);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateContext failed. ret = %d \n", ret); return ret);
            ret = aclrtCreateStream(&stream[rankId]);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateStream failed. ret = %d \n", ret); return ret);
        }

        int32_t devices[EP_WORLD_SIZE];
        for (int i = 0; i < EP_WORLD_SIZE; i++) {
            devices[i] = i;
        }
        //初始化集合通信域
        HcclComm comms[EP_WORLD_SIZE];
        ret = HcclCommInitAll(EP_WORLD_SIZE, devices, comms);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclCommInitAll failed. ret = %d \n", ret); return ret);

        Args args[EP_WORLD_SIZE];
        // 启动多线程
        std::vector<std::unique_ptr<std::thread>> threads(EP_WORLD_SIZE);
        for (uint32_t rankId = 0; rankId < EP_WORLD_SIZE; rankId++) {
            args[rankId].rankId = rankId;
            args[rankId].hcclComm = comms[rankId];
            args[rankId].stream = stream[rankId];
            args[rankId].context = context[rankId];
            threads[rankId].reset(new std::thread(&LaunchOneThreadAlltoAllvGmm, std::ref(args[rankId])));
        }
        for (uint32_t rankId = 0; rankId < EP_WORLD_SIZE; rankId++) {
            threads[rankId]->join();
        }
        aclFinalize();
        return 0;
    }
    ```

<!-- end id9 -->
