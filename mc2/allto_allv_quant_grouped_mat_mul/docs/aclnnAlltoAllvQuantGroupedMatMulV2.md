# aclnnAlltoAllvQuantGroupedMatMulV2

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

- **算子功能**：完成路由专家 AlltoAllv、量化 GroupedMatMul 融合并实现与共享专家量化 MatMul 并行融合，**先通信后计算**。

- **计算公式**：
假设通信域中的总卡数为 epWorldSize，每张卡上通信后路由专家个数为 e，每张卡分组矩阵乘只负责本卡专家的计算。对于每张卡的计算公式如下：
  - 本卡共享专家分组矩阵乘计算

    $$
    mm\_y=(mm\_x × mm\_x\_scale) @ (mm\_weight × mm\_weight\_scale)
    $$

  - Alltoallv通信和permute

    $$
    permute\_out=Alltoallv(gmm\_x)
    $$

  - 本卡路由专家按专家维度分组矩阵乘计算

    $$
    gmm\_y=(permute\_out × gmm\_x\_scale) @ (gmm\_weight × gmm\_weight\_scale)
    $$

相较于`aclnnAlltoAllvQuantGroupedMatMul`接口，该接口变更如下：

- 新增`commMode`参数，用户根据该参数指定芯片使用的通信引擎。

  <!-- npu="950" id7 -->
  - <term>Ascend 950DT</term>：支持`ai_cpu`和`ccu`。

  <!-- end id7 -->

## 函数原型

每个算子分为两段式接口，必须先调用`aclnnAlltoAllvQuantGroupedMatMulV2GetWorkspaceSize`接口获取入参并根据计算流程计算所需workspace大小，再调用`aclnnAlltoAllvQuantGroupedMatMulV2`接口执行计算。

```cpp
aclnnStatus aclnnAlltoAllvQuantGroupedMatMulV2GetWorkspaceSize(
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
    int64_t            gmmXQuantMode,
    int64_t            gmmWeightQuantMode,
    int64_t            mmXQuantMode,
    int64_t            mmWeightQuantMode,
    const char*        group,
    const char*        commMode,
    int64_t            epWorldSize,
    const aclIntArray* sendCounts,
    const aclIntArray* recvCounts,
    bool               transGmmWeight,
    bool               transMmWeight,
    int64_t            groupSize,
    bool               permuteOutFlag,
    aclTensor*         gmmY,
    aclTensor*         mmYOptional,
    aclTensor*         permuteOutOptional,
    uint64_t*          workspaceSize,
    aclOpExecutor**    executor)
```

```cpp
aclnnStatus aclnnAlltoAllvQuantGroupedMatMulV2(
    void*          workspace,
    uint64_t       workspaceSize,
    aclOpExecutor* executor,
    aclrtStream    stream)
```

## aclnnAlltoAllvQuantGroupedMatMulV2GetWorkspaceSize

- **参数说明**

    <table style="undefined;table-layout: fixed; width: 1567px"><colgroup>
    <col style="width: 170px">
    <col style="width: 120px">
    <col style="width: 300px">
    <col style="width: 330px">
    <col style="width: 212px">
    <col style="width: 100px">
    <col style="width: 190px">
    <col style="width: 145px">
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
    </tr></thead>
    <tbody>
    <tr>
        <td>gmmX</td>
        <td>输入</td>
        <td>该输入进行 AlltoAllv 通信与 Permute 操作后结果作为 GroupedMatMul 计算的左矩阵。</td>
        <td>当前版本仅支持2维输入，shape为(BSK, H1)，且仅支持不转置场景。</td>
        <td>HIFLOAT8、FLOAT8_E4M3FN、FLOAT8_E5M2、FLOAT4_E2M1</td>
        <td>ND</td>
        <td>2</td>
        <td>x</td>
    </tr>
    <tr>
        <td>gmmWeight</td>
        <td>输入</td>
        <td>GroupedMatMul 计算的右矩阵。</td>
        <td>当前版本仅支持3维输入，不转置shape为(e, H1, N1)，转置shape为(e, N1, H1)。</td>
        <td>HIFLOAT8、FLOAT8_E4M3FN、FLOAT8_E5M2、FLOAT4_E2M1</td>
        <td>ND</td>
        <td>3</td>
        <td>√（仅适用转置场景）</td>
    </tr>
    <tr>
        <td>gmmXScale</td>
        <td>输入</td>
        <td>gmmX的量化系数。</td>
        <td><ul><li>pertensor量化场景支持1维，shape为(1)。
            </li><li>mx量化场景支持3维，shape为(BSK, ceildiv(H1, 64), 2)。
            </li></ul></td>
        <td>FLOAT32、FLOAT8_E8M0</td>
        <td>ND</td>
        <td>1、3</td>
        <td>x</td>
    </tr>
    <tr>
        <td>gmmWeightScale</td>
        <td>输入</td>
        <td>gmmWeight的量化系数。</td>
        <td><ul><li>pertensor量化场景支持1维，shape为(1)。
            </li><li>mx量化场景支持4维，不转置shape为(e, ceildiv(H1, 64), N1, 2)，转置shape为(e, N1, ceildiv(H1, 64), 2)。
            </li></ul></td>
        <td>FLOAT32、FLOAT8_E8M0</td>
        <td>ND</td>
        <td>1、4</td>
        <td>√（仅适用转置场景）</td>
    </tr>
    <tr>
        <td>sendCountsTensorOptional</td>
        <td>输入</td>
        <td>预留参数，当前版本仅支持传nullptr。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>recvCountsTensorOptional</td>
        <td>输入</td>
        <td>预留参数，当前版本仅支持传nullptr。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>mmXOptional</td>
        <td>输入</td>
        <td>可选输入，共享专家MatMul计算中的左矩阵。</td>
        <td><ul><li>支持2维，shape为(BS, H2)。
            </li><li>需与mmWeightOptional同时传入或同为nullptr。
            </li></ul></td>
        <td>与gmmX保持一致</td>
        <td>ND</td>
        <td>2</td>
        <td>x</td>
    </tr>
    <tr>
        <td>mmWeightOptional</td>
        <td>输入</td>
        <td>可选输入，共享专家MatMul计算中的右矩阵。</td>
        <td><ul><li>支持2维，不转置shape为(H2, N2)，转置shape为(N2, H2)。
            </li><li>需与mmXOptional同时传入或同为nullptr。
            </li></ul></td>
        <td>与gmmWeight保持一致</td>
        <td>ND</td>
        <td>2</td>
        <td>√（仅适用转置场景）</td>
    </tr>
    <tr>
        <td>mmXScaleOptional</td>
        <td>输入</td>
        <td>可选输入，mmX的量化系数。</td>
        <td><ul><li>pertensor量化场景支持1维，shape为(1)。
            </li><li>mx量化场景支持3维，shape为(BS, ceildiv(H2, 64), 2)。
            </li><li>需与mmXOptional同时传入或同为nullptr。
            </li></ul></td>
        <td>FLOAT32、FLOAT8_E8M0</td>
        <td>ND</td>
        <td>1、3</td>
        <td>x</td>
    </tr>
    <tr>
        <td>mmWeightScaleOptional</td>
        <td>输入</td>
        <td>可选输入，mmWeight的量化系数。</td>
        <td><ul><li>pertensor量化场景支持1维，shape为(1)。
            </li><li>mx量化场景支持3维，不转置shape为(ceildiv(H2, 64), N2, 2)，转置shape为(N2, ceildiv(H2, 64), 2)。
            </li><li>需与mmXOptional同时传入或同为nullptr。
            </li></ul></td>
        <td>FLOAT32、FLOAT8_E8M0</td>
        <td>ND</td>
        <td>1、3</td>
        <td>√（仅适用转置场景）</td>
    </tr>
    <tr>
        <td>gmmXQuantMode</td>
        <td>输入</td>
        <td>gmmX的量化模式。</td>
        <td>当前版本支持1(pertensor量化场景)、6(mx量化场景)。</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>gmmWeightQuantMode</td>
        <td>输入</td>
        <td>gmmWeight的量化模式。</td>
        <td>当前版本支持1(pertensor量化场景)、6(mx量化场景)。</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>mmXQuantMode</td>
        <td>输入</td>
        <td>mmX的量化模式。</td>
        <td>当前版本支持1(pertensor量化场景)、6(mx量化场景)。</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>mmWeightQuantMode</td>
        <td>输入</td>
        <td>mmWeight的量化模式。</td>
        <td>当前版本支持1(pertensor量化场景)、6(mx量化场景)。</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>group</td>
        <td>输入</td>
        <td>专家并行的通信域名，字符串长度要求 (0, 128)。</td>
        <td>通过Hccl提供的接口“extern HcclResult HcclGetCommName(HcclComm comm, char* commName);”获取，其中commName即为group。</td>
        <td>STRING</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>commMode</td>
        <td>输入</td>
        <td>指定当前通信类型。</td>
        <td>支持输入"ai_cpu"和"ccu"。</td>
        <td>STRING</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>epWorldSize</td>
        <td>输入</td>
        <td>ep通信域大小。</td>
        <td><term>Ascend 950DT</term>支持2、4、8、16、32、64、128、256。</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>sendCounts</td>
        <td>输入</td>
        <td>表示发送给其他卡的token数。</td>
        <td>数据类型支持INT64，长度为e * epWorldSize，最大为256。输入类型需为list。</td>
        <td>aclIntArray*（元素类型INT64）</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>recvCounts</td>
        <td>输入</td>
        <td>表示接收其他卡的token数。</td>
        <td>数据类型支持INT64，长度为e * epWorldSize，最大为256。输入类型需为list。</td>
        <td>aclIntArray*（元素类型INT64）</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>transGmmWeight</td>
        <td>输入</td>
        <td>GroupedMatMul的右矩阵是否需要转置。</td>
        <td>true表示需要转置，false表示不转置。</td>
        <td>BOOL</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>transMmWeight</td>
        <td>输入</td>
        <td>共享专家MatMul的右矩阵是否需要转置。</td>
        <td>true表示需要转置，false表示不转置。</td>
        <td>BOOL</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>groupSize</td>
        <td>输入</td>
        <td>用于表示量化中gmmXScale/gmmWeightScale/mmXScale/mmWeightScale输入的一个数在其所在的对应维度方向上可以用于该方向gmmX/gmmWeight/mmX/mmWeight输入的多少个数的量化。</td>
        <td>groupSize输入由3个方向的groupSizeM、groupSizeN、groupSizeK三个值拼接组成，每个值占16位，计算公式为groupSize = groupSizeK | groupSizeN << 16 | groupSizeM << 32。</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>permuteOutFlag</td>
        <td>输入</td>
        <td>permuteOutOptional是否需要输出。</td>
        <td>true表明需要输出，false表明不需要输出。</td>
        <td>BOOL</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>gmmY</td>
        <td>输出</td>
        <td>路由专家计算的输出。</td>
        <td>支持2维，shape为(A, N1)。</td>
        <td>FLOAT16、BFLOAT16</td>
        <td>ND</td>
        <td>2</td>
        <td>x</td>
    </tr>
    <tr>
        <td>mmYOptional</td>
        <td>输出</td>
        <td>共享专家计算的输出。</td>
        <td>支持2维，shape为(BS, N2)，仅当传入mmXOptional与mmWeightOptional才输出。</td>
        <td>与gmmY保持一致</td>
        <td>ND</td>
        <td>2</td>
        <td>x</td>
    </tr>
    <tr>
        <td>permuteOutOptional</td>
        <td>输出</td>
        <td>permute之后的输出。</td>
        <td>支持2维，shape为(A, H1)，仅当permuteOutFlag为true时输出。</td>
        <td>与gmmX保持一致</td>
        <td>ND</td>
        <td>2</td>
        <td>x</td>
    </tr>
    <tr>
        <td>workspaceSize</td>
        <td>输出</td>
        <td>返回需要在Device侧申请的workspace大小。</td>
        <td>-</td>
        <td>UINT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>executor</td>
        <td>输出</td>
        <td>返回op执行器，包含了算子的计算流程。</td>
        <td>-</td>
        <td>aclOpExecutor*</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    </tbody></table>

- gmmXQuantMode、gmmWeightQuantMode、mmXQuantMode、mmWeightQuantMode的枚举值跟量化模式关系如下:
  - 0: 非量化
  - 1: pertensor
  - 2: perchannel
  - 3: pertoken
  - 4: pergroup
  - 5: perblock
  - 6: mx量化
  - 7: pertoken动态量化

- **返回值**

    返回aclnnStatus状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。第一阶段接口完成入参校验，出现以下场景报错：

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
    </tr></thead>
    <tbody>
    <tr>
    <td>ACLNN_ERR_PARAM_NULLPTR</td>
    <td>161001</td>
    <td>传入参数要求是必选输入、输出或者必选属性，但实际传入了空指针。</td>
    </tr>
    <tr>
    <td>ACLNN_ERR_PARAM_INVALID</td>
    <td>161002</td>
    <td>gmmX、gmmWeight、sendCountsTensorOptional、recvCountsTensorOptional、mmXOptional、mmWeightOptional、group、epWorldSize、sendCounts、recvCounts的数据类型、数据格式或者维度不在支持的范围内。</td>
    </tr>
    </tbody></table>

## aclnnAlltoAllvQuantGroupedMatMulV2

- **参数说明**

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
    <td>在Device侧申请的workspace大小，由第一段接口<code>aclnnAlltoAllvQuantGroupedMatMulV2GetWorkspaceSize</code>获取。</td>
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

- 通信引擎约束：支持 CCU 通信和 AI_CPU 通信，CCU 仅支持单机 UB 域内互联，AI_CPU 可支持跨机 UB 域内互联。

- 确定性计算：
  - `aclnnAlltoAllvQuantGroupedMatMulV2`默认确定性实现。

- 参数说明里shape使用的变量：
  - BSK：本卡发送的token数，是sendCounts参数累加之和，取值范围(0, 52428800)。
  - H1：表示路由专家hidden size隐藏层大小，取值范围(0, 65536)。
  - H2：表示共享专家hidden size隐藏层大小，取值范围(0, 12288]。
  - e：表示单卡上专家个数，取值范围(0, 32]，e * epWorldSize最大支持256。
  - N1：表示路由专家的head_num，取值范围(0, 65536)。
  - N2：表示共享专家的head_num，取值范围(0, 65536)。
  - BS：batch sequence size。
  - K：表示选取TopK个专家，K的范围[2, 8]。
  - A：本卡收到的token数，是recvCounts参数累加之和。
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

    | gmmX | gmmWeight | gmmXScale | gmmWeightScale | mmXScale | mmWeightScale | gmmY |
    | :------: | :------: | :------: | :------: | :------: | :------: | :------: |
    | HIFLOAT8 | HIFLOAT8 | FLOAT32 | FLOAT32 | FLOAT32 | FLOAT32 | FLOAT16/BFLOAT16 |

  - mx量化

    | gmmX | gmmWeight | gmmXScale | gmmWeightScale | mmXScale | mmWeightScale | gmmY |
    | :------: | :------: | :------: | :------: | :------: | :------: | :------: |
    | FLOAT8_E4M3FN | FLOAT8_E4M3FN | FLOAT8_E8M0 | FLOAT8_E8M0 | FLOAT8_E8M0 | FLOAT8_E8M0 | FLOAT16/BFLOAT16 |
    | FLOAT8_E4M3FN | FLOAT8_E5M2 | FLOAT8_E8M0 | FLOAT8_E8M0 | FLOAT8_E8M0 | FLOAT8_E8M0 | FLOAT16/BFLOAT16 |
    | FLOAT8_E5M2 | FLOAT8_E4M3FN | FLOAT8_E8M0 | FLOAT8_E8M0 | FLOAT8_E8M0 | FLOAT8_E8M0 | FLOAT16/BFLOAT16 |
    | FLOAT8_E5M2 | FLOAT8_E5M2 | FLOAT8_E8M0 | FLOAT8_E8M0 | FLOAT8_E8M0 | FLOAT8_E8M0 | FLOAT16/BFLOAT16 |
    | FLOAT4_E2M1 | FLOAT4_E2M1 | FLOAT8_E8M0 | FLOAT8_E8M0 | FLOAT8_E8M0 | FLOAT8_E8M0 | FLOAT16/BFLOAT16 |

  - mmX类型与gmmX类型保持一致，mmWeight类型与gmmWeight类型保持一致，mmY类型与gmmY类型保持一致，permuteOut类型与gmmX保持一致。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考编译与运行样例。

注意：由于量化接口仅支持<term>Ascend 950DT</term>，以下示例基于该系列实现。本示例代码以2卡为例，请根据实际环境卡数修改 `EP_WORLD_SIZE`。

```cpp
#include <thread>
#include <iostream>
#include <string>
#include <vector>
#include "acl/acl.h"
#include "hccl/hccl.h"
#include "aclnnop/aclnn_allto_allv_quant_grouped_mat_mul_v2.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)           \
    do {                                  \
        printf(message, ##__VA_ARGS__);   \
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
constexpr int64_t EP_WORLD_SIZE = 2;
constexpr int64_t BS = 4096;
constexpr int64_t K = 2;
constexpr int64_t H = 7168;
constexpr int64_t e = 4;
constexpr int64_t N1 = 4096;
constexpr int64_t N2 = 4096;
constexpr int64_t A = BS * K;

int LaunchOneThreadAlltoAllvQuantGroupedMatMul(Args &args)
{
    int ret = aclrtSetCurrentContext(args.context);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetCurrentContext failed. ret: %d\n", ret); return ret);
    char hcomName[128] = {0};
    ret = HcclGetCommName(args.hcclComm, hcomName);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclGetCommName failed. ret: %d\n", ret); return -1);

    std::vector<int64_t> gmmXShape = {BS * K, H};
    std::vector<int64_t> gmmWShape = {e, H, N1};
    std::vector<int64_t> gmmYShape = {A, N1};
    std::vector<int64_t> permuteShape = {A, H};
    std::vector<int64_t> mmXShape = {BS, H};
    std::vector<int64_t> mmWShape = {H, N2};
    std::vector<int64_t> mmYShape = {BS, N2};
    std::vector<int64_t> scaleShape = {1}; // 缩放因子形状

    std::vector<int64_t> sendCountsList(EP_WORLD_SIZE * e, BS * K / (EP_WORLD_SIZE * e));
    std::vector<int64_t> recvCountsList(EP_WORLD_SIZE * e, BS * K / (EP_WORLD_SIZE * e));

    void *gmmXDeviceAddr = nullptr;
    void *gmmWDeviceAddr = nullptr;
    void *gmmYDeviceAddr = nullptr;
    void *permuteDeviceAddr = nullptr;
    void *mmXDeviceAddr = nullptr;
    void *mmWDeviceAddr = nullptr;
    void *mmYDeviceAddr = nullptr;
    void *gmmXScaleDeviceAddr = nullptr;
    void *gmmWScaleDeviceAddr = nullptr;
    void *mmXScaleDeviceAddr = nullptr;
    void *mmWScaleDeviceAddr = nullptr;

    aclTensor *gmmX = nullptr;
    aclTensor *gmmW = nullptr;
    aclTensor *gmmY = nullptr;
    aclTensor *mmX = nullptr;
    aclTensor *mmW = nullptr;
    aclTensor *mmY = nullptr;
    aclTensor *permute = nullptr;
    aclTensor *gmmXScale = nullptr;
    aclTensor *gmmWScale = nullptr;
    aclTensor *mmXScale = nullptr;
    aclTensor *mmWScale = nullptr;
    int64_t groupSize = 0;

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    void *workspaceAddr = nullptr;

    long long gmmXShapeSize = GetShapeSize(gmmXShape);
    long long gmmWShapeSize = GetShapeSize(gmmWShape);
    long long gmmYShapeSize = GetShapeSize(gmmYShape);
    long long permuteShapeSize = GetShapeSize(permuteShape);
    long long mmXShapeSize = GetShapeSize(mmXShape);
    long long mmWShapeSize = GetShapeSize(mmWShape);
    long long mmYShapeSize = GetShapeSize(mmYShape);

    // HIFLOAT8数据(使用uint8_t模拟)
    std::vector<uint8_t> gmmXHostData(gmmXShapeSize, (args.rankId + 1) * 10);
    std::vector<uint8_t> gmmWHostData(gmmWShapeSize, (args.rankId + 1) * 5);
    std::vector<uint8_t> mmXHostData(mmXShapeSize, (args.rankId + 1) * 10);
    std::vector<uint8_t> mmWHostData(mmWShapeSize, (args.rankId + 1) * 5);

    // 输出数据(FLOAT16/BFLOAT16)
    std::vector<uint16_t> gmmYHostData(gmmYShapeSize, 65535);
    std::vector<uint16_t> mmYHostData(mmYShapeSize, 0);
    std::vector<uint8_t> permuteHostData(permuteShapeSize, 255);

    // 缩放因子数据(FLOAT32)
    std::vector<float> gmmXScaleHostData(1, 1.0f);
    std::vector<float> gmmWScaleHostData(1, 1.0f);
    std::vector<float> mmXScaleHostData(1, 1.0f);
    std::vector<float> mmWScaleHostData(1, 1.0f);

    // 创建张量
    ret = CreateAclTensor(gmmXHostData, gmmXShape, &gmmXDeviceAddr, ACL_HIFLOAT8, &gmmX);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gmmWHostData, gmmWShape, &gmmWDeviceAddr, ACL_HIFLOAT8, &gmmW);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gmmYHostData, gmmYShape, &gmmYDeviceAddr, ACL_FLOAT16, &gmmY);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(mmXHostData, mmXShape, &mmXDeviceAddr, ACL_HIFLOAT8, &mmX);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(mmWHostData, mmWShape, &mmWDeviceAddr, ACL_HIFLOAT8, &mmW);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(mmYHostData, mmYShape, &mmYDeviceAddr, ACL_FLOAT16, &mmY);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(permuteHostData, permuteShape, &permuteDeviceAddr, ACL_HIFLOAT8, &permute);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gmmXScaleHostData, scaleShape, &gmmXScaleDeviceAddr, ACL_FLOAT, &gmmXScale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gmmWScaleHostData, scaleShape, &gmmWScaleDeviceAddr, ACL_FLOAT, &gmmWScale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(mmXScaleHostData, scaleShape, &mmXScaleDeviceAddr, ACL_FLOAT, &mmXScale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(mmWScaleHostData, scaleShape, &mmWScaleDeviceAddr, ACL_FLOAT, &mmWScale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    aclIntArray *sendCounts = aclCreateIntArray(sendCountsList.data(), sendCountsList.size());
    aclIntArray *recvCounts = aclCreateIntArray(recvCountsList.data(), recvCountsList.size());

    // 调用第一阶段接口
    ret = aclnnAlltoAllvQuantGroupedMatMulV2GetWorkspaceSize(gmmX,
        gmmW,
        gmmXScale,
        gmmWScale,
        nullptr, // sendCountsTensorOptional
        nullptr, // recvCountsTensorOptional
        mmX,
        mmW,
        mmXScale,
        mmWScale,
        1, // gmmXQuantMode
        1, // gmmWeightQuantMode
        1, // mmXQuantMode
        1, // mmWeightQuantMode
        hcomName,
        "ai_cpu",
        EP_WORLD_SIZE,
        sendCounts,
        recvCounts,
        false, // transGmmWeight
        false, // transMmWeight
        groupSize,
        true,  // permuteOutFlag
        gmmY,
        mmY,
        permute,
        &workspaceSize,
        &executor);
    CHECK_RET(
        ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclnnAlltoAllvQuantGroupedMatMulV2GetWorkspaceSize failed. ret = %d \n", ret);
        return ret);

    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc workspace failed. ret = %d \n", ret); return ret);
    }

    // 调用第二阶段接口
    ret = aclnnAlltoAllvQuantGroupedMatMulV2(workspaceAddr, workspaceSize, executor, args.stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclnnAlltoAllvQuantGroupedMatMulV2 failed. ret = %d \n", ret);
            return ret);

    // 同步等待任务执行结束
    ret = aclrtSynchronizeStreamWithTimeout(args.stream, 10000000);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSynchronizeStreamWithTimeout failed. ret = %d \n", ret);
            return ret);

    // 释放device资源
    if (gmmX != nullptr) aclDestroyTensor(gmmX);
    if (gmmW != nullptr) aclDestroyTensor(gmmW);
    if (gmmY != nullptr) aclDestroyTensor(gmmY);
    if (mmX != nullptr) aclDestroyTensor(mmX);
    if (mmW != nullptr) aclDestroyTensor(mmW);
    if (mmY != nullptr) aclDestroyTensor(mmY);
    if (permute != nullptr) aclDestroyTensor(permute);
    if (gmmXScale != nullptr) aclDestroyTensor(gmmXScale);
    if (gmmWScale != nullptr) aclDestroyTensor(gmmWScale);
    if (mmXScale != nullptr) aclDestroyTensor(mmXScale);
    if (mmWScale != nullptr) aclDestroyTensor(mmWScale);

    if (gmmXDeviceAddr != nullptr) aclrtFree(gmmXDeviceAddr);
    if (gmmWDeviceAddr != nullptr) aclrtFree(gmmWDeviceAddr);
    if (gmmYDeviceAddr != nullptr) aclrtFree(gmmYDeviceAddr);
    if (mmXDeviceAddr != nullptr) aclrtFree(mmXDeviceAddr);
    if (mmWDeviceAddr != nullptr) aclrtFree(mmWDeviceAddr);
    if (mmYDeviceAddr != nullptr) aclrtFree(mmYDeviceAddr);
    if (permuteDeviceAddr != nullptr) aclrtFree(permuteDeviceAddr);
    if (gmmXScaleDeviceAddr != nullptr) aclrtFree(gmmXScaleDeviceAddr);
    if (gmmWScaleDeviceAddr != nullptr) aclrtFree(gmmWScaleDeviceAddr);
    if (mmXScaleDeviceAddr != nullptr) aclrtFree(mmXScaleDeviceAddr);
    if (mmWScaleDeviceAddr != nullptr) aclrtFree(mmWScaleDeviceAddr);
    if (workspaceSize > 0) aclrtFree(workspaceAddr);

    HcclCommDestroy(args.hcclComm);
    aclrtDestroyStream(args.stream);
    aclrtDestroyContext(args.context);
    aclrtResetDevice(args.rankId);
    return 0;
}

int main(int argc, char *argv[])
{
    // 本样例基于Ascend 950DT实现
    int ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclInit failed. ret = %d \n", ret); return ret);

    constexpr uint32_t WORLD_SIZE = EP_WORLD_SIZE;
    aclrtStream stream[WORLD_SIZE];
    aclrtContext context[WORLD_SIZE];

    for (uint32_t rankId = 0; rankId < WORLD_SIZE; rankId++) {
        ret = aclrtSetDevice(rankId);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetDevice failed. ret = %d \n", ret); return ret);
        ret = aclrtCreateContext(&context[rankId], rankId);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateContext failed. ret = %d \n", ret); return ret);
        ret = aclrtCreateStream(&stream[rankId]);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateStream failed. ret = %d \n", ret); return ret);
    }

    int32_t devices[WORLD_SIZE];
    for (int i = 0; i < WORLD_SIZE; i++) {
        devices[i] = i;
    }

    // 初始化集合通信域
    HcclComm comms[WORLD_SIZE];
    ret = HcclCommInitAll(WORLD_SIZE, devices, comms);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclCommInitAll failed. ret = %d \n", ret); return ret);

    Args args[WORLD_SIZE];
    // 启动多线程
    std::vector<std::unique_ptr<std::thread>> threads(WORLD_SIZE);
    for (uint32_t rankId = 0; rankId < WORLD_SIZE; rankId++) {
        args[rankId].rankId = rankId;
        args[rankId].hcclComm = comms[rankId];
        args[rankId].stream = stream[rankId];
        args[rankId].context = context[rankId];
        threads[rankId].reset(new std::thread(&LaunchOneThreadAlltoAllvQuantGroupedMatMul, std::ref(args[rankId])));
    }

    for (uint32_t rankId = 0; rankId < WORLD_SIZE; rankId++) {
        threads[rankId]->join();
    }

    aclFinalize();
    return 0;
}

```
