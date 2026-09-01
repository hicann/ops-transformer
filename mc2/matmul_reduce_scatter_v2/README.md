# MatmulReduceScatterV2

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------:|
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>       |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                               |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×     |

## 功能说明

- **接口功能**:
    aclnnMatmulReduceScatterV2接口是对aclnnMatmulReduceScatter接口的功能扩展，在支持x1和x2输入类型为FLOAT16/BFLOAT16的基础上，
    - <term>Ascend 950PR/Ascend 950DT</term>：
        - 新增了对低精度数据类型FLOAT8_E4M3FN/FLOAT8_E5M2/HIFLOAT8/FLOAT4_E2M1的支持。支持pertensor、perblock、mx[量化方式](../../docs/zh/context/quant_mode_introduction.md)。
    - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：
        - 新增了对低精度数据类型INT8的支持。支持pertoken/perchannel[量化方式](../../docs/zh/context/quant_mode_introduction.md)。

- **计算公式**：
    - 情形1：如果x1和x2数据类型为FLOAT16/BFLOAT16时，对入参x1、x2、bias进行matmul计算后，进行ReduceScatter通信。

        $$
        output=ReduceScatter(x1@x2 + bias_{optional})
        $$

    - 情形2：如果x1和x2数据类型为FLOAT8_E4M3FN/FLOAT8_E5M2/HIFLOAT8的pertensor场景，或者x1和x2数据类型为INT8的perchannel、pertoken场景，且不输出amaxOut，入参x1、x2进行matmul计算和dequant计算后，进行ReduceScatter通信。

        $$
        output=ReduceScatter((x1Scale*x2Scale)*(x1@x2) + bias_{optional})
        $$

    - 情形3：如果x1和x2数据类型为FLOAT8_E4M3FN/FLOAT8_E5M2/HIFLOAT8的perblock场景，且不输出amaxOut，当x1的shape为(m, k)、x2的shape为(k, n)时， x1Scale的shape为(ceildiv(m, 128), ceildiv(k, 128))、x2Scale的shape为(ceildiv(k, 128), ceildiv(n, 128))时，入参x1、x2进行matmul计算和dequant计算后，再进行ReduceScatter通信。

        $$
        output=ReduceScatter(\sum_{0}^{\left \lfloor \frac{k}{blockSize=128} \right \rfloor} (x1_{pr}@x2_{rq}*(x1Scale_{pr}*x2Scale_{rq})))
        $$

    - 情形4：如果x1和x2数据类型为FLOAT8_E4M3FN/FLOAT8_E5M2/FLOAT4_E2M1的mx量化场景，且不输出amaxOut，当x1的shape为(m, k)、x2的shape为(n, k)时， x1Scale的shape为(m, ceildiv(k, 64), 2)、x2Scale的shape为(n, ceildiv(k, 64), 2)时，入参x1、x2进行matmul计算和dequant计算后，再进行ReduceScatter通信。mx量化仅支持x2、x2Scale转置场景。

        $$
        output=ReduceScatter(\sum_{0}^{\left \lfloor \frac{k}{blockSize=32} \right \rfloor} (x1_{pr}@x2_{rq}*(x1Scale_{pr}*x2Scale_{rq})))
        $$

## 函数原型

每个算子分为两段式接口，必须先调用`aclnnMatmulReduceScatterV2GetWorkspaceSize`接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用`aclnnMatmulReduceScatterV2`接口执行计算。

```cpp
aclnnStatus aclnnMatmulReduceScatterV2GetWorkspaceSize(
    const aclTensor* x1,
    const aclTensor* x2,
    const aclTensor* bias,
    const aclTensor* x1Scale,
    const aclTensor* x2Scale,
    const aclTensor* quantScale,
    int64_t          blockSize,
    const char*      group,
    const char*      reduceOp,
    int64_t          commTurn,
    int64_t          streamMode,
    int64_t          groupSize,
    const char*      commMode,
    aclTensor*       output,
    aclTensor*       amaxOutOptional,
    uint64_t*        workspaceSize,
    aclOpExecutor**  executor)
```

```cpp
aclnnStatus aclnnMatmulReduceScatterV2(
    void          *workspace,
    uint64_t       workspaceSize,
    aclOpExecutor *executor,
    aclrtStream    stream)
```

## aclnnMatmulReduceScatterV2GetWorkspaceSize

- **参数说明：**
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
        <td>x1</td>
        <td>输入</td>
        <td>MM左矩阵，即计算公式中的x1。</td>
        <td>当前版本仅支持两维输入，shape为[m, k]，且仅支持不转置场景。</td>
        <td>FLOAT16、BFLOAT16、FLOAT8_E4M3FN、FLOAT8_E5M2、HIFLOAT8、FLOAT4_E2M1、INT8</td>
        <td>ND</td>
        <td>2</td>
        <td>×</td>
    </tr>
    <tr>
        <td>x2</td>
        <td>输入</td>
        <td>MM右矩阵，即计算公式中的x2。</td>
        <td><ul><li>当前版本仅支持二维输入， shape为[m, k]，支持转置/不转置场景。</li><li>仅支持两根轴转置情况下的非连续Tensor，其他场景的<a href="../../docs/zh/context/non_contiguous_tensor.md">非连续的Tensor</a>不支持。MX量化场景下x2仅支持转置输入，且k必须为偶数。</li></ul></td>
        <td>FLOAT16、BFLOAT16、FLOAT8_E4M3FN、FLOAT8_E5M2、HIFLOAT8、FLOAT4_E2M1、INT8</td>
        <td>ND、FRACTAL_NZ</td>
        <td>2</td>
        <td>√（仅适用转置场景）</td>
    </tr>
    <tr>
        <td>bias</td>
        <td>输入</td>
        <td>即计算公式中的bias。</td>
        <td><ul><li>支持传入空指针场景。</li><li>当前版本仅支持一维输入。</li></ul></td>
        <td>FLOAT16、BFLOAT16、FLOAT</td>
        <td>ND</td>
        <td>1</td>
        <td>×</td>
    </tr>
    <tr>
        <td>x1Scale</td>
        <td>输入</td>
        <td>mm左矩阵反量化参数。</td>
        <td>支持传入空指针场景。</td>
        <td>FLOAT16、BFLOAT16、FLOAT、FLOAT8_E8M0</td>
        <td>ND</td>
        <td>1-3</td>
        <td>×</td>
    </tr>
    <tr>
        <td>x2Scale</td>
        <td>输入</td>
        <td>mm右矩阵反量化参数。</td>
        <td>支持传入空指针场景。MX量化场景下x2Scale仅支持转置输入。</td>
        <td>FLOAT16、BFLOAT16、FLOAT、INT64、FLOAT8_E8M0</td>
        <td>ND</td>
        <td>1-3</td>
        <td>√（仅适用转置场景）</td>
    </tr>
    <tr>
        <td>quantScale</td>
        <td>输入</td>
        <td>输出矩阵量化scale。</td>
        <td>当前仅支持传入空指针场景。</td>
        <td>FLOAT</td>
        <td>ND</td>
        <td>1</td>
        <td>×</td>
    </tr>
    <tr>
        <td>blockSize</td>
        <td>输入</td>
        <td>用于表示mm输出矩阵在M轴方向上和N轴方向上可以用于对应方向上的多少个数的量化。</td>
        <td>blockSize由blockSizeM、blockSizeN、blockSizeK三个值拼接而成，每个值占16位，计算公式为blockSize = blockSizeK | blockSizeN << 16 | blockSizeM << 32，mm输出矩阵不涉及K轴，blockSizeK固定为0,当前版本只支持blockSizeM=blockSizeN=0。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>group</td>
        <td>输入</td>
        <td>通信域名称。</td>
        <td>通过Hccl提供的接口“extern HcclResult HcclGetCommName(HcclComm comm, char* commName);”获取，其中commName即为group。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>reduceOp</td>
        <td>输入</td>
        <td>reduce操作类型。</td>
        <td>当前版本仅支持“sum”。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>commTurn</td>
        <td>输入</td>
        <td>通信数据切分数，即总数据量/单次通信量。</td>
        <td>当前版本仅支持输入0。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>streamMode</td>
        <td>输入</td>
        <td>流模式的枚举。</td>
        <td>当前只支持枚举值1。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>groupSize</td>
        <td>输入</td>
        <td>用于表示反量化中x1Scale/x2Scale输入的一个数在其所在的对应维度方向上可以用于该方向x1/x2输入的多少个数的反量化。</td>
        <td>groupSize输入由3个方向的groupSizeM、groupSizeN、groupSizeK三个值拼接组成，每个值占16位，计算公式为groupSize = groupSizeK | groupSizeN << 16 | groupSizeM << 32。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>commMode</td>
        <td>输入</td>
        <td>通信模式。</td>
        <td>支持输入"ai_cpu"、"ccu"或"aiv"</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>output</td>
        <td>输出</td>
        <td>ReduceScatter通信与MatMul计算的结果，即计算公式中的output。</td>
        <td>仅当输出类型为FLOAT16、BFLOAT16时支持空Tensor。</td>
        <td>FLOAT16、BFLOAT16、FLOAT</td>
        <td>ND</td>
        <td>2</td>
        <td>×</td>
    </tr>
    <tr>
        <td>amaxOutOptional</td>
        <td>输出</td>
        <td>MM计算的最大值结果，即公式中的amaxOut。</td>
        <td>当前版本仅支持nullptr或空tensor。</td>
        <td>FLOAT</td>
        <td>ND</td>
        <td>1</td>
        <td>×</td>
    </tr>
    <tr>
        <td>workspaceSize</td>
        <td>输出</td>
        <td>返回需要在Device侧申请的workspace大小。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>executor</td>
        <td>输出</td>
        <td>返回op执行器，包含了算子计算流程。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    </tbody></table>

    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>  ：
        - x1、x2：在commMode为aicpu时，数据类型支持FLOAT16、BFLOAT16；commMode为aiv时，数据类型支持FLOAT16、BFLOAT16、INT8，x1数据格式仅支持ND，x2数据格式支持ND、FRACTAL_NZ。
        - bias：在commMode为aicpu时，数据类型支持FLOAT16、BFLOAT16，仅支持为0的输入。在commMode为aiv时，当x1和x2数据类型为INT8时，bias数据类型可以是FLOAT16、BFLOAT16、FLOAT；当x1和x2数据类型为FLOAT16时，bias数据类型必须为FLOAT16、FLOAT；当x1和x2数据类型为BFLOAT16时，bias数据类型必须为BFLOAT16、FLOAT。
        - x1Scale：在commMode为aicpu时，仅支持输入nullptr。在commMode为aiv时，数据类型支持FLOAT。当x1和x2数据类型为FLOAT16/BFLOAT16时，仅支持输入为nullptr。在pertoken场景，shape为(m, 1)。
        - x2Scale：在commMode为aicpu时，仅支持输入nullptr。在commMode为aiv时，数据类型支持FLOAT、INT64，数据格式支持ND。INT64数据类型仅在output数据类型为FLOAT16场景支持。当x1和x2数据类型为FLOAT16、BFLOAT16时，仅支持输入为nullptr。在perchannel场景，shape为(1, n)。
        - groupSize：当前版本仅支持输入为0。
        - commMode：当前仅支持aiv模式。aiv模式下使用AI VECTOR核完成通信任务。当前版本仅支持输入“aiv”。
        - output：数据类型支持FLOAT16、BFLOAT16。如果x1类型为FLOAT16、BFLOAT16，则output类型与x1保持一致。
    - <term>Ascend 950PR/Ascend 950DT</term>：
        - x1、x2：数据类型支持FLOAT16、BFLOAT16、FLOAT8_E4M3FN、FLOAT8_E5M2、HIFLOAT8、FLOAT4_E2M1，数据格式仅支持ND。
        - bias：如果x1的数据类型是FLOAT16、BFLOAT16，则bias的数据类型必须为FLOAT16、BFLOAT16。如果x1的数据类型是FLOAT8_E4M3FN、FLOAT8_E5M2、HIFLOAT8、FLOAT4_E2M1时，在pertensor和mx量化场景下，bias的数据类型必须为FLOAT。在perblock场景下，仅支持输入为nullptr。
        - x1Scale：当x1和x2数据类型为FLOAT16、BFLOAT16时，仅支持输入为nullptr。在pertensor场景下，shape为[1]。在perblock场景下，shape为[ceildiv(m, 128), ceildiv(k, 128)]。在pertensor和perblock场景下，数据类型支持FLOAT。在mx量化场景下，数据类型为FLOAT8_E8M0，shape为(m, ceilDiv(k, 64), 2)。
        - x2Scale：当x1和x2数据类型为FLOAT16、BFLOAT16时，仅支持输入为nullptr。在pertensor场景下，shape为[1]。在perblock场景下，shape为[ceildiv(k, 128), ceildiv(n, 128)]。在pertensor和perblock场景下，数据类型支持FLOAT。在mx场景下，数据类型为FLOAT8_E8M0，shape为(n, ceilDiv(k, 64), 2)，仅支持转置输入。
        - groupSize:
            - 仅当x1Scale和x2Scale输入都是2维及以上数据时，groupSize取值有效，其他场景需传入0。
            - groupSize值支持公式推导：传入的groupSize内部会按如下公式分解得到groupSizeM、groupSizeN、groupSizeK，当其中有1个或多个为0，会根据x1/x2/x1Scale/x2Scale输入shape重新设置groupSizeM、groupSizeN、groupSizeK用于计算。设置原理：如果groupSizeM=0，表示m方向量化分组值由接口推导，推导公式为groupSizeM = m / scaleM（需保证m能被scaleM整除），其中m与x1 shape中的m一致，scaleM与x1Scale shape中的m一致；如果groupSizeK=0，表示k方向量化分组值由接口推导，推导公式为groupSizeK = k / scaleK（需保证k能被scaleK整除），其中k与x1 shape中的k一致，scaleK与x1Scale shape中的k一致；如果groupSizeN=0，表示n方向量化分组值由接口推导，推导公式为groupSizeN = n / scaleN（需保证n能被scaleN整除），其中n与x2 shape中的n一致，scaleN与x2Scale shape中的n一致。
            $$
            groupSize = groupSizeK | groupSizeN << 16 | groupSizeM << 32
            $$
            - 如果满足重新设置条件，一般情况下，当x1Scale、x2Scale输入都是2维，且数据类型都为FLOAT时，[groupSizeM, groupSizeN, groupSizeK]取值组合会推导为[128, 128, 128]，对应groupSize的值为549764202624；当x1Scale、x2Scale输入都是3维，且数据类型都为FLOAT8_E8M0时，[groupSizeM, groupSizeN, groupSizeK]取值组合会推导为[1, 1, 32]，对应groupSize的值为4295032864。
        - commMode：当前版本支持输入“ai_cpu” 或 “ccu”。
        - output：如果x1类型为FLOAT16、BFLOAT16，则output类型与x1保持一致。如果x1类型为FLOAT8_E4M3FN、FLOAT8_E5M2、HIFLOAT8、FLOAT4_E2M1，则数据类型支持FLOAT16、BFLOAT16、FLOAT。

            $$
            groupSize = groupSizeK | groupSizeN << 16 | groupSizeM << 32
            $$

- **返回值**

    返回aclnnStatus状态码，具体参见[aclnn返回码](../../docs/zh/context/aclnn_return_code.md)。

    第一段接口完成入参校验，出现以下场景时报错：

    <table style="undefined;table-layout: fixed; width: 1166px"> <colgroup>
    <col style="width: 267px">
    <col style="width: 124px">
    <col style="width: 775px">
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
        <td>传入的x1、x2或output为空指针。</td>
    </tr>
    <tr>
        <td>ACLNN_ERR_PARAM_INVALID</td>
        <td>161002</td>
        <td>传入的x1、x2、output、bias(非空场景)、x1Scale(非空场景)、x2Scale(非空场景)、quantScale(非空场景)的数据格式或数据类型不在支持范围内。</td>
    </tr>
    </tbody>
    </table>

## aclnnMatmulReduceScatterV2

- **参数说明：**

    <table style="undefined;table-layout：fixed; width：1166px"> <colgroup>
    <col style="width：173px">
    <col style="width：133px">
    <col style="width：860px">
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
        <td>在Device侧申请的workspace大小，由第一段接口<code>aclnnMatmulReduceScatterV2GetWorkspaceSize</code>获取。</td>
    </tr>
    <tr>
        <td>executor</td>
        <td>输入</td>
        <td>op执行器，包含了算子计算流程。</td>
    </tr>
    <tr>
        <td>stream</td>
        <td>输入</td>
        <td>指定执行任务的stream。</td>
    </tr>
    </tbody></table>

- **返回值**

    返回aclnnStatus状态码，具体参见[aclnn返回码](../../docs/zh/context/aclnn_return_code.md)。

## 约束说明

- 通信引擎约束：
  - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：仅支持commMode为"aiv"。
  - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：仅支持commMode为"aiv"。
  - <term>Ascend 950PR/Ascend 950DT</term>：当前版本支持输入commMode为“ai_cpu” 或 “ccu”。

- 确定性计算：
  - `aclnnMatmulReduceScatterV2`默认采用确定性计算实现。
- <term>Ascend 950PR/Ascend 950DT</term>：
    - 只支持x2矩阵转置/不转置，x1矩阵仅支持不转置场景。
    - 输入x1为2维，其shape为\(m, k\)，m须为卡数rank\_size的整数倍。
    - 输入x2必须是2维，其shape为\(k, n\)，轴满足mm算子入参要求，k轴相等，且k轴取值范围为\[256, 65535\)。
    - bias为1维，shape为\(n,\)。
    - 输出为2维，其shape为\(m/rank\_size, n\), rank\_size为卡数。
    - 当x1、x2的数据类型为FLOAT16/BFLOAT16时，x1/x2支持的空tensor场景，m和n可以为空，k不可为空，且需满足以下条件：
        - m为空，k不为空，n不为空；
        - m不为空，k不为空，n为空；
        - m为空，k不为空，n为空。
    - 当x1、x2的数据类型为FLOAT8_E4M3FN/FLOAT8_E5M2/HIFLOAT8/FLOAT4_E2M1时，不支持空tensor。
    - 当x1、x2的数据类型为FLOAT16/BFLOAT16/HIFLOAT8/FLOAT4_E2M1时，x1和x2的数据类型需要保持一致。
    - 当x1、x2的数据类型为FLOAT8_E4M3FN/FLOAT8_E5M2时，x1和x2的数据类型可以为其中任意一种。
    - mx量化场景下，x2/x2Scale仅支持转置输入。
    - mx量化场景下，且x1和x2输入为FLOAT4_E2M1（MXFP4量化）时，k必须是偶数。
    - 支持2、4、8、16、32、64卡；支持CCU通信和AICPU通信，CCU仅支持单机UB域内互联，AICPU可支持跨机UB域内互联。
    - ReduceScatter集合通信数据总量不能超过16 * 256MB，集合通信数据总量计算方式为：m * n * sizeof(output_dtype)。由于shape不同，算子内部实现可能存在差异，实际支持的总通信量可能略小于该值。

- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：
    - 只支持x2矩阵转置/不转置，x1矩阵仅支持不转置场景。
    - 输入x1为2维，其shape为\(m, k\)，m须为卡数rank\_size的整数倍。
    - 输入x2必须是2维，其shape为\(k, n\)，轴满足mm算子入参要求，k轴相等，且k轴取值范围为\[256, 65535\)。
    - bias为1维，shape为\(n,\)。
    - 输出为2维，其shape为\(m/rank\_size, n\), rank\_size为卡数。
    - 不支持空tensor。
    - x1和x2的数据类型需要保持一致。
    - 支持2、4、8卡。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_matmul_reduce_scatter_v2.cpp](./examples/test_aclnn_matmul_reduce_scatter_v2.cpp) | 通过[aclnnMatmulReduceScatterV2](./docs/aclnnMatmulReduceScatterV2.md)接口方式调用MatmulReduceScatterV2算子。 |
