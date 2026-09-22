# AlltoAllvQuantGroupedMatMul

## 产品支持情况

| 产品                                        | 是否支持 |
| :------------------------------------------ | :------: |
| <term>Ascend 950DT系列产品</term>               |    √     |
| <term>Atlas A3系列产品</term>  |    ×     |
| <term>Atlas A2系列产品</term>  |    ×     |
| <term>Atlas 200I/500 A2推理产品</term>                   |    ×     |
| <term>Atlas推理系列产品</term>                           |    ×     |
| <term>Atlas训练系列产品</term>                           |    ×     |

## 功能说明

- **算子功能**：完成路由专家AlltoAllv、量化GroupedMatMul融合并实现与共享专家量化MatMul并行融合，**先通信后计算**。

- **计算公式**：
假设通信域中的总卡数为epWorldSize，每张卡上通信后路由专家个数为e，每张卡分组矩阵乘只负责本卡专家的计算。对于每张卡的计算公式如下：
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

## 参数说明

<table style="undefined;table-layout: fixed; width: 1576px"><colgroup>
    <col style="width: 170px">
    <col style="width: 170px">
    <col style="width: 200px">
    <col style="width: 200px">
    <col style="width: 170px">
    </colgroup>
    <thead>
    <tr>
        <th>参数名</th>
        <th>输入/输出/属性</th>
        <th>描述</th>
        <th>数据类型</th>
        <th>数据格式</th>
    </tr></thead>
    <tbody>
    <tr>
        <td>gmmX</td>
        <td>输入</td>
        <td>该输入进行AlltoAllv通信后结果作为GroupedMatMul计算的左矩阵。</td>
        <td>HIFLOAT8、FLOAT8_E4M3FN、FLOAT8_E5M2、FLOAT4_E2M1</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>gmmWeight</td>
        <td>输入</td>
        <td>GroupedMatMul计算的右矩阵。</td>
        <td>HIFLOAT8、FLOAT8_E4M3FN、FLOAT8_E5M2、FLOAT4_E2M1</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>gmmXScale</td>
        <td>输入</td>
        <td>gmmX的量化系数。</td>
        <td>FLOAT32、FLOAT8_E8M0</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>gmmWeightScale</td>
        <td>输入</td>
        <td>gmmWeight的量化系数。</td>
        <td>FLOAT32、FLOAT8_E8M0</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>sendCountsTensorOptional</td>
        <td>输入</td>
        <td>预留参数，当前版本仅支持传nullptr。</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>recvCountsTensorOptional</td>
        <td>输入</td>
        <td>预留参数，当前版本仅支持传nullptr。</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>mmXOptional</td>
        <td>输入</td>
        <td>可选输入，共享专家MatMul计算中的左矩阵。</td>
        <td>与gmmX保持一致</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>mmWeightOptional</td>
        <td>输入</td>
        <td>可选输入，共享专家MatMul计算中的右矩阵。</td>
        <td>与gmmWeight保持一致</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>mmXScaleOptional</td>
        <td>输入</td>
        <td>可选输入，mmX的量化系数。</td>
        <td>FLOAT32、FLOAT8_E8M0</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>mmWeightScaleOptional</td>
        <td>输入</td>
        <td>可选输入，mmWeight的量化系数。</td>
        <td>FLOAT32、FLOAT8_E8M0</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>gmmXQuantMode</td>
        <td>输入</td>
        <td>gmmX的量化模式。</td>
        <td>INT64</td>
        <td>-</td>
    </tr>
    <tr>
        <td>gmmWeightQuantMode</td>
        <td>输入</td>
        <td>gmmWeight的量化模式。</td>
        <td>INT64</td>
        <td>-</td>
    </tr>
    <tr>
        <td>mmXQuantMode</td>
        <td>输入</td>
        <td>mmX的量化模式。</td>
        <td>INT64</td>
        <td>-</td>
    </tr>
    <tr>
        <td>mmWeightQuantMode</td>
        <td>输入</td>
        <td>mmWeight的量化模式。</td>
        <td>INT64</td>
        <td>-</td>
    </tr>
    <tr>
        <td>group</td>
        <td>输入</td>
        <td>专家并行的通信域名，字符串长度要求(0, 128)。</td>
        <td>STRING</td>
        <td>-</td>
    </tr>
    <tr>
        <td>epWorldSize</td>
        <td>输入</td>
        <td>ep通信域大小。</td>
        <td>INT64</td>
        <td>-</td>
    </tr>
    <tr>
        <td>sendCounts</td>
        <td>输入</td>
        <td>表示发送给其他卡的token数。</td>
        <td>aclIntArray*（元素类型INT64）</td>
        <td>-</td>
    </tr>
    <tr>
        <td>recvCounts</td>
        <td>输入</td>
        <td>表示接收其他卡的token数。</td>
        <td>aclIntArray*（元素类型INT64）</td>
        <td>-</td>
    </tr>
    <tr>
        <td>transGmmWeight</td>
        <td>输入</td>
        <td>GroupedMatMul的右矩阵是否需要转置。</td>
        <td>BOOL</td>
        <td>-</td>
    </tr>
    <tr>
        <td>transMmWeight</td>
        <td>输入</td>
        <td>共享专家MatMul的右矩阵是否需要转置。</td>
        <td>BOOL</td>
        <td>-</td>
    </tr>
    <tr>
        <td>groupSize</td>
        <td>输入</td>
        <td>用于表示量化中gmmXScale/gmmWeightScale/mmXScale/mmWeightScale输入的一个数在其所在的对应维度方向上可以用于该方向gmmX/gmmWeight/mmX/mmWeight输入的多少个数的量化。</td>
        <td>INT64</td>
        <td>-</td>
    </tr>
    <tr>
        <td>permuteOutFlag</td>
        <td>输入</td>
        <td>permuteOutOptional是否需要输出。</td>
        <td>BOOL</td>
        <td>-</td>
    </tr>
    <tr>
        <td>gmmY</td>
        <td>输出</td>
        <td>路由专家计算的输出。</td>
        <td>FLOAT16、BFLOAT16</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>mmYOptional</td>
        <td>输出</td>
        <td>共享专家计算的输出。</td>
        <td>与gmmY保持一致</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>permuteOutOptional</td>
        <td>输出</td>
        <td>permute之后的输出。</td>
        <td>与gmmX保持一致</td>
        <td>ND</td>
    </tr>
    </tbody></table>

## 约束说明

- 通信引擎约束：支持CCU通信和AI_CPU通信，CCU仅支持单机UB域内互联，AI_CPU可支持跨机UB域内互联。

- 确定性计算：
  - `aclnnAlltoAllvQuantGroupedMatMul`默认确定性实现。

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

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_allto_allv_quant_grouped_mat_mul.cpp](./examples/test_aclnn_allto_allv_quant_grouped_mat_mul.cpp) | 通过[aclnnAlltoAllvQuantGroupedMatMul](./docs/aclnnAlltoAllvQuantGroupedMatMul.md)接口方式调用AlltoAllvQuantGroupedMatMul算子。 |
