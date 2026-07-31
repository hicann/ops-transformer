# AlltoAllMatmul

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>       |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                             |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×     |

## 功能说明

- 算子功能：完成AlltoAll通信、Permute（保证通信后地址连续）和Matmul计算的融合，**先通信后计算**，支持非量化、K-C量化、K-C动态量化和mx[量化模式](../../docs/zh/context/quant_mode_introduction.md)。
- 计算公式：假设x1输入shape为(BS, H)，mx量化场景下x1Scale输入shape为(BS, ceil(H/64), 2)，rankSize为NPU卡数

    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：
      - 非量化场景：

        $$
        commOut = AlltoAll(x1.view(rankSize, BS/rankSize, H)) \\
        permutedOut = commOut.permute(1, 0, 2).view(BS/rankSize, rankSize*H) \\
        output = permutedOut @ x2 + bias \\
        $$

      - K-C量化场景：

        $$
        commOut = AlltoAll(x1.view(rankSize, BS/rankSize, H)) \\
        permutedOut = commOut.permute(1, 0, 2).view(BS/rankSize, rankSize*H) \\
        output_{quant} = x1 @ x2 \\
        output = output_{quant} \times x1_{scale} \times x2_{scale} \\
        output = output + bias
        $$

      - K-C动态量化场景：

        $$
        commOut = AlltoAll(x1.view(rankSize, BS/rankSize, H)) \\
        permutedOut = commOut.permute(1, 0, 2).view(BS/rankSize, rankSize*H) \\
        x1_{quant}, x1_{scale} = Quant(permutedOut) \\
        output_{quant} = x1_{quant} @ x2 \\
        output = output_{quant} \times x1_{scale} \times x2_{scale} \\
        output = output + bias
        $$

    - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>  ：
      - 非量化场景：

        $$
        commOut = AlltoAll(x1.view(rankSize, BS/rankSize, H)) \\
        permutedOut = commOut.permute(1, 0, 2).view(BS/rankSize, rankSize*H) \\
        output = permutedOut @ x2 + bias \\
        $$

    - <term>Ascend 950PR/Ascend 950DT</term>：
      - 非量化场景：

        $$
        commOut = AlltoAll(x1.view(rankSize, BS/rankSize, H)) \\
        permutedOut = commOut.permute(1, 0, 2).view(BS/rankSize, rankSize*H) \\
        output = permutedOut @ x2 + bias \\
        $$

      - K-C动态量化场景：

        $$
        commOut = AlltoAll(x1.view(rankSize, BS/rankSize, H)) \\
        permutedOut = commOut.permute(1, 0, 2).view(BS/rankSize, rankSize*H) \\
        dynQuantX1, dynQuantX1Scale = dynamicQuant(permutedOut) \\
        output = (dynQuantX1@x2 + bias) \times dynQuantX1Scale \times x2Scale
        $$

      - mx量化场景：

        $$
        commOut = AlltoAll(x1.view(rankSize, BS/rankSize, H)) \\
        permutedOut = commOut.permute(1, 0, 2).view(BS/rankSize, rankSize*H) \\
        commScale = AlltoAll(x1Scale.view(rankSize, BS/rankSize, ceil(H/64), 2)) \\
        permutedScale = commScale.permute(1, 0, 2, 3).view(BS/rankSize, ceil(H/64)*rankSize, 2) \\
        output = \sum_{0}^{\left \lfloor \frac{k}{blockSize=32} \right \rfloor} (permutedOut @ x2 * (permutedScale * x2Scale)) + bias
        $$

## 参数说明​

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
    <td>x1</td>
    <td>输入</td>
    <td>融合算子的左矩阵，即公式中的输入x1。</td>
    <td>FLOAT16、BFLOAT16、INT4、FLOAT8_E4M3FN、FLOAT8_E5M2、FLOAT4_E2M1</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>x2</td>
    <td>输入</td>
    <td>融合算子的右矩阵，也是Matmul的右矩阵，即公式中的输入x2。</td>
    <td>FLOAT16、BFLOAT16、INT8、INT4、FLOAT8_E4M3FN、FLOAT8_E5M2、FLOAT4_E2M1</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>bias</td>
    <td>可选输入</td>
    <td>可选输入，阵乘运算后累加的偏置，对应公式中的bias。</td>
    <td>FLOAT16、BFLOAT16、FLOAT32</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>x1_scale</td>
    <td>可选输入</td>
    <td>左矩阵的量化系数，对应公式中的x1Scale。</td>
    <td>FLOAT32、FLOAT8_E8M0</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>x2_scale</td>
    <td>可选输入</td>
    <td>右矩阵的量化系数，对应公式中的x2Scale。</td>
    <td>FLOAT32、FLOAT8_E8M0</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>comm_scale</td>
    <td>可选输入</td>
    <td>预留参数，低比特通信的量化系数。</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>x1_offset</td>
    <td>可选输入</td>
    <td>预留参数，左矩阵的量化偏置。</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>x2_offset</td>
    <td>可选输入</td>
    <td>预留参数，右矩阵的量化偏置。</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>y</td>
    <td>输出</td>
    <td>计算+通信的结果，即公式中的输出output。</td>
    <td>FLOAT16、BFLOAT16、FLOAT32</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>alltoall_out</td>
    <td>输出</td>
    <td>接收AlltoAll和Permute后的结果，即公式中的输出permutedOut。</td>
    <td>与输入x1保持一致</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>group</td>
    <td>必选属性</td>
    <td>Host侧标识列组的字符串，即通信域名称，通过Hccl接口HcclGetCommName获取commName作为该参数，字符串长度要求(0, 128)。</td>
    <td>STRING</td>
    <td>-</td>
    </tr>
    <tr>
    <td>world_size</td>
    <td>必选属性</td>
    <td>使用的npu卡数，公式中的rankSize。</td>
    <td>INT</td>
    <td>-</td>
    </tr>
    <tr>
    <td>all2all_axes</td>
    <td>可选属性</td>
    <td>AlltoAll和Permute数据交换的方向，支持配置空或者[-2, -1]，传入空时默认按[-2, -1]处理，表示将输入由(BS, H)转为(BS/rankSize, H*rankSize)。</td>
    <td>aclIntArray*(元素类型INT64)</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>x1_quant_mode</td>
    <td>可选属性</td>
    <td>左矩阵的量化方式，按照实际场景配置。</td>
    <td>INT</td>
    <td>-</td>
    </tr>
    <tr>
    <td>x2_quant_mode</td>
    <td>可选属性</td>
    <td>右矩阵的量化方式，按照实际场景配置。</td>
    <td>INT</td>
    <td>-</td>
    </tr>
    <tr>
    <td>comm_quant_mode</td>
    <td>可选属性</td>
    <td>低比特通信的量化方式，预留参数，当前仅支持配置为0，表示不量化。</td>
    <td>INT</td>
    <td>-</td>
    </tr>
    <tr>
    <td>comm_quant_dtype</td>
    <td>可选属性</td>
    <td>低比特通信的量化类型，预留参数，当前仅支持配置为-1，表示ACL_DT_UNDEFINED。</td>
    <td>INT</td>
    <td>-</td>
    </tr>
    <tr>
    <td>x1_quant_dtype</td>
    <td>可选属性</td>
    <td>量化Matmul左矩阵的量化类型，AlltoAll通信与Permute后的结果，按照该参数配置量化后作为Matmul计算的左矩阵输入，按照实际场景配置。</td>
    <td>INT</td>
    <td>-</td>
    </tr>
    <tr>
    <td>transpose_x1</td>
    <td>可选属性</td>
    <td>标识左矩阵是否转置过，暂不支持配置为True。</td>
    <td>bool</td>
    <td>-</td>
    </tr>
    <tr>
    <td>transpose_x2</td>
    <td>可选属性</td>
    <td>标识右矩阵是否转置过，配置为True时右矩阵Shape为(N，H*rankSize)。</td>
    <td>bool</td>
    <td>-</td>
    </tr>
    <tr>
    <td>group_size</td>
    <td>可选属性</td>
    <td>用于Matmul计算三个方向上的量化分组大小，仅在scale输入都是2维及以上数据时取值有效，其他场景默认传入0即可。</td>
    <td>INT</td>
    <td>-</td>
    </tr>
    <tr>
    <td>alltoall_out_flag</td>
    <td>可选属性</td>
    <td>用于标识是否需要保留AlltoAll和Permute后的结果。</td>
    <td>bool</td>
    <td>-</td>
    </tr>
    </tbody></table>

x1QuantMode、x2QuantMode、commQuantMode的枚举值与[量化模式](../../docs/zh/context/quant_mode_introduction.md)关系如下:

* 0: 不量化
* 1: pertensor
* 2: perchannel
* 3: pertoken
* 4: pergroup
* 5: perblock
* 6: mx量化
* 7: pertoken动态量化

## 约束说明

* 默认支持确定性计算。
* NPU卡数（rankSize），根据设备型号有不同限制：
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：支持2、4、8卡。
    - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>  ：支持2、4、8、16卡。
    - <term>Ascend 950PR/Ascend 950DT</term>：支持2、4、8、16卡。
* 空tensor和非连续tensor的支持度根据不同设备型号有不同的限制：
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：不支持任何空tensor；不支持任何非连续tensor。
    - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Ascend 950PR/Ascend 950DT</term>：仅支持非量化场景下输入x1的第一维度（BS）为0的空tensor，其它空tensor均不支持；仅支持输入x2的转置非连续tensor，其它非连续tensor均不支持。
* 输入x1必须是2维，其shape为(BS, H)，BS必须整除NPU卡数，BS和N的值不得超过2147483647(INT32_MAX)，不支持转置。
* 输入x2必须是2维，其shape为(H\*rankSize, N)，H*rankSize范围根据芯片型号和场景不同有不同约束，详见[量化aclnn约束说明](./docs/aclnnAlltoAllQuantMatmul.md#约束说明)
  和[非量化aclnn约束说明](./docs/aclnnAlltoAllMatmul.md#约束说明)。当处于mx量化场景时，x2必须转置，其shape为(N, H\*rankSize)，transpose_x2配置为True。
* bias若非空，其维度必须为1维，shape为(N)。
* x1_scale若非空，在mx量化场景时，其维度为3维，shape为(BS, ceil(H/64), 2)；在K-C量化场景时，其维度为1维，shape为(BS)；在K-C动态量化场景时，其维度为1维，shape为(H*rankSize)。
* x2_scale若非空，在mx量化场景时，其维度为3维，shape为(N, ceil(H\*rankSize/64), 2)；其它场景中其维度为1维，shape为(N)。
* all2all_axes为1维数组，shape必须为(2)。
* 目前支持的量化模式，根据设备型号有不同限制：
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：支持K-C量化和K-C动态量化模式，x1QuantMode=3或7，x2QuantMode=2。
    - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>  ：目前不支持量化场景。
    - <term>Ascend 950PR/Ascend 950DT</term>：支持K-C动态量化模式，x1QuantMode=7，x2QuantMode=2；mx量化模式，x1QuantMode=6，x2QuantMode=6。
* 非量化场景x1、x2计算输入的数据类型要和output、alltoAllOutOptional计算输出的数据类型一致，传入的x1、x2与output均不为空指针。
* 量化场景x1和alltoAllOutOptional的数据类型一致，传入的x1、x2、x2Scale与output均不为空指针。
* x1、x2和bias计算输入的数据类型根据不同设备型号有不同的限制：
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：
        - 非量化场景下，output计算输出的数据类型为FLOAT16时，bias计算输入的数据类型支持FLOAT16；output计算输出的数据类型为BFLOAT16时，bias计算输入的数据类型支持FLOAT32。
        - 量化场景下，数据类型组合详见[量化aclnn约束说明](./docs/aclnnAlltoAllMatmul.md#约束说明)。
    - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>  ：
        - 非量化场景下，output计算输出的数据类型为FLOAT16时，bias计算输入的数据类型支持FLOAT16；output计算输出的数据类型为BFLOAT16时，bias计算输入的数据类型支持FLOAT32。
        - A3目前不支持量化场景。
    - <term>Ascend 950PR/Ascend 950DT</term>：
        - 非量化场景下，x1/x2计算输入的数据类型为FLOAT16时，bias计算输入的数据类型支持FLOAT16和FLOAT32；x1/x2计算输入的数据类型为BFLOAT16时，bias计算输入的数据类型支持BFLOAT16和FLOAT32。
        - 量化场景下，支持K-C动态量化模式和mx量化模式，x1计算输入的数据类型为FLOAT16、BFLOAT16、FLOAT8_E4M3FN、FLOAT8_E5M2、FLOAT4_E2M1，x2计算输入的数据类型为FLOAT8_E4M3FN、FLOAT8_E5M2、FLOAT4_E2M1，bias的数据类型为FLOAT32或者bias为空，具体类型组合详见[量化aclnn约束说明](./docs/aclnnAlltoAllQuantMatmul.md#约束说明)。
        - mx量化模式下，当x1和x2的数据类型为FLOAT4_E2M1时，两者的数据类型必须一致。
* 通算融合算子不支持并发调用，不同的通算融合算子也不支持并发调用。
* 不支持跨超节点通信，只支持超节点内。
* 通信引擎约束：
   - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：支持MTE通信。
   - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：支持AI\_CPU通信。
   - <term>Ascend 950PR/Ascend 950DT</term>：支持CCU通信和AI\_CPU通信，CCU通信仅支持单机UB域内互联，AI_CPU可支持跨机UB域内互联。

## 调用说明

| 调用方式   | 样例代码                                                                                      | 说明                                                                                           |
| ---------------- |-------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------|
| aclnn接口  | [test_aclnn_allto_all_matmul.cpp](./examples/test_aclnn_allto_all_matmul.cpp)             | 通过[aclnnAlltoAllMatMul](./docs/aclnnAlltoAllMatmul.md)接口方式调用非量化场景的AlltoAllMatMul算子。          |
| aclnn接口  | [test_aclnn_allto_all_quant_matmul.cpp](./examples/test_aclnn_allto_all_quant_matmul.cpp) | 通过[aclnnAlltoAllQuantMatMul](./docs/aclnnAlltoAllQuantMatmul.md)接口方式调用量化场景的AlltoAllMatMul算子。 |
