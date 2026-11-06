# AlltoAllvQuantGroupedMatMul

English | [简体中文](./README.md)

## Product Support Status

| Product                                        | Supported |
| :------------------------------------------ | :------: |
| <term>Ascend 950DT products</term>               |    √     |
| <term>Atlas A3 products</term>  |    ×     |
| <term>Atlas A2 products</term>  |    ×     |
| <term>Atlas 200I/500 A2 inference products</term>                   |    ×     |
| <term>Atlas inference products</term>                           |    ×     |
| <term>Atlas training products</term>                           |    ×     |

## Function Description

- **Operator function**: Fuses AlltoAllv and quantized GroupedMatMul for routed experts, and enables parallel fusion with the quantized shared-expert MatMul. **Communication comes before computation**.

- **Computation formula**:
Assume that the total number of cards in the communicator is epWorldSize, the number of routed experts on each card after communication is e, and the grouped matrix multiplication on each card computes only the experts of that card. The computation formula for each card is as follows:
  - Grouped matrix multiplication computation of the shared expert on the current card

    $$
    mm\_y=(mm\_x × mm\_x\_scale) @ (mm\_weight × mm\_weight\_scale)
    $$

  - Alltoallv communication and permute

    $$
    permute\_out=Alltoallv(gmm\_x)
    $$

  - Grouped matrix multiplication computation of routed experts along the expert dimension on the current card

    $$
    gmm\_y=(permute\_out × gmm\_x\_scale) @ (gmm\_weight × gmm\_weight\_scale)
    $$

## Parameter Description

<table style="undefined;table-layout: fixed; width: 1576px"><colgroup>
    <col style="width: 170px">
    <col style="width: 170px">
    <col style="width: 200px">
    <col style="width: 200px">
    <col style="width: 170px">
    </colgroup>
    <thead>
    <tr>
        <th>Parameter Name</th>
        <th>Input/Output/Attribute</th>
        <th>Description</th>
        <th>Data Type</th>
        <th>Data Format</th>
    </tr></thead>
    <tbody>
    <tr>
        <td>gmmX</td>
        <td>Input</td>
        <td>After AlltoAllv communication, the result of this input serves as the left matrix of the GroupedMatMul computation.</td>
        <td>HIFLOAT8, FLOAT8_E4M3FN, FLOAT8_E5M2, FLOAT4_E2M1</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>gmmWeight</td>
        <td>Input</td>
        <td>Right matrix of the GroupedMatMul computation.</td>
        <td>HIFLOAT8, FLOAT8_E4M3FN, FLOAT8_E5M2, FLOAT4_E2M1</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>gmmXScale</td>
        <td>Input</td>
        <td>Quantization factor of gmmX.</td>
        <td>FLOAT32, FLOAT8_E8M0</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>gmmWeightScale</td>
        <td>Input</td>
        <td>Quantization factor of gmmWeight.</td>
        <td>FLOAT32, FLOAT8_E8M0</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>sendCountsTensorOptional</td>
        <td>Input</td>
        <td>Reserved parameter. In the current version, only nullptr can be passed.</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>recvCountsTensorOptional</td>
        <td>Input</td>
        <td>Reserved parameter. In the current version, only nullptr can be passed.</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>mmXOptional</td>
        <td>Input</td>
        <td>Optional input. Left matrix in the shared-expert MatMul computation.</td>
        <td>Same as gmmX</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>mmWeightOptional</td>
        <td>Input</td>
        <td>Optional input. Right matrix in the shared-expert MatMul computation.</td>
        <td>Same as gmmWeight</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>mmXScaleOptional</td>
        <td>Input</td>
        <td>Optional input. Quantization factor of mmX.</td>
        <td>FLOAT32, FLOAT8_E8M0</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>mmWeightScaleOptional</td>
        <td>Input</td>
        <td>Optional input. Quantization factor of mmWeight.</td>
        <td>FLOAT32, FLOAT8_E8M0</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>gmmXQuantMode</td>
        <td>Input</td>
        <td>Quantization mode of gmmX.</td>
        <td>INT64</td>
        <td>-</td>
    </tr>
    <tr>
        <td>gmmWeightQuantMode</td>
        <td>Input</td>
        <td>Quantization mode of gmmWeight.</td>
        <td>INT64</td>
        <td>-</td>
    </tr>
    <tr>
        <td>mmXQuantMode</td>
        <td>Input</td>
        <td>Quantization mode of mmX.</td>
        <td>INT64</td>
        <td>-</td>
    </tr>
    <tr>
        <td>mmWeightQuantMode</td>
        <td>Input</td>
        <td>Quantization mode of mmWeight.</td>
        <td>INT64</td>
        <td>-</td>
    </tr>
    <tr>
        <td>group</td>
        <td>Input</td>
        <td>Communicator name for expert parallelism. The string length must be in the range (0, 128).</td>
        <td>STRING</td>
        <td>-</td>
    </tr>
    <tr>
        <td>epWorldSize</td>
        <td>Input</td>
        <td>Size of the EP communicator.</td>
        <td>INT64</td>
        <td>-</td>
    </tr>
    <tr>
        <td>sendCounts</td>
        <td>Input</td>
        <td>Number of tokens sent to other cards.</td>
        <td>aclIntArray* (element type INT64)</td>
        <td>-</td>
    </tr>
    <tr>
        <td>recvCounts</td>
        <td>Input</td>
        <td>Number of tokens received from other cards.</td>
        <td>aclIntArray* (element type INT64)</td>
        <td>-</td>
    </tr>
    <tr>
        <td>transGmmWeight</td>
        <td>Input</td>
        <td>Whether the right matrix of GroupedMatMul needs to be transposed.</td>
        <td>BOOL</td>
        <td>-</td>
    </tr>
    <tr>
        <td>transMmWeight</td>
        <td>Input</td>
        <td>Whether the right matrix of the shared-expert MatMul needs to be transposed.</td>
        <td>BOOL</td>
        <td>-</td>
    </tr>
    <tr>
        <td>groupSize</td>
        <td>Input</td>
        <td>Indicates, in quantization, how many elements of the gmmX/gmmWeight/mmX/mmWeight input along the corresponding dimension can be quantized using one element of the gmmXScale/gmmWeightScale/mmXScale/mmWeightScale input along that dimension.</td>
        <td>INT64</td>
        <td>-</td>
    </tr>
    <tr>
        <td>permuteOutFlag</td>
        <td>Input</td>
        <td>Whether permuteOutOptional needs to be output.</td>
        <td>BOOL</td>
        <td>-</td>
    </tr>
    <tr>
        <td>gmmY</td>
        <td>Output</td>
        <td>Output of the routed-expert computation.</td>
        <td>FLOAT16, BFLOAT16</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>mmYOptional</td>
        <td>Output</td>
        <td>Output of the shared-expert computation.</td>
        <td>Same as gmmY</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>permuteOutOptional</td>
        <td>Output</td>
        <td>Output after the permute operation.</td>
        <td>Same as gmmX</td>
        <td>ND</td>
    </tr>
    </tbody></table>

## Constraints

- Communication engine constraints: CCU communication and AI_CPU communication are supported. CCU supports only intra-node interconnection within the UB domain, and AI_CPU supports inter-node interconnection within the UB domain.

- Deterministic computation:
  - `aclnnAlltoAllvQuantGroupedMatMul` is deterministic by default.

- Variables used in shapes in the parameter description:
  - BSK: number of tokens sent by the current card. It is the sum of the sendCounts parameter. Value range: (0, 52428800).
  - H1: hidden size of the routed expert. Value range: (0, 65536).
  - H2: hidden size of the shared expert. Value range: (0, 12288].
  - e: number of experts on a single card. Value range: (0, 32]. e * epWorldSize supports a maximum of 256.
  - N1: head_num of the routed expert. Value range: (0, 65536).
  - N2: head_num of the shared expert. Value range: (0, 65536).
  - BS: batch sequence size.
  - K: number of TopK experts selected. Range of K: [2, 8].
  - A: number of tokens received by the current card. It is the sum of the recvCounts parameter.
  - The sum of A over all cards in the EP communicator equals the sum of BSK over all cards.
  - For mx quantization when gmmX and gmmWeight are FLOAT4_E2M1, H1 and H2 must be even numbers and cannot be 2. When transGmmWeight and transMmWeight are false, N1 and N2 must be even numbers.
  - The transpose states of gmmWeight and gmmWeightScale must be consistent: both transposed or both not transposed. The transpose states of mmWeight and mmWeightScale must also be consistent.
  - groupSize:
    - groupSize is valid only when the gmmXScale/gmmWeightScale/mmXScale/mmWeightScale inputs are 2D or higher; in other scenarios, 0 must be passed.
    - The groupSize value supports formula-based derivation: the passed groupSize is internally decomposed into groupSizeM, groupSizeN, and groupSizeK according to the following formula. When one or more of them are 0, groupSizeM, groupSizeN, and groupSizeK used for computation are reset based on the shapes of the gmmX/gmmWeight/mmX/mmWeight/gmmXScale/gmmWeightScale/mmXScale/mmWeightScale inputs. The setting principle: if groupSizeM=0, the quantization grouping value along the m dimension is derived by the API, and the derivation formula is groupSizeM = m / scaleM (m must be divisible by scaleM), where m is the same as m in the gmmX/mmX shape, and scaleM is the same as m in the gmmXScale/mmXScale shape; if groupSizeK=0, the quantization grouping value along the k dimension is derived by the API, and the derivation formula is groupSizeK = k / scaleK (k must be divisible by scaleK), where k is the same as k in the gmmX/mmX shape, and scaleK is the same as k in the gmmXScale/mmXScale shape; if groupSizeN=0, the quantization grouping value along the n dimension is derived by the API, and the derivation formula is groupSizeN = n / scaleN (n must be divisible by scaleN), where n is the same as n in the gmmWeight/mmWeight shape, and scaleN is the same as n in the gmmWeightScale/mmWeightScale shape.
    $$
    groupSize = groupSizeK | groupSizeN << 16 | groupSizeM << 32
    $$
    - If the reset condition is met, when the gmmXScale/gmmWeightScale/mmXScale/mmWeightScale inputs are 2D or higher and their data types are all FLOAT8_E8M0, the value combination [groupSizeM, groupSizeN, groupSizeK] is derived as [1, 1, 32], and the corresponding groupSize value is 4295032864.

- Quantization parameter constraints:
  - The current version supports pertensor quantization and mx quantization.
- Type Constraints
  - pertensor quantization

    | gmmX | gmmWeight | gmmXScale | gmmWeightScale | mmXScale | mmWeightScale | gmmY |
    | :------: | :------: | :------: | :------: | :------: | :------: | :------: |
    | HIFLOAT8 | HIFLOAT8 | FLOAT32 | FLOAT32 | FLOAT32 | FLOAT32 | FLOAT16/BFLOAT16 |

  - mx quantization

    | gmmX | gmmWeight | gmmXScale | gmmWeightScale | mmXScale | mmWeightScale | gmmY |
    | :------: | :------: | :------: | :------: | :------: | :------: | :------: |
    | FLOAT8_E4M3FN | FLOAT8_E4M3FN | FLOAT8_E8M0 | FLOAT8_E8M0 | FLOAT8_E8M0 | FLOAT8_E8M0 | FLOAT16/BFLOAT16 |
    | FLOAT8_E4M3FN | FLOAT8_E5M2 | FLOAT8_E8M0 | FLOAT8_E8M0 | FLOAT8_E8M0 | FLOAT8_E8M0 | FLOAT16/BFLOAT16 |
    | FLOAT8_E5M2 | FLOAT8_E4M3FN | FLOAT8_E8M0 | FLOAT8_E8M0 | FLOAT8_E8M0 | FLOAT8_E8M0 | FLOAT16/BFLOAT16 |
    | FLOAT8_E5M2 | FLOAT8_E5M2 | FLOAT8_E8M0 | FLOAT8_E8M0 | FLOAT8_E8M0 | FLOAT8_E8M0 | FLOAT16/BFLOAT16 |
    | FLOAT4_E2M1 | FLOAT4_E2M1 | FLOAT8_E8M0 | FLOAT8_E8M0 | FLOAT8_E8M0 | FLOAT8_E8M0 | FLOAT16/BFLOAT16 |

  - The type of mmX is the same as that of gmmX, the type of mmWeight is the same as that of gmmWeight, the type of mmY is the same as that of gmmY, and the type of permuteOut is the same as that of gmmX.

## Calling Instructions

| Calling Method   | Sample Code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn API  | [test_aclnn_allto_allv_quant_grouped_mat_mul.cpp](./examples/test_aclnn_allto_allv_quant_grouped_mat_mul.cpp) | Calls the AlltoAllvQuantGroupedMatMul operator through the [aclnnAlltoAllvQuantGroupedMatMul](docs/aclnnAlltoAllvQuantGroupedMatMul_en.md) API. |
