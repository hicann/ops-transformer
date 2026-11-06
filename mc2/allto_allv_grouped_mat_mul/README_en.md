# AlltoAllvGroupedMatMul

English | [简体中文](./README.md)

## Product Support

| Product                                                         | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950DT products</term>                             |    √     |
| <term>Atlas A3 products</term>     |    √     |
| <term>Atlas A2 products</term> |    ×     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                             |    ×     |
| <term>Atlas training products</term>                              |    ×     |

## Description

- **Operator function**: Fuses routed-expert AlltoAllv, Permute, and GroupedMatMul, and achieves parallel fusion with the shared-expert MatMul. **Communication is performed before computation.**

- **Computation formula**:
    - Routed experts:

    $$
    ataOut = AlltoAllv(gmmX) \\
    permuteOut = Permute(ataOut) \\
    gmmY = permuteOut \times gmmWeight
    $$

    - Shared experts:

    $$
    mmY = mmX \times mmWeight
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
        <td>After AlltoAllv communication and Permute operations are performed on this input, the result is used as the left matrix of the GroupedMatMul computation.</td>
        <td>FLOAT16, BFLOAT16</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>gmmWeight</td>
        <td>Input</td>
        <td>Right matrix of the GroupedMatMul computation.</td>
        <td>The same as gmmX</td>
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
        <td>The same as gmmX</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>mmWeightOptional</td>
        <td>Input</td>
        <td>Optional input. Right matrix in the shared-expert MatMul computation.</td>
        <td>The same as gmmX</td>
        <td>ND</td>
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
        <td>Size of the ep communicator.</td>
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
        <td>Indicates whether the right matrix of GroupedMatMul needs to be transposed.</td>
        <td>BOOL</td>
        <td>-</td>
    </tr>
    <tr>
        <td>transMmWeight</td>
        <td>Input</td>
        <td>Indicates whether the right matrix of the shared-expert MatMul needs to be transposed.</td>
        <td>BOOL</td>
        <td>-</td>
    </tr>
    <tr>
        <td>permuteOutFlag</td>
        <td>Input</td>
        <td>Indicates whether permuteOutOptional needs to be output.</td>
        <td>BOOL</td>
        <td>-</td>
    </tr>
    <tr>
        <td>gmmY</td>
        <td>Output</td>
        <td>Output of the routed-expert computation.</td>
        <td>The same as gmmX</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>mmYOptional</td>
        <td>Output</td>
        <td>Output of the shared-expert computation.</td>
        <td>The same as mmXOptional</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>permuteOutOptional</td>
        <td>Output</td>
        <td>Output after permute.</td>
        <td>The same as gmmX</td>
        <td>ND</td>
    </tr>
    </tbody></table>

## Constraints

- Communication engine constraints:
  - <term>Atlas A3 products</term>: Support AICPU communication.
  - <term>Ascend 950DT products</term>: Support CCU communication and AICPU communication. CCU supports interconnection only within the UB domain of a single machine, while AI CPU supports cross-machine interconnection within the UB domain.

- Deterministic computation:
  - aclnnAlltoAllvGroupedMatMul uses a deterministic implementation by default.

- Variables used in shapes in the parameter description:
  - BSK: Number of tokens sent by the local card. It is the sum of the sendCounts parameter. The value range is (0, 52428800).
  - H1: Hidden layer size (hidden size) of the routed experts. The value range is (0, 65536).
  - H2: Hidden layer size (hidden size) of the shared experts. The value range is (0, 12288].
  - e: Number of experts on a single card. e<=32, and e * epWorldSize supports a maximum of 256. This does not apply to the AIV mode. For details, see the additional AIV constraints below.
  - N1: head_num of the routed experts. The value range is (0, 65536).
  - N2: head_num of the shared experts. The value range is (0, 65536).
  - BS: batch sequence size.
  - K: Number of selected TopK experts. The range of K is [2, 8].
  - A: Number of tokens received by the local card. It is the sum of the recvCounts parameter.
  - The sum of the A parameter of all cards in the ep communicator equals the sum of the BSK parameter of all cards.

- Current reserved implementation of the AIV mode:
  - For Atlas A2, epWorldSize supports 2, 4, and 8; for Atlas A3, epWorldSize supports 2, 4, 8, 16, 32, 64, and 128. gmmX, gmmWeight, and gmmY must be of the same FLOAT16 or BFLOAT16 type and in ND format.
  - In the AIV mode, the value range of BSK/A is [1, 5000000], and the value range of H1/N1 is [1, 65535]. The number of local experts e does not exceed 512, and `epWorldSize * e` does not exceed 1024.
  - The interface attributes of sendCounts and recvCounts are still INT64 direct count arrays, laid out as `[rank][localExpert]`. The Host side verifies that each item is non-negative and does not exceed the corresponding local card input/output M, and verifies that the sum of sendCounts equals BSK and the sum of recvCounts equals A.

- <term>Atlas A3 products</term>: Performance degradation may occur when the communication volume of a single card is less than 2MB.

## Invocation

- <term>Atlas A3 products</term> and <term>Atlas A2 products</term>:

| Invocation Method | Sample Code                                  | Description                                                     |
| :--------: | :----------------------------------------: | :-------------------------------------------------------: |
| aclnn API | [test_aclnn_allto_allv_grouped_mat_mul.cpp](./examples/test_aclnn_allto_allv_grouped_mat_mul.cpp). | Invokes the allto_allv_grouped_mat_mul operator through the [aclnnAlltoAllvGroupedMatMul](docs/aclnnAlltoAllvGroupedMatMul_en.md) API. |

- <term>Ascend 950DT products</term>:

| Invocation Method | Sample Code                                  | Description                                                     |
| :--------: | :----------------------------------------: | :-------------------------------------------------------: |
| aclnn API | [test_aclnn_allto_allv_grouped_mat_mul.cpp](./examples/arch35/test_aclnn_allto_allv_grouped_mat_mul.cpp). | Invokes the allto_allv_grouped_mat_mul operator through the [aclnnAlltoAllvGroupedMatMul](docs/aclnnAlltoAllvGroupedMatMul_en.md) API. |
