# GroupedMatMulAlltoAllv

English | [简体中文](./README.md)

## Product Support

| Product                                                         | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950DT products</term>                             |    √     |
| <term>Atlas A3 products</term>     |    √     |
| <term>Atlas A2 products</term> |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                               |    ×     |
| <term>Atlas training products</term>                              |    ×     |

## Description

- Operator function: Fuses routed expert GroupedMatMul, Unpermute, and AlltoAllv, and fuses them in parallel with the shared expert MatMul. **Compute first, then communicate.**

- Calculation formula:
  - Routed experts:

    $$
    gmmY = gmmX \times gmmWeight \\
    unpermuteOut = Unpermute(gmmY) \\
    y = AlltoAllv(unpermuteOut)
    $$

  - Shared expert:

    $$
    mmY = mmX \times mmWeight
    $$

## Parameter Description

<table style="undefined;table-layout: fixed; width: 1392px"> <colgroup>
 <col style="width: 120px">
 <col style="width: 120px">
 <col style="width: 160px">
 <col style="width: 150px">
 <col style="width: 80px">
 </colgroup>
 <thead>
  <tr>
   <th>Parameter Name</th>
   <th>Input/Output</th>
   <th>Description</th>
   <th>Data Type</th>
   <th>Data Format</th>
  </tr></thead>
 <tbody>
  <tr>
   <td>gmmX</td>
   <td>Input</td>
   <td>This input undergoes AlltoAllv communication, and the result after communication serves as the left matrix of the GroupedMatMul calculation. 2 dimensions are supported, with shape (A, H1).</td>
   <td>FLOAT16, BFLOAT16</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>gmmWeight</td>
   <td>Input</td>
   <td>Right matrix of the GroupedMatMul calculation. The data type is the same as that of gmmX. 3 dimensions are supported, with shape (e, H1, N1).</td>
   <td>FLOAT16, BFLOAT16</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>sendCountsTensorOptional</td>
   <td>Input</td>
    <td>Optional input with shape (e * epWorldSize,). Not yet supported in the current version. Pass nullptr.</td>
   <td>INT32, INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>recvCountsTensorOptional</td>
   <td>Input</td>
    <td>Optional input with shape (e * epWorldSize,). Not yet supported in the current version. Pass nullptr.</td>
   <td>INT32, INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>mmXOptional</td>
   <td>Input</td>
   <td>Optional input, the left matrix in the shared expert MatMul calculation. It must be passed together with mmWeightOptional, or both must be nullptr. The data type is the same as that of gmmX. 2 dimensions are supported, with shape (BS, H2).</td>
   <td>FLOAT16, BFLOAT16</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>mmWeightOptional</td>
   <td>Input</td>
   <td>Optional input, the right matrix in the shared expert MatMul calculation. It must be passed together with mmXOptional, or both must be nullptr. The data type is the same as that of gmmX. 2 dimensions are supported, with shape (H2, N2).</td>
   <td>FLOAT16, BFLOAT16</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>group</td>
   <td>Input</td>
   <td>Communicator for expert parallelism. The string length must be in (0, 128).</td>
   <td>STRING</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>epWorldSize</td>
   <td>Input</td>
   <td>Size of the EP communicator: <br><term>Atlas A2 products</term> support 2, 4, and 8; <br><term>Atlas A3 products</term> support 8, 16, 32, 64, and 128; <br><term>Ascend 950DT products</term> support 2, 4, 8, 16, 32, 64, 128, and 256.</td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>sendCounts</td>
   <td>Input</td>
   <td>Number of tokens sent to other cards. The data type is INT64, the value size is e * epWorldSize, the maximum is 1024 for AIV communication, and the maximum is 256 for other communication engines.</td>
   <td>aclIntArray* (element type INT64)</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>recvCounts</td>
   <td>Input</td>
   <td>Number of tokens received from other cards. The data type is INT64, the value size is e * epWorldSize, the maximum is 1024 for AIV communication, and the maximum is 256 for other communication engines.</td>
   <td>aclIntArray* (element type INT64)</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>transGmmWeight</td>
   <td>Input</td>
   <td>Whether gmmWeight needs to be transposed. true indicates that transposition is required, and false indicates that no transposition is performed.</td>
   <td>BOOL</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>transMmWeight</td>
   <td>Input</td>
   <td>Whether the shared expert mmWeightOptional needs to be transposed. true indicates that transposition is required, and false indicates that no transposition is performed.</td>
   <td>BOOL</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>y</td>
   <td>Output</td>
   <td>Final calculation result. The data type is the same as that of the input gmmX. 2 dimensions are supported, with shape (BSK, N1).</td>
   <td>FLOAT16, BFLOAT16</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>mmYOptional</td>
   <td>Output</td>
   <td>Output of the shared expert MatMul. The data type is the same as that of mmXOptional. 2 dimensions are supported, with shape (BS, N2). It is output only when both mmXOptional and mmWeightOptional are passed.</td>
   <td>FLOAT16, BFLOAT16</td>
   <td>ND</td>
  </tr>
 </tbody></table>

## Constraints

- Communication engine constraints:
  - <term>Atlas A2 products</term>: Only AIV communication is supported
  - <term>Atlas A3 products</term>: AI CPU communication and AIV communication are supported.
  - <term>Ascend 950DT products</term>: CCU communication and AI_CPU communication are supported. CCU supports interconnection only within the UB domain of a single machine, and AI_CPU supports UB domain interconnection across machines.

- Variables used in shapes in the parameter description:
  - BSK: Number of tokens received by the current card, which is the sum of the recvCounts parameter. The value range is (0, 52428800).
  - H1: Indicates the hidden size of the routed experts. The value range is (0, 65536).
  - H2: Indicates the hidden size of the shared expert. The value range is (0, 12288].
  - e: Indicates the number of experts on a single card. AIV communication requires e > 0 and e * epWorldSize supports a maximum of 1024; other communication engines require e<=32 and e * epWorldSize supports a maximum of 256.
  - N1: Indicates head_num of the routed experts. The value range is (0, 65536).
  - N2: Indicates head_num of the shared expert. The value range is (0, 65536).
  - BS: batch sequence size.
  - K: Indicates that TopK experts are selected. AIV communication of Atlas A3 products supports [2, 16], and other scenarios support [2, 8].
  - A: Number of tokens sent by the current card, which is the sum of the sendCounts parameter.
  - The sum of the A parameters of all cards in the EP communicator equals the sum of the BSK parameters of all cards.

- <term>Atlas A3 products</term>  : Performance degradation may occur when the communication volume of a single card is below 2MB.

- <term>Atlas A2 products</term>: Both A and BSK must be in the range [1, 5000000], and N1 must not exceed 32768. The `HCCL_BUFFSIZE` of each card in the communicator must be set based on the maximum sending volume and satisfy `HCCL_BUFFSIZE >= max(200, ceil(A * N1 * 2 / 1048576) + 21)`, in MiB. Each FLOAT16 and BFLOAT16 element occupies 2 bytes.

## Calling Instructions

| Calling Method  | Sample Code                                  | Description                                                     |
| :--------: | :----------------------------------------: | :-------------------------------------------------------: |
| aclnn API | [test_aclnn_grouped_mat_mul_allto_allv.cpp](./examples/test_aclnn_grouped_mat_mul_allto_allv.cpp) | Calls the grouped_mat_mul_allto_allv operator through the [aclnnGroupedMatMulAlltoAllv](docs/aclnnGroupedMatMulAlltoAllv_en.md) API. |
