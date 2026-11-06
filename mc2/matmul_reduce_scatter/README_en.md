# MatmulReduceScatter

English | [简体中文](./README.md)

## Product Support

| Product | Supported |
| :-- | :------: |
| <term>Ascend 950PR&950DT products</term> |    √     |
| <term>Atlas A3 products</term> |    √     |
| <term>Atlas A2 products</term> |    √     |
| <term>Atlas 200I/500 A2 inference products</term> |    ×     |
| <term>Atlas inference products</term> |    ×     |
| <term>Atlas training products</term> |    ×     |

**Note:** When using this API, ensure that both the driver and firmware package and the CANN package are the matching 8.0.RC2 version or a matching later version. Otherwise, errors such as Bus Error are raised.

## Function Description

- Interface function: This API fuses `Matmul` computation and `ReduceScatter` communication, with computation performed before communication. The fused operator implements pipelined parallelism of computation and communication internally.

- Calculation formula:

    $$
    output=ReduceScatter(x1@x2+bias)
    $$

    - x1 refers to the left matrix of the Matmul computation.
    - x2 refers to the right matrix of the Matmul computation.
    - bias refers to the bias of the Matmul computation.

## Function Prototype

Each operator has a two-phase API. You must first call the "aclnnMatmulReduceScatterGetWorkspaceSize" API to obtain the workspace size required for the computation and the executor that contains the operator computation process, and then call the "aclnnMatmulReduceScatter" API to perform the computation.

```cpp
aclnnStatus aclnnMatmulReduceScatterGetWorkspaceSize(
    const aclTensor* x1,
    const aclTensor* x2,
    const aclTensor* bias,
    const char*      group,
    const char*      reduceOp,
    int64_t          commTurn,
    int64_t          streamMode,
    const aclTensor* output,
    uint64_t*        workspaceSize,
    aclOpExecutor**  executor)
```

```cpp
aclnnStatus aclnnMatmulReduceScatter(
    void*           workspace,
    uint64_t        workspaceSize,
    aclOpExecutor*  executor,
    aclrtStream     stream)
```

## aclnnMatmulReduceScatterGetWorkspaceSize

- **Parameter Description**

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
      <th>Parameter Name</th>
      <th>Input/Output</th>
        <th>Description</th>
        <th>Usage</th>
        <th>Data Type</th>
        <th>Data Format</th>
        <th>Dimensions (shape)</th>
        <th>Non-contiguous Tensor</th>
    </tr></thead>
    <tbody>
    <tr>
        <td>x1</td>
        <td>Input</td>
        <td>Left matrix input of the Matmul computation, that is, x1 in the calculation formula.</td>
        <td>
            <ul>
                <li>The shape is (m, k).</li>
                <li>Empty tensors are supported.</li>
            </ul>
        </td>
        <td>FLOAT16, BFLOAT16</td>
        <td>ND</td>
        <td>2</td>
        <td>×</td>
    </tr>
    <tr>
        <td>x2</td>
        <td>Input</td>
        <td>Right matrix input of the Matmul computation, that is, x2 in the calculation formula.</td>
        <td>
            <ul>
                <li>The shape is (k, n) in the non-transposed scenario and (n, k) in the transposed scenario.</li>
                <li>Empty tensors are supported.</li>
                <li>The data type is the same as that of x1.</li>
                <li>The current version supports only 2D inputs, in both transposed and non-transposed scenarios.</li>
                <li>Only the <a href="../../docs/zh/context/non_contiguous_tensor_en.md">non-contiguous tensor</a> in the two-axis transposition case is supported; non-contiguous tensors in other scenarios are not supported.</li>
            </ul>
        </td>
        <td>FLOAT16, BFLOAT16</td>
        <td>ND</td>
        <td>2</td>
        <td>√ (applies only to the transposed scenario)</td>
    </tr>
    <tr>
        <td>bias</td>
        <td>Input</td>
        <td>Bias of the Matmul computation, that is, bias in the calculation formula.</td>
        <td>
            <ul>
                <li>The shape is (n).</li>
                <li>Passing a null pointer is supported.</li>
                <li><term>Atlas A2 products</term> and <term>Atlas A3 products</term>: Passing a non-zero value is currently not supported.</li>
                <li><term>Ascend 950PR&950DT products</term>: Passing a non-zero value is supported.</li>
            </ul>
        </td>
        <td>FLOAT16, BFLOAT16</td>
        <td>ND</td>
        <td>1</td>
        <td>×</td>
    </tr>
    <tr>
        <td>group</td>
        <td>Input</td>
        <td>Communicator name.</td>
        <td>Obtained through the Hccl API "extern HcclResult HcclGetCommName(HcclComm comm, char* commName);", where commName is the group.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>reduceOp</td>
        <td>Input</td>
        <td>reduce operation type.</td>
        <td>The current version supports only "sum".</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>commTurn</td>
        <td>Input</td>
        <td>Number of communication data splits, that is, total data volume divided by the volume of a single communication.</td>
        <td>The current version supports only 0.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>streamMode</td>
        <td>Input</td>
        <td>Enumeration of the stream mode.</td>
        <td>streamMode: stream processing mode after a task fails. The value range is as follows:
            1 indicates ACL_STOP_ON_FAILURE. After a task fails, subsequent tasks stop being executed. This mode is the only value supported in the current version.
    </td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>output</td>
        <td>Output</td>
        <td>Result of ReduceScatter communication and the MatMul computation, that is, output in the calculation formula.</td>
        <td>
            <ul>
                <li>The shape is (m/rank_size, n).</li>
                <li>Empty tensors are supported.</li>
                <li>The data type is the same as that of x1.</li>
            </ul>
        </td>
        <td>FLOAT16, BFLOAT16</td>
        <td>ND</td>
        <td>2</td>
        <td>×</td>
    </tr>
    <tr>
        <td>workspaceSize</td>
        <td>Output</td>
        <td>Returns the size of the workspace to be requested on the Device.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>executor</td>
        <td>Output</td>
        <td>Returns the op executor, which contains the operator computation process.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    </tbody></table>

- **Return Value**

    Returns an aclnnStatus status code. For details, see [aclnn Return Codes](../../docs/zh/context/aclnn_return_code_en.md).

    The first-phase API validates input parameters and reports an error in the following scenarios:

    <table style="undefined;table-layout: fixed; width: 1166px"> <colgroup>
    <col style="width: 267px">
    <col style="width: 124px">
    <col style="width: 775px">
    </colgroup>
    <thead>
    <tr>
        <th>Return Value</th>
        <th>Error Code</th>
        <th>Description</th>
    </tr></thead>
    <tbody>
    <tr>
        <td>ACLNN_ERR_PARAM_NULLPTR</td>
        <td>161001</td>
        <td>The passed x1, x2, or output is a null pointer.</td>
    </tr>
    <tr>
        <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
        <td rowspan="3">161002</td>
        <td>The data type of x1, x2, bias, or output is not in the supported range.</td>
    </tr>
    <tr>
        <td>streamMode is not in the valid range.</td>
    </tr>
    <tr>
        <td>The shape of x1, x2, or output does not meet the constraint requirements.</td>
    </tr>
    </tbody></table>

## aclnnMatmulReduceScatter

- **Parameter Description**
    <table style="undefined;table-layout: fixed; width: 1166px"> <colgroup>
    <col style="width: 173px">
    <col style="width: 133px">
    <col style="width: 860px">
    </colgroup>
    <thead>
    <tr>
    <th>Parameter Name</th>
    <th>Input/Output</th>
    <th>Description</th>
    </tr></thead>
    <tbody>
    <tr>
        <td>workspace</td>
        <td>Input</td>
        <td>Memory address of the workspace requested on the Device.</td>
    </tr>
    <tr>
        <td>workspaceSize</td>
        <td>Input</td>
        <td>Size of the workspace requested on the Device, obtained from the first-phase API <code>aclnnMatmulReduceScatterGetWorkspaceSize</code>.</td>
    </tr>
    <tr>
        <td>executor</td>
        <td>Input</td>
        <td>Op executor, which contains the operator computation process.</td>
    </tr>
    <tr>
        <td>stream</td>
        <td>Input</td>
        <td>Stream on which tasks are executed.</td>
    </tr>
    </tbody></table>

- **Return Value:**

    Returns an aclnnStatus status code. For details, see [aclnn Return Codes](../../docs/zh/context/aclnn_return_code_en.md).

## Constraints

- Communication engine constraints:
  - <term>Atlas A2 products</term>: Supports only AICPU communication.
  - <term>Atlas A3 products</term>: Supports only AICPU communication.
  - <term>Ascend 950PR&950DT products</term>: Supports only AICPU communication.

- Description of variables involved in the shapes in the parameter description:
  - m is an integer multiple of the number of chips rank_size.
  - k is in the range [256, 65535).
  - For the empty tensor scenarios supported by x1/x2, m and n can be 0, k cannot be 0, and the following conditions must be met:
    - m is 0, k is not 0, and n is not 0;
    - m is not 0, k is not 0, and n is 0;
    - m is 0, k is not 0, and n is 0.
  - rank_size is the number of chips.
- x2 supports transposed and non-transposed scenarios; x1 supports only the non-transposed scenario.
- The data types of computation inputs x1 and x2 must be the same as the data type of computation output `output`.
- <term>Atlas A2 products</term>:
  - Supports 2, 4, and 8 chips, and only the all mesh networking over hccs links.
  - Communication-computation fused MC2 operators in a model support only the same communicator.
- <term>Atlas A3 products</term>:
  - Supports 2, 4, 8, 16, and 32 chips, and only the double ring networking over hccs links.
- <term>Ascend 950PR&950DT products</term>: Supports 2, 4, 8, 16, 32, and 64 chips, and only the all mesh networking over hccs links.
  - The total data volume of ReduceScatter collective communication cannot exceed 16*256MB. The total data volume is calculated as m* n * sizeof(output_dtype). Because shapes differ, the internal implementation of the operator may vary, and the actually supported total communication volume may be slightly smaller than this value.
  - aclnnMatmulReduceScatter is deterministic by default.

## Calling Instructions

| Calling Method | Sample Code | Description |
| :--------: | :----------------------------------------: | :-------------------------------------------------------: |
| aclnn API | [test_aclnn_matmul_reduce_scatter.cpp](./examples/test_aclnn_matmul_reduce_scatter.cpp) | Calls the matmul_reduce_scatter operator through the [aclnnMatmulReduceScatter](docs/aclnnMatmulReduceScatter_en.md) API. |
