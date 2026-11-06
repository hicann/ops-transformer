# aclnnBandwidthTest

## Product Support Status

| Product                                                      | Supported |
| :----------------------------------------------------------- | :-------: |
| <term>Ascend 950PR&950DT products</term>                     |    √     |

## Description

- **Interface function**: Implements the `AlltoAll Dispatch` communication bandwidth test in the MOE (Mixture of Experts) scenario. This operator tests the communication bandwidth performance between NPU devices. Each rank sends tokens to the corresponding destination rank based on `dst_rank_id`, distributing data across ranks.

    Execution flow (mode=0, full flow):
    1. SendToMoeExpert: Writes tokens to the communication window of the destination rank based on `dst_rank_id`
    2. SetStatus: Counts the tokens sent to each rank and writes the count to the status area to notify the peer
    3. WaitDispatch: Waits until all ranks complete data sending and write their status
    4. LocalWindowCopy: Reads the received data from the local window and copies the data to the output tensor

    Execution flow (mode=1, send only):
    Only SendToMoeExpert is executed for one-way bandwidth testing.

- Computation formula:

    ```
    y = AlltoAllV(x, dst_rank_id)
    receiveCnt = counts the number of tokens received from each rank
    ```

    Details:
    - Each token is distributed to the destination rank specified by `dst_rank_id`
    - Data is transferred between ranks through the HCCL communication window
    - `receiveCnt` records the number of tokens received from each rank

## Function Prototype

Each operator has two-phase APIs. You must first call the "aclnnBandwidthTestGetWorkspaceSize" API to obtain the workspace size required for the computation and the executor that contains the operator computation process, and then call the "aclnnBandwidthTest" API to perform the computation.

```cpp
aclnnStatus aclnnBandwidthTestGetWorkspaceSize(
    const aclTensor* x,
    const aclTensor* dstRankId,
    const char*      group,
    int64_t          worldSize,
    int64_t          maxBs,
    int64_t          mode,
    const char*      commAlg,
    int64_t          aivNum,
    aclTensor*       y,
    aclTensor*       receiveCnt,
    uint64_t*        workspaceSize,
    aclOpExecutor**  executor)
```

```cpp
aclnnStatus aclnnBandwidthTest(
    void            *workspace,
    uint64_t        workspaceSize,
    aclOpExecutor   *executor,
    aclrtStream     stream)
```

## aclnnBandwidthTestGetWorkspaceSize

- **Parameter Description**

    <table style="undefined;table-layout: fixed; width: 1200px"> <colgroup>
    <col style="width: 120px">
    <col style="width: 100px">
    <col style="width: 200px">
    <col style="width: 280px">
    <col style="width: 150px">
    <col style="width: 80px">
    <col style="width: 150px">
    <col style="width: 120px">
    </colgroup>
    <thead>
    <tr>
    <th>Parameter</th>
    <th>Input/Output</th>
    <th>Description</th>
    <th>Usage</th>
    <th>Data Type</th>
    <th>Data Format</th>
    <th>Dimensions (shape)</th>
    <th>Non-contiguous Tensor</th>
    </tr>
    </thead>
    <tbody>
    <tr>
    <td>x</td>
    <td>Input</td>
    <td>Token data sent by the current device.</td>
    <td>Must be a 2D tensor.</td>
    <td>FLOAT16</td>
    <td>ND</td>
    <td><code>(BS, H)</code> (BS=batch size, H=hidden size)</td>
    <td>-</td>
    </tr>
    <tr>
    <td>dstRankId</td>
    <td>Input</td>
    <td>Destination rank index of each token.</td>
    <td>Must be a 1D tensor. Each element indicates the destination rank of the corresponding token.</td>
    <td>INT32</td>
    <td>ND</td>
    <td><code>(BS,)</code></td>
    <td>-</td>
    </tr>
    <tr>
    <td>group</td>
    <td>Input</td>
    <td>Communicator name.</td>
    <td>The string length ranges from <code>[1, 128)</code>.</td>
    <td>STRING</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>worldSize</td>
    <td>Input</td>
    <td>Communicator size (total number of ranks).</td>
    <td>Value range: <code>[2, 768]</code>.</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>maxBs</td>
    <td>Input</td>
    <td>Maximum batch size.</td>
    <td>-</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>mode</td>
    <td>Input</td>
    <td>Execution mode.</td>
    <td>Value range: <ul><li>0: precision mode (full flow: send + synchronization + receive)</li><li>1: send-only mode (send only, used for one-way bandwidth testing)</li></ul></td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>commAlg</td>
    <td>Input</td>
    <td>Communication algorithm.</td>
    <td>Optional parameter. Default value: empty string.</td>
    <td>STRING</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>aivNum</td>
    <td>Input</td>
    <td>Number of AIV cores.</td>
    <td>Optional parameter. Default value: -1 (obtained automatically).</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>y</td>
    <td>Output</td>
    <td>Output token data.</td>
    <td>Must be a 2D tensor. The data type is the same as that of input x.</td>
    <td>FLOAT16</td>
    <td>ND</td>
    <td><code>(totalReceiveCnt, H)</code> (totalReceiveCnt is the total number of tokens received from all ranks)</td>
    <td>√</td>
    </tr>
    <tr>
    <td>receiveCnt</td>
    <td>Output</td>
    <td>Number of tokens received from each rank.</td>
    <td>Must be a 1D tensor.</td>
    <td>INT32</td>
    <td>ND</td>
    <td><code>(worldSize,)</code></td>
    <td>√</td>
    </tr>
    <tr>
    <td>workspaceSize</td>
    <td>Output</td>
    <td>Returns the workspace size to be requested on the device.</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>executor</td>
    <td>Output</td>
    <td>Returns the op executor that contains the operator computation process.</td>
    <td>-</td>
    <td>aclOpExecutor*</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    </tbody>
    </table>

- **Return Values**

    aclnnStatus: Returns a status code. For details, see [aclnn Return Codes](../../../../docs/zh/context/aclnn_return_code_en.md).

    The first-phase API performs input parameter validation. An error is reported in the following scenarios:

    <table style="undefined;table-layout: fixed; width: 800px"><colgroup>
    <col style="width: 200px">
    <col style="width: 100px">
    <col style="width: 500px">
    </colgroup>
    <thead>
    <tr>
    <th>Return Value</th>
    <th>Error Code</th>
    <th>Description</th>
    </tr>
    </thead>
    <tbody>
    <tr>
    <td>ACLNN_ERR_PARAM_NULLPTR</td>
    <td>161001</td>
    <td>The required input or output parameter tensor is a null pointer.</td>
    </tr>
    <tr>
    <td>ACLNN_ERR_PARAM_INVALID</td>
    <td>161002</td>
    <td>The data types of inputs or outputs are not in the supported range.</td>
    </tr>
    <tr>
    <td rowspan="2">ACLNN_ERR_INNER_TILING_ERROR</td>
    <td rowspan="2">561002</td>
    <td>The shapes of inputs or outputs are not in the supported range.</td>
    </tr>
    <tr>
        <td>The value of a parameter is not in the supported range (such as worldSize, maxBs, or mode).</td>
    </tr>
    </tbody>
    </table>

## aclnnBandwidthTest

- **Parameter Description**

    <table style="undefined;table-layout: fixed; width: 800px"><colgroup>
    <col style="width: 150px">
    <col style="width: 100px">
    <col style="width: 550px">
    </colgroup>
    <thead>
    <tr>
    <th>Parameter</th>
    <th>Input/Output</th>
    <th>Description</th>
    </tr>
    </thead>
    <tbody>
    <tr>
    <td>workspace</td>
    <td>Input</td>
    <td>Memory address of the workspace requested on the device.</td>
    </tr>
    <tr>
    <td>workspaceSize</td>
    <td>Input</td>
    <td>Size of the workspace requested on the device, obtained by the first-phase API <code>aclnnBandwidthTestGetWorkspaceSize</code>.</td>
    </tr>
    <tr>
    <td>executor</td>
    <td>Input</td>
    <td>Op executor that contains the operator computation process.</td>
    </tr>
    <tr>
    <td>stream</td>
    <td>Input</td>
    <td>Stream on which the task is executed.</td>
    </tr>
    </tbody>
    </table>

- **Return Values**

    Returns an aclnnStatus status code. For details, see [aclnn Return Codes](../../../../docs/zh/context/aclnn_return_code_en.md).

## Constraints

- **Deterministic computation**:
  - aclnnBandwidthTest uses a deterministic implementation by default.

- **Driver constraints**:
  - The driver versions of all nodes in the operator communicator must be the same.

- **Shape variable constraints**:

    | Variable     | Definition and Value Range                                                     |
    | :----------- | :----------------------------------------------------------------------------- |
    | worldSize    | Indicates the total number of ranks in the communicator. Value range: <code>[2, 768]</code>. |

- **Data type constraints**:
    - The data types of input `x` and output `y` must be the same. FLOAT16, BFLOAT16, and FLOAT32 are supported.
    - The data type of input `dstRankId` must be INT32.
    - The data type of output `receiveCnt` must be INT32.

- **Parameter constraints**:
    - The value of each element in `dstRankId` must be in the range <code>[0, worldSize)</code>.
    - The value of `mode` must be 0 or 1.
    - The length of the `group` string must be less than 128.

- **Environment variable constraints**:
    - Before calling this API, check whether the value of the `HCCL_BUFFSIZE` environment variable is appropriate. This environment variable indicates the memory size occupied by a single communicator, in MB.
    - The value must be <code>≥ 2 *(maxBs* worldSize *H* sizeof(dtype) + 2MB)</code>.

- **Communicator usage constraints**:
    - The nodes in a communicator must be within one supernode. Cross-supernode communication is not supported.

- **Other constraints**:
    - "×" in the formula indicates multiplication.
