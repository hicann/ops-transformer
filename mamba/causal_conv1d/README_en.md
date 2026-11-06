# CausalConv1d

English | [简体中文](./README.md)

## Product Support Status

|Product      | Supported |
|:----------------------------|:-----------:|
|<term>Ascend 950PR&950DT products</term>|      √     |
|<term>Atlas A3 products</term>|      x     |
|<term>Atlas A2 products</term>|      x     |
|<term>Atlas 200I/500 A2 inference products</term>|      ×     |
|<term>Atlas inference products</term>|      ×     |
|<term>Atlas training products</term>|      ×     |

## Functional Description

- Operator function: Performs causal 1D convolution (Causal Conv1d) computation, and supports two running modes: forward computation (prefill / chunk-prefill) and state update (decode / update). The mode is automatically inferred from the input shapes.

- Computation formula:

  Causal Conv1d is a causal 1D convolution operator, commonly used in sequence modeling. At each time step $t$, the convolution output $y_t$ is computed from the current input $x_t$, the convolution weight $w$, and the historical state.

  $$
  y_t = \text{Activation}\left(\sum_{j=0}^{W-1} w_j \cdot x_{t-j} + b\right)
  $$

  where $W$ is the convolution kernel width (2, 3, and 4 are supported), $w_j$ is the convolution weight, $b$ is the bias (optional), and $\text{Activation}$ is the activation function (optional, SiLU). When `activation_mode="none"`, no activation function is used; when `activation_mode="silu"`, the SiLU activation function is used.

   The operator also maintains the convolution state `conv_states`, which caches historical inputs during incremental inference to enable efficient state updates.

## Parameter Description

<table style="undefined;table-layout: fixed; width: 900px"><colgroup>
<col style="width: 180px">
<col style="width: 120px">
<col style="width: 200px">
<col style="width: 300px">
<col style="width: 100px">
</colgroup>
<thead>
  <tr>
    <th>Parameter</th>
    <th>Input/Output/Attribute</th>
    <th>Description</th>
    <th>Data Type</th>
    <th>Data Format</th>
  </tr></thead>
<tbody>
  <tr>
    <td>x</td>
    <td>Input</td>
    <td>Input sequence, which is x in the formula.</td>
    <td>BFLOAT16, FLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>weight</td>
    <td>Input</td>
    <td>Convolution weight, which is w in the formula.</td>
    <td>Same as x</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>conv_states</td>
    <td>Input/Output</td>
    <td>
      <ul>
        <li>Convolution state, which caches historical inputs for causal convolution computation.</li>
        <li>Updated in place after the computation of each sequence is complete.</li>
      </ul>
    </td>
    <td>Same as x</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>bias</td>
    <td>Optional input</td>
    <td>Bias, which is b in the formula. If not provided, the default value is 0.</td>
    <td>Same as x</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>query_start_loc</td>
    <td>Optional input</td>
    <td>Start position indexes of sequences, recording the start position of each sequence in the concatenated tensor x. queryStartLoc[0] must be 0, and queryStartLoc[-1] must be cu_seq_len.</td>
    <td>INT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>cache_indices</td>
    <td>Optional input</td>
    <td>Cache indexes, specifying the index of the cached state of each sequence in conv_states. If not passed, the identity mapping is used.</td>
    <td>INT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>initial_state_mode</td>
    <td>Optional input</td>
    <td>Initial state flag. 1: use the cached conv_states as history; 0: initialize the history with zeros (cold start).</td>
    <td>INT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>num_accepted_tokens</td>
    <td>Optional input</td>
    <td>Number of tokens accepted for each sequence, used for speculative decoding.</td>
    <td>INT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>activation_mode</td>
    <td>Optional attribute</td>
    <td>
      <ul>
        <li>Activation function type. "silu": use the SiLU activation function; "none": do not use an activation function.</li>
        <li>The default value is "silu".</li>
      </ul>
    </td>
    <td>STR</td>
    <td>-</td>
  </tr>
  <tr>
    <td>null_block_id</td>
    <td>Optional attribute</td>
    <td>
      <ul>
        <li>Marker ID of invalid cache slots, used to skip sequences that do not need computation.</li>
        <li>The default value is 0.</li>
      </ul>
    </td>
    <td>INT64</td>
    <td>-</td>
  </tr>
  <tr>
    <td>y</td>
    <td>Output</td>
    <td>Convolution output, which is y in the formula.</td>
    <td>Same as x</td>
    <td>ND</td>
  </tr>
</tbody>
</table>

## Constraints

- The shape sizes of the input tensors must satisfy certain constraints. For details, see [aclnnCausalConv1dFn](./docs/aclnnCausalConv1dFn.md) and [aclnnCausalConv1dUpdate](./docs/aclnnCausalConv1dUpdate.md).

## Invocation Description

| Invocation Method | Sample Code                                                  | Description                                                  |
| --------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| aclnn API (prefill) | [test_aclnn_causal_conv1d_fn.cpp](./examples/test_aclnn_causal_conv1d_fn.cpp) | Calls the prefill mode through [aclnnCausalConv1dFn](./docs/aclnnCausalConv1dFn.md) |
| aclnn API (update) | [test_aclnn_causal_conv1d_update.cpp](./examples/test_aclnn_causal_conv1d_update.cpp) | Calls the update mode through [aclnnCausalConv1dUpdate](./docs/aclnnCausalConv1dUpdate.md) |
| PyTorch API | - | Calls the causal_conv1d_fn operator through the [causal_conv1d_fn](docs/torchapi_causal_conv1d_fn_en.md) API. |
| PyTorch API | - | Calls the causal_conv1d_update operator through the [causal_conv1d_update](docs/torchapi_causal_conv1d_update_en.md) API. |
