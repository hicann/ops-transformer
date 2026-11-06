# FFNToAttention

English | [简体中文](./README.md)

## Product Support

| Product                                                         | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                             |    ×     |
| <term>Atlas A3 products</term>       |    √     |
| <term>Atlas A2 products</term> |    ×     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                               |    ×     |
| <term>Atlas training products</term>                              |    ×     |

## Description

- Operator function: Sends data on the FFN node to the Attention node.

## Parameter Description

<table style="table-layout: auto; width: 100%">
  <thead>
    <tr>
      <th style="white-space: nowrap">Parameter Name</th>
      <th style="white-space: nowrap">Input/Output/Attribute</th>
      <th style="white-space: nowrap">Description</th>
      <th style="white-space: nowrap">Data Type</th>
      <th style="white-space: nowrap">Data Format</th>
    </tr>
  </thead>
  <tbody>
    <tr>
    <td>x</td>
    <td>Input</td>
    <td>Token data sent by the current card. This is a 2D Tensor with shape <code>(Y, H)</code> (H=hidden size).</td>
    <td>FLOAT16, BFLOAT16</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>sessionIds</td>
    <td>Input</td>
    <td>Index of the Attention Worker node for each token. This is a 1D Tensor with shape <code>(Y, )</code>. The value range of sessionIds is [0, attnRankNum-1]</td>
    <td>INT32</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>microBatchIds</td>
    <td>Input</td>
    <td>microBatch index of each token. This is a 1D Tensor with shape <code>(Y, )</code>. The value range of microBatchIds is [0, MicroBatchNum-1]</td>
    <td>INT32</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>tokenIds</td>
    <td>Input</td>
    <td>Token index of each token in the microBatch. This is a 1D Tensor with shape <code>(Y, )</code>. The value range of tokenIds is [0, Bs-1]</td>
    <td>INT32</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>expertOffsets</td>
    <td>Input</td>
    <td>Index of each token in PerTokenExpertNum of tokenInfoTableShape. This is a 1D Tensor with shape <code>(Y, )</code>. The value range of expertOffsets is [0, ExpertNumPerToken-1].</td>
    <td>INT32</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>actualTokenNum</td>
    <td>Input</td>
    <td>Total number of tokens actually sent by the current card. This is a 1D Tensor with shape <code>(1, )</code>. The value of actualTokenNum is in [0, Y].</td>
    <td>INT64</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>attnRankTable</td>
    <td>Optional input</td>
    <td>Maps each Attention Worker to its card ID.
    <br>Attention Workers must be deployed consecutively starting from card 0.
    <br>If a null pointer is passed, the default policy is used: the ID of each card is used as the ID of the corresponding Attention Worker, with the value range [0, attnRankNum-1].</td>
    <td>INT32</td>
    <td>ND (non-contiguous Tensors supported)</td>
    </tr>
    <tr>
    <td>group</td>
    <td>Attribute</td>
    <td>Communicator name (expert parallelism). The string length is [1, 128).</td>
    <td>STRING</td>
    <td>-</td>
    </tr>
    <tr>
    <td>worldSize</td>
    <td>Attribute</td>
    <td>Communicator size. The value range is [2, 768].</td>
    <td>INT64</td>
    <td>-</td>
    </tr>
    <tr>
    <td>tokenInfoTableShape</td>
    <td>Attribute</td>
    <td>Size of the token information list, including the number of microBatches (MicroBatchNum), the batch size (Bs), and the number of experts per token (ExpertNumPerToken).</td>
    <td>INT32</td>
    <td>-</td>
    </tr>
    <tr>
    <td>tokenDataShape</td>
    <td>Attribute</td>
    <td>Size of the token data list, including the number of microBatches (MicroBatchNum), the batch size (Bs), the number of experts per token (ExpertNumPerToken), and the token and scale length (HS).</td>
    <td>INT32</td>
    <td>-</td>
    </tr>

  </tbody>
</table>

## Constraints

- The values of the `group`, `worldSize`, `tokenInfoTableShape`, and `tokenDataShape` parameters and `HCCL_BUFFSIZE` used during operator calling must be consistent across all cards and across different layers in the network.

- Shape format description in the parameter description:
    - `Y`: Indicates the maximum number of tokens that the current card needs to dispatch.
    - `BS`: Indicates the number of tokens sent on each Attention node. The value range is 0 < `BS` ≤ 512.
    - `H`: Indicates the hidden size of the hidden layer. The value range is 1024 ≤ `H` ≤ 8192.
    - `HS`: Indicates the combined hidden and scale size of the hidden layer. The value range is 1152 ≤ `HS` ≤ 8320.
    - `ffnRankNum`: Indicates that ffnRankNum cards are selected as FFnWorkers. The value range is 0 < `ffnRankNum` < `worldSize`.
    - `attnRankNum`: Indicates that attnRankNum cards are selected as AttnWorkers. The value range is 0 < `attnRankNum` < `worldSize`.
    - `sharedExpertNum`: Indicates the number of shared experts (one shared expert can be replicated and deployed on multiple ffnRank cards). The value range is 0 ≤ `sharedExpertNum` ≤ 4.

- Communicator usage constraints:
    - No other operators are allowed in the communicator of the FFNToAttention operator.

- <term>Atlas A3 products</term>  :
    - In this scenario, a single card contains dual DIEs (referred to as "grains" or "bare dies" for short). Therefore, "the current card" in the parameter description refers to a single die.

## Calling Instructions

| Calling Method  | Sample Code                                  | Description                                                     |
| :--------: | :----------------------------------------: | :-------------------------------------------------------: |
| aclnn API | [test_aclnn_ffn_to_attention.cpp](./examples/test_aclnn_ffn_to_attention.cpp) | Calls the FFNToAttention operator through the [aclnnFFNToAttention](docs/aclnnFFNToAttention_en.md) API. |
