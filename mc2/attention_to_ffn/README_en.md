# AttentionToFFN

English | [简体中文](./README.md)

## Product Support Status

| Product                                                         | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                             |    ×     |
| <term>Atlas A3 products</term>       |    √     |
| <term>Atlas A2 products</term> |    ×     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                               |    ×     |
| <term>Atlas training products</term>                              |    ×     |

## Function Description

- Operator function: Sends the data on the Attention node to the FFN node.

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
    <td style="white-space: nowrap">x</td>
    <td style="white-space: nowrap">Input</td>
    <td style="white-space: nowrap">Token data sent by the current card, a 3D tensor with shape <code>(X, Bs, H).</code></td>
    <td style="white-space: nowrap">FLOAT16, BFLOAT16</td>
    <td style="white-space: nowrap">ND</td>
    </tr>
    <tr>
    <td style="white-space: nowrap">sessionId</td>
    <td style="white-space: nowrap">Input</td>
    <td style="white-space: nowrap">ID of the current Attention Worker node, a 1D tensor with shape <code>(X, ).</code></td>
    <td style="white-space: nowrap">INT32</td>
    <td style="white-space: nowrap">ND</td>
    </tr>
    <tr>
    <td style="white-space: nowrap">microBatchId</td>
    <td style="white-space: nowrap">Input</td>
    <td style="white-space: nowrap">ID of the current microBatch, a 1D tensor with shape <code>(X, ).</code></td>
    <td style="white-space: nowrap">INT32</td>
    <td style="white-space: nowrap">ND</td>
    </tr>
    <tr>
    <td style="white-space: nowrap">layerId</td>
    <td style="white-space: nowrap">Input</td>
    <td style="white-space: nowrap">ID of the current model layer, a 1D tensor with shape <code>(X, ).</code></td>
    <td style="white-space: nowrap">INT32</td>
    <td style="white-space: nowrap">ND</td>
    </tr>
    <tr>
    <td style="white-space: nowrap">expertIds</td>
    <td style="white-space: nowrap">Input</td>
    <td style="white-space: nowrap">Indexes of the topK experts of each token in each micro batch group, a 3D tensor with shape <code>(X, Bs, K)</code>. The value range of expertIds is [0, moeExpertNum).</td>
    <td style="white-space: nowrap">INT32</td>
    <td style="white-space: nowrap">ND</td>
    </tr>
    <tr>
    <td style="white-space: nowrap">expertRankTable</td>
    <td style="white-space: nowrap">Input</td>
    <td style="white-space: nowrap">Mapping table from expert IDs to the expert deployment on FFN cards in each micro batch group (the caller must ensure that the values are correct), a 3D tensor with shape <code>(L, moeExpertNum + sharedExpertNum, M).</code></td>
    <td style="white-space: nowrap">INT32</td>
    <td style="white-space: nowrap">ND</td>
    </tr>
    <tr>
    <td style="white-space: nowrap">scales</td>
    <td style="white-space: nowrap">Input</td>
    <td style="white-space: nowrap">Quantization smoothing parameter of each expert. When quantMode is 0 (no quantization), a null pointer must be passed. When quantMode is 2 (dynamic quantization), a valid tensor or a null pointer can be passed. When passed, it is a 3D tensor with shape <code>(L, moeExpertNum + sharedExpertNum, H).</code></td>
    <td style="white-space: nowrap">FLOAT32</td>
    <td style="white-space: nowrap">ND</td>
    </tr>
    <tr>
    <td style="white-space: nowrap">activeMask</td>
    <td style="white-space: nowrap">Input</td>
    <td style="white-space: nowrap">Indicates whether a token participates in communication. Valid data or a null pointer can be passed. When valid data is passed, the true entries must be placed before the false entries; the shape is <code>(X, Bs).</code></td>
    <td style="white-space: nowrap">BOOL</td>
    <td style="white-space: nowrap">ND</td>
    </tr>
    <tr>
    <td style="white-space: nowrap">group</td>
    <td style="white-space: nowrap">Attribute</td>
    <td style="white-space: nowrap">Communicator name (expert parallelism). The string length is in the range [1, 128).</td>
    <td style="white-space: nowrap">STRING</td>
    <td style="white-space: nowrap">-</td>
    </tr>
    <tr>
    <td style="white-space: nowrap">worldSize</td>
    <td style="white-space: nowrap">Attribute</td>
    <td style="white-space: nowrap">Size of the communicator. Value range: [2, 768].</td>
    <td style="white-space: nowrap">INT64</td>
    <td style="white-space: nowrap">-</td>
    </tr>
    <tr>
    <td style="white-space: nowrap">ffnTokenInfoTableShape</td>
    <td style="white-space: nowrap">Attribute</td>
    <td style="white-space: nowrap">List of the shape sizes of the token information table on the FFN node. The length is 3.</td>
    <td style="white-space: nowrap">INT64</td>
    <td style="white-space: nowrap">-</td>
    </tr>
    <tr>
    <td style="white-space: nowrap">ffnTokenDataShape</td>
    <td style="white-space: nowrap">Attribute</td>
    <td style="white-space: nowrap">List of the shape sizes of the token data table on the FFN node. The length is 5.</td>
    <td style="white-space: nowrap">INT64</td>
    <td style="white-space: nowrap">-</td>
    </tr>
    <tr>
    <td style="white-space: nowrap">attnTokenInfoTableShape</td>
    <td style="white-space: nowrap">Attribute</td>
    <td style="white-space: nowrap">List of the shape sizes of the token information table on the Attention node. The length is 3.</td>
    <td style="white-space: nowrap">INT64</td>
    <td style="white-space: nowrap">-</td>
    </tr>
    <tr>
    <td style="white-space: nowrap">moeExpertNum</td>
    <td style="white-space: nowrap">Attribute</td>
    <td style="white-space: nowrap">Number of MoE experts. Value range: (0, 1024].</td>
    <td style="white-space: nowrap">INT64</td>
    <td style="white-space: nowrap">-</td>
    </tr>
    <tr>
    <td style="white-space: nowrap">quantMode</td>
    <td style="white-space: nowrap">Attribute</td>
    <td style="white-space: nowrap">Quantization mode. 0 (no quantization) and 2 (dynamic quantization) are supported.</td>
    <td style="white-space: nowrap">INT64</td>
    <td style="white-space: nowrap">-</td>
    </tr>
    <tr>
    <td style="white-space: nowrap">syncFlag</td>
    <td style="white-space: nowrap">Attribute</td>
    <td style="white-space: nowrap">Synchronization mode of the FFN node. 0 (synchronous) and 1 (asynchronous) are supported.</td>
    <td style="white-space: nowrap">INT64</td>
    <td style="white-space: nowrap">-</td>
    </tr>
    <tr>
    <td style="white-space: nowrap">ffnStartrankId</td>
    <td style="white-space: nowrap">Attribute</td>
    <td style="white-space: nowrap">Start ID of the FFN node.</td>
    <td style="white-space: nowrap">INT64</td>
    <td style="white-space: nowrap">-</td>
    </tr>
  </tbody>
</table>

## Constraints

- The `group`, `worldSize`, `tokenInfoTableShape`, and `tokenDataShape` parameters and the `HCCL_BUFFSIZE` value used when calling the operator must be consistent on all cards, and must also be consistent across different layers in the network.

- Format description of shapes in the parameter description:
    - `X`: micro batch sequence size (number of token groups). The current version supports only `X` = 1.
    - `BS`: batch sequence size (number of tokens finally output by the current card). The value range is 0 < `BS` ≤ 512.
    - `K`: number of topK experts selected. The value range is 0 < `K` ≤ 16 and 0 < `K` ≤ moeExpertNum.
    - `H`: hidden size. The value range is 1024 ≤ `H` ≤ 8192.
    - `L`: number of model layers. The current version supports only `L` = 1.
    - `M`: length of the last dimension of expertRankTable, which is the length of the deployment information list of the experts with the largest number deployed on FFN nodes.
    - `moeExpertNum`: number of MoE experts. The value range is 0 < `moeExpertNum` ≤ 1024.

- Communicator usage constraints:
    - The communicator of the AttentionToFFN operator must not contain other operators.

## Calling Instructions

| Calling Method      | Calling Sample                 | Description                                                         |
|--------------|-------------------------|--------------------------------------------------------------|
| aclnn call | [test_aclnn_attention_to_ffn.cpp](./examples/test_aclnn_attention_to_ffn.cpp) | Calls the AttentionToFFN operator through the [aclnnAttentionToFFN](docs/aclnnAttentionToFFN_en.md) API. |
