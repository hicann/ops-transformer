# AttentionToFFN

English | [简体中文](./README.md)

## Product Support

| Product                                                         | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                             |    √     |
| <term>Atlas A3 products</term>       |    ×     |
| <term>Atlas A2 products</term> |    ×     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                               |    ×     |
| <term>Atlas training products</term>                              |    ×     |

## Description

- Operator function: Sends data from the Attention node to the FFN node.

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
    <td style="white-space: nowrap">context</td>
    <td style="white-space: nowrap">Input</td>
    <td style="white-space: nowrap">Communicator information data of the current card.</td>
    <td style="white-space: nowrap">INT32</td>
    <td style="white-space: nowrap">ND</td>
    </tr>
    <tr>
    <td style="white-space: nowrap">x</td>
    <td style="white-space: nowrap">Input</td>
    <td style="white-space: nowrap">Token data sent by the current card, a 3D tensor with the shape <code>(X, Bs, H)</code>.</td>
    <td style="white-space: nowrap">FLOAT16, BFLOAT16</td>
    <td style="white-space: nowrap">ND</td>
    </tr>
    <tr>
    <td style="white-space: nowrap">sessionId</td>
    <td style="white-space: nowrap">Input</td>
    <td style="white-space: nowrap">ID of the current Attention Worker node, a 1D tensor with the shape <code>(X, )</code>.</td>
    <td style="white-space: nowrap">INT32</td>
    <td style="white-space: nowrap">ND</td>
    </tr>
    <tr>
    <td style="white-space: nowrap">microBatchId</td>
    <td style="white-space: nowrap">Input</td>
    <td style="white-space: nowrap">ID of the current microBatch, a 1D tensor with the shape <code>(X, )</code>.</td>
    <td style="white-space: nowrap">INT32</td>
    <td style="white-space: nowrap">ND</td>
    </tr>
    <tr>
    <td style="white-space: nowrap">layerId</td>
    <td style="white-space: nowrap">Input</td>
    <td style="white-space: nowrap">ID of the current model layer, a 1D tensor with the shape <code>(X, )</code>.</td>
    <td style="white-space: nowrap">INT32</td>
    <td style="white-space: nowrap">ND</td>
    </tr>
    <tr>
    <td style="white-space: nowrap">expertIds</td>
    <td style="white-space: nowrap">Input</td>
    <td style="white-space: nowrap">TopK expert indexes of each token in each micro batch group, a 3D tensor with the shape <code>(X, Bs, K)</code>. The value range of expertIds is [0, moeExpertNum).</td>
    <td style="white-space: nowrap">INT32</td>
    <td style="white-space: nowrap">ND</td>
    </tr>
    <tr>
    <td style="white-space: nowrap">expertRankTable</td>
    <td style="white-space: nowrap">Input</td>
    <td style="white-space: nowrap">Mapping table from expert IDs to expert deployment on FFN cards in each micro batch group (callers must ensure that the values are correct), a 3D tensor with the shape <code>(L, moeExpertNum + sharedExpertNum, M)</code>.</td>
    <td style="white-space: nowrap">INT32</td>
    <td style="white-space: nowrap">ND</td>
    </tr>
    <tr>
    <td style="white-space: nowrap">scalesOptional</td>
    <td style="white-space: nowrap">Input</td>
    <td style="white-space: nowrap">Optional parameter. Quantization smoothing parameter of each expert, a 3D tensor with the shape <code>(L, moeExpertNum + sharedExpertNum, H)</code>.</td>
    <td style="white-space: nowrap">FLOAT32</td>
    <td style="white-space: nowrap">ND</td>
    </tr>
    <tr>
    <td style="white-space: nowrap">activeMaskOptional</td>
    <td style="white-space: nowrap">Input</td>
    <td style="white-space: nowrap">Optional parameter. Indicates whether a token participates in communication. Valid data or a null pointer can be passed. A 2D tensor with the shape <code>(X, Bs)</code>.</td>
    <td style="white-space: nowrap">BOOL</td>
    <td style="white-space: nowrap">ND</td>
    </tr>
    <tr>
    <td style="white-space: nowrap">group</td>
    <td style="white-space: nowrap">Attribute</td>
    <td style="white-space: nowrap">Communicator name (expert parallelism). The string length is [1, 128).</td>
    <td style="white-space: nowrap">STRING</td>
    <td style="white-space: nowrap">-</td>
    </tr>
    <tr>
    <td style="white-space: nowrap">worldSize</td>
    <td style="white-space: nowrap">Attribute</td>
    <td style="white-space: nowrap">Size of the communicator. Value range: [2, 1024].</td>
    <td style="white-space: nowrap">INT64</td>
    <td style="white-space: nowrap">-</td>
    </tr>
    <tr>
    <td style="white-space: nowrap">ffnTokenInfoTableShape</td>
    <td style="white-space: nowrap">Attribute</td>
    <td style="white-space: nowrap">List of shape sizes of the token information table on the FFN node. The length is 3.</td>
    <td style="white-space: nowrap">INT32</td>
    <td style="white-space: nowrap">-</td>
    </tr>
    <tr>
    <td style="white-space: nowrap">ffnTokenDataShape</td>
    <td style="white-space: nowrap">Attribute</td>
    <td style="white-space: nowrap">List of shape sizes of the token data table on the FFN node. The length is 5.</td>
    <td style="white-space: nowrap">INT32</td>
    <td style="white-space: nowrap">-</td>
    </tr>
    <tr>
    <td style="white-space: nowrap">attnTokenInfoTableShape</td>
    <td style="white-space: nowrap">Attribute</td>
    <td style="white-space: nowrap">List of shape sizes of the token information table on the Attention node. The length is 3.</td>
    <td style="white-space: nowrap">INT32</td>
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
    <td style="white-space: nowrap">Quantization mode. The value encodes both the algorithm and the output type. Supported values: 0 (non-quantization), 2 (PERTOKEN+INT8), 3 (MX+FP8_E5M2), 4 (MX+FP8_E4M3), 5 (MX+FP4_E2M1), 6 (MX_CLIP+FP8_E5M2), 7 (MX_CLIP+FP8_E4M3).</td>
    <td style="white-space: nowrap">INT64</td>
    <td style="white-space: nowrap">-</td>
    </tr>
    <tr>
    <td style="white-space: nowrap">syncFlag</td>
    <td style="white-space: nowrap">Attribute</td>
    <td style="white-space: nowrap">Synchronization mode of the FFN node. Supported values: 0 (synchronous) and 1 (asynchronous).</td>
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
    <tr>
    <td style="white-space: nowrap">cclBufferSize</td>
    <td style="white-space: nowrap">Attribute</td>
    <td style="white-space: nowrap">Total size of the CCL communication buffer (Bytes), automatically computed inside the <code>get_buffer_for_attention_to_ffn</code> API. The value must be >= token_info_size + token_data_size (aligned up to 2 MB). The staging area and the resident flag area are now requested from the operator workspace and do not occupy the window.</td>
    <td style="white-space: nowrap">INT64</td>
    <td style="white-space: nowrap">-</td>
    </tr>
  </tbody>
</table>

## Quantization Mode Description

The `quantMode` value encodes both the quantization algorithm and the output data type (aligned with the mega_moe dispatch quant encoding):

| quantMode | Algorithm | Output Type | Description |
| :--------: | :--: | :------: | :---: |
| 0 | Non-quantization | Same as input | FP16/BF16 data is directly transmitted |
| 2 | PERTOKEN | INT8 | The scale is dynamically computed for each token and quantized to INT8 |
| 3 | MX | FP8_E5M2 | The MX scale (E8M0 format) is computed for each group of 32 elements, and the output is FP8_E5M2 |
| 4 | MX | FP8_E4M3 | The MX scale (E8M0 format) is computed for each group of 32 elements, and the output is FP8_E4M3 |
| 5 | MX | FP4_E2M1 | The MX scale (E8M0 format) is computed for each group of 32 elements, and the output is FP4_E2M1 |
| 6 | MX_CLIP | FP8_E5M2 | The MX scale is computed for each group of 32 elements (with 1e-4 lower-bound clipping), and the output is FP8_E5M2 |
| 7 | MX_CLIP | FP8_E4M3 | The MX scale is computed for each group of 32 elements (with 1e-4 lower-bound clipping), and the output is FP8_E4M3 |

> **Internal implementation note**: The tiling layer splits quantMode into two fields of the tiling key: `QUANT_MODE` (algorithm: 0=NO_QUANT, 2=PERTOKEN, 3=MX, 4=MX_CLIP) and `OUT_DTYPE` (output type: 0=INT8, 1=E5M2, 2=E4M3, 3=E2M1). You do not need to be aware of this split.

## Constraints

- The values of the `group`, `worldSize`, `tokenInfoTableShape`, and `tokenDataShape` parameters and `ccl_buffer_size` used in the operator calling process must be consistent across all cards and consistent across different layers in the network.

- `ccl_buffer_size` is the **total size** (Bytes) of the CCL communication buffer allocated on HBM, which must satisfy:

$$ccl\_buffer\_size \ge \mathrm{CeilAlign}(\mathrm{CeilAlign}(\mathrm{tokenInfoSize},\ 512) + \mathrm{CeilAlign}(\mathrm{tokenDataSize},\ 512),\ 2\,\mathrm{MB})$$

where:

  - `tokenInfoSize = attentionWorkerNum × microBatchNum × infoTableLastDim × 4B`
  - `tokenDataSize = attentionWorkerNum × microBatchNum × BS × (K+shared) × HS × bytesPerElem`
  - `infoTableLastDim`: length of the last dimension of the token information table on the FFN side, that is, the last dimension of `ffnTokenInfoTableShape`, equal to `2 + BS × expertNumPerToken` (`expertNumPerToken = K + sharedExpertNum`; the first two int32 entries are reserved entries, and the remaining entries are information table entries stored for each token by expert, with each entry being 4B).
  - `bytesPerElem`: number of transmission bytes per element of the token data, determined by quantMode: in non-quantization mode (quantMode=0), it is 2B (FP16/BF16 is transmitted at the original precision); in quantization modes (quantMode=2~7, including INT8/FP8/FP4), it is 1B, where in FP4 mode two elements are packed into 1B, and `HS` is the number of bytes converted based on packing.

> Only the token information table and the token data area are retained in the window. The staging area at the end of the original window (one staging slot per token, with the slot stride aligned up to 32 B) and the resident URMA flag area (one 4 B compact flag slot per token plus one 32 B layer flag slot) are now requested from the operator workspace, and their sizes are automatically computed by tiling, so you do not need to be aware of them. The non-quantization mode (quantMode=0) is consistent with the quantization mode: rows of the input `x` are first relayed through the UB (GM→UB→GM) and written into the staging area, the relay WQEs read data from the staging area and send it to the target card, and consecutive tokenOffsets can be merged into a single WQE for continuous transmission.

This size is automatically computed inside the `get_buffer_for_attention_to_ffn` API. You do not need to compute or set `ccl_buffer_size` yourself.

- MX/MX_CLIP quantization constraints:
  - The MX modes (quantMode 3/4/5) and MX_CLIP modes (quantMode 6/7) do not support passing the scales parameter.
  - When the output is FP4 (quantMode=5), the H dimension must be an even number.
  - The MX_CLIP modes (quantMode 6/7) support only FP8 output (E5M2/E4M3) and do not support FP4.

- Shape format description in the parameter description:
    - `X`: Indicates the micro batch sequence size (number of token groups). Only `X` = 1 is supported in the current version.
    - `microBatchNum`: Indicates the number of micro batches in the communication window layout, that is, dimension 1 of `ffnTokenDataShape` (equal to dimension 1 of `ffnTokenInfoTableShape` and dimension 0 of `attnTokenInfoTableShape`). The value range is [1, 3]; the value of `microBatchId` at runtime must be in the range `[0, microBatchNum)`.
    - `BS`: Indicates the batch sequence size (number of tokens finally output by the current card). The value range is 0 < `BS` ≤ 512.
    - `K`: Indicates that topK experts are selected. The value range is 0 < `K` ≤ 16, and 0 < `K` ≤ moeExpertNum must be satisfied.
    - `H`: Indicates the hidden size. The value range is 1024 ≤ `H` ≤ 8192.
    - `L`: Indicates the number of model layers. The value range is [1, 43], and the value of `LayerId` at runtime must be less than `L`.
    - `M`: Indicates the length of the last dimension of expertRankTable, which is the length of the deployment information list of the experts most deployed on an FFN node. The value range is 0 < `M` ≤ 2049 (each row is laid out as `[rankCnt, (toRankId, localExpId) × rankCnt]`, and rankCnt does not exceed worldSize).
    - `moeExpertNum`: Indicates the number of MoE experts. The value range is 0 < `moeExpertNum` ≤ 1024.

- Communicator usage constraints:
    - The communicator of the AttentionToFFN operator must not contain other operators.

## Calling Instructions

| Calling Method  | Sample Code                                  | Description                                                     |
| :--------: | :----------------------------------------: | :-------------------------------------------------------: |
| PyTorch API call | - | Call the attention_to_ffn_v2 operator through the [attention_to_ffn](docs/torchapi_attention_to_ffn_en.md) PyTorch API. |
