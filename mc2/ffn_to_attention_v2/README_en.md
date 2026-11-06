# FFNToAttentionV2

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
    <td>context</td>
    <td>Input</td>
    <td>Communicator information data of the current card.</td>
    <td>INT32</td>
    <td>ND</td>
    </tr>
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
    <td>attnRankTableOptional</td>
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
    <td>Communicator size. The value range is [2, 1024].</td>
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
    <tr>
    <td>cclBufferSize</td>
    <td>Attribute</td>
    <td>Total size of the CCL communication buffer (Bytes). The value must be >= the landing area written by this operator into the attention-side window (per-token flag area + single-session FP16/BF16 write-back data area, aligned up to 2 MB); the <code>get_ffn_to_attention_ccl_buffer_size</code> API returns a conservative capacity that covers the receive areas dispatched by attention_to_ffn and always satisfies this lower bound.</td>
    <td>INT64</td>
    <td>-</td>
    </tr>

  </tbody>
</table>

## Constraints

- The `group`, `worldSize`, `tokenInfoTableShape`, `tokenDataShape`, and `cclBufferSize` values used by all ranks must be consistent.
- This operator shares the same rank communication window with `attention_to_ffn` (identical registered tag). This operator writes back only the window slots of each attention worker itself: the per-token flag area + the single-session compact FP16/BF16 write-back data area, and the write offsets are independent of the number of attention workers; the attention-side window must accommodate this landing area (the attention-side window size is computed by `get_buffer_for_attention_to_ffn`, and its tiling has been validated). The receive areas of the N sessions dispatched by `attention_to_ffn` are laid out in the FFN-side window, and their size depends on the peer `quant_mode` (1B for quantization and 2B for non-quantization), which is validated by the tiling of `attention_to_ffn_v2` itself; this operator no longer conservatively mirrors at the non-quantized width. When `attnRankTable` is passed, its dim0 (the number of attention workers `N`) must satisfy `0 < N < worldSize`.
- All ranks must complete MC2 context initialization first. While the FFN Worker executes the operator, the Attention Worker must keep the context, communication window, and communicator valid.
- The communication memory must be destroyed only after the receiving operator finishes execution. Otherwise, the source and destination addresses of the URMA write become invalid, causing data loss, write exceptions, or service hangs. This operator asynchronously writes token data and flags into the window on the Attention side through a one-way URMA write. The return of the operator on the sending side does not indicate that the peer has finished consuming the data.
- The shape values must satisfy the following constraints:
    - `Y` indicates the maximum number of tokens that the current rank needs to dispatch.
    - `tokenInfoTableShape` is `[1, BS, expertNumPerToken]`, and `tokenDataShape` is `[1, BS, expertNumPerToken, HS]`. The first three dimensions of the two shapes must be identical.
    - The value range of `BS` is `[1, 512]`.
    - `H` indicates the hidden size, with the value range `[1024, 8192]`.
    - `HS` indicates the length of each token slot in the communication window, which must satisfy `H <= HS <= 8320`.
    - The value range of `MicroBatchNum` is `[1, 3]`.
    - The value range of `worldSize` is `[2, 1024]`.
- The optional input `attnRankTableOptional` must be a 1D Tensor whose length is smaller than `worldSize`. The value range of each target rank in the table is `[0, worldSize)`.
- Within the same target rank, multiple valid tokens must not map to the same `(microBatchId, tokenId, expertOffset)` slot. Otherwise, the written results may overwrite each other.

- `cclBufferSize` is the total size of the communication window allocated in HBM, in bytes. It must be greater than 0 and satisfy:

$$cclBufferSize \ge \mathrm{CeilAlign}\big(\mathrm{CeilAlign}(\mathrm{flagSize},\,512) + \mathrm{CeilAlign}(\mathrm{returnDataSize},\,512),\ 2\,\mathrm{MB}\big)$$

where:

  - `flagSize = microBatchNum × BS × expertNumPerToken × 4B` (the per-token flag area written by this operator into the attention-side window)
  - `returnDataSize = microBatchNum × BS × expertNumPerToken × HS × 2B` (the single-session compact FP16/BF16 write-back data area written by this operator into the attention-side window; the write offsets are independent of the number of attention workers)

The flag source area and the address table and count row areas of this operator at the end of the original window are now requested from the operator workspace, and their sizes are automatically computed by tiling, so you do not need to be aware of them.

The receive areas of the N sessions dispatched by `attention_to_ffn` are laid out in the FFN-side window, and their size depends on the peer `quant_mode` (1B for quantization and 2B for non-quantization dispatch), which is validated by the tiling of `attention_to_ffn_v2` based on the actual element width; this operator does not conservatively mirror at the non-quantized width.


- Communicator usage constraints:
    - No other operators are allowed in the communicator of the FFNToAttentionV2 operator.


## Calling Instructions

| Calling Method  | Sample Code                                  | Description                                                     |
| :--------: | :----------------------------------------: | :-------------------------------------------------------: |
| PyTorch API call | - | Calls the ffn_to_attention operator through the [ffn_to_attention](docs/torchapi_ffn_to_attention_en.md) PyTorch API. |
