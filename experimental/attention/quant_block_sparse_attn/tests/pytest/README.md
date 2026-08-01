# quant_block_sparse_attn pytest

## Scope

The pytest framework follows the current `QuantBlockSparseAttn` op definition:

- Inputs: `query`, `key`, `value`, `q_descale`, `k_descale`, `v_descale`, `p_scale`, `cu_seqlens_q`, `cu_seqlens_kv`, `seqused_q`, `seqused_kv`, `sparse_indices`, `sparse_seq_len`, `block_table`, `atten_mask`, `metadata` (`atten_mask` is optional when `mask_mode=0`)
- Outputs: `attention_out`, optional `softmax_lse`
- Attrs: `softmax_scale`, `sparse_q_block_size`, `sparse_kv_block_size`, `layout_kv`, `layout_sparse_indices`, `quant_mode`, `mask_mode`, `return_softmax_lse`

## Golden Semantics

The CPU golden follows BSA scheme 2.4:

- `sparse_indices` stores logical KV block ids in actual KV token space.
- `q_descale` and `k_descale` are applied in Vec1 as row x column scale on raw `QK^T`.
- `p_scale[0]` is a static per-tensor P quant scale; golden casts `softmax * p_scale` to FP8 and multiplies BMM2 output by `v_descale[n2] / p_scale`.
- `key`, `value`, and `k_descale` are rebuilt as separate Tensor views over one combined `uint8` PA storage. Each physical block is packed as `K block | V block | k_descale block`. The NPU call receives 4D `key`/`value` views and a 4D `k_descale` view with shape `(block_num, N2, sparse_kv_block_size, 1)`.
- `pa_block_padding_bytes` is fixed at `0`; the segmented cache uses a tight physical block stride, exposed through `key.stride(0)`.
- Empty sparse tasks produce zero `attention_out` and `EMPTY_LSE` softmax LSE.
- `mask_mode`: `0` no mask, `3` lower-triangular causal.

## Typical Cases

`quant_block_sparse_attn_paramset.py` covers:

- TND dense-equivalent sparse selection
- TND random actual lengths
- NTD typical layout coverage

## Usage

```bash
bash test_run.sh single
bash test_run.sh batch
```

If `torch.ops.custom.npu_quant_block_sparse_attn` is not registered, NPU precision comparison is skipped. Batch generation still writes `.pt` files under `bsa_testcase`.
