# MiniMaxSparseAttentionSplitKv PyTorch Extension

This experimental PyTorch interface is maintained with the operator implementation. It is not part of the
commercial `cann_ops_transformer` interface package.

## Build and install

```bash
cd experimental/attention/minimax_sparse_attention_split_kv/torch_ops_extension
bash build_and_install.sh
```

## Use

```python
import torch
import custom_ops
from custom_ops import build_k2q_csr, npu_minimax_sparse_attention_split_kv

row_ptr, q_idx, slot_idx = build_k2q_csr(...)
attn_out, softmax_lse = npu_minimax_sparse_attention_split_kv(...)
# or: torch.ops.custom.npu_minimax_sparse_attention_split_kv(...)
```
