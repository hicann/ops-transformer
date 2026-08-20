# TurboQuantSparseFlashAttention PyTorch Extension

This experimental PyTorch interface is maintained with the operator implementation. It is not part of the
commercial `cann_ops_transformer` interface package.

## Build and install

```bash
cd experimental/attention/turbo_quant_sparse_flash_attention/torch_ops_extension
bash build_and_install.sh
```

## Use

```python
import torch
import custom_ops  # noqa: F401

result = torch.ops.custom.npu_turbo_quant_sparse_flash_attention(...)
```

Importing `custom_ops` also mounts the operation as `torch_npu.npu_turbo_quant_sparse_flash_attention`.
