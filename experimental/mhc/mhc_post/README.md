# mhc_post Ascend C Operator

Broadcast single stream to multiple streams with per-stream scaling (post-connection in mHC).

## Formula

```
output[b * N + n, seq, d] = x[b, seq, d] × h_post[n]
```

Equivalent einsum: `torch.einsum('bsd,n->bnsd', x, h_post).reshape(B*N,S,D)`

## Notes

- `h_post` is a static weight vector [num_streams], shared across all batches and token positions
- This matches the tokenbender/mHC open-source implementation
- Weight normalization is handled upstream, not in this operator

## Adaptive Strategy

The kernel automatically selects between two parallelization strategies based on shape:

- **Strategy A**: Parallelize over (batch, stream) pairs. Best for small/medium data sizes.
- **Strategy B**: Read input once, write N outputs. Best for large data with large batch.

See `docs/performance.md` for detailed benchmarks.

## Build

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 1. Build AscendC kernel
mkdir -p build && cd build
cmake .. -DSOC_VERSION=ascend910b2
make -j
cd ..

# 2. Build PyTorch C++ extension
python setup.py build_ext --inplace
```

## Test

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# C++ test
cd build && LD_LIBRARY_PATH=./lib:$LD_LIBRARY_PATH ./test_multi_dtype

# Python test
LD_LIBRARY_PATH=./build/lib:$LD_LIBRARY_PATH python mhc_post_ops.py
```

## API

```python
# Python (via C++ extension)
import mhc_post_ext
output = mhc_post_ext.forward(x, h_post)  # x on NPU, returns NPU tensor

# Or use wrapper
from mhc_post_ops import mhc_post, mhc_post_einsum
output = mhc_post(x, h_post)
```

```cpp
// C++ kernel (auto-selects strategy)
extern "C" void mhc_post_do_fp32(uint32_t blockDim, void* stream,
    uint8_t* input, uint8_t* h_post, uint8_t* output,
    int64_t batch, int64_t seq_len, int64_t dim, int64_t num_streams);
```

## Performance

On Ascend 910B2, mhc_post achieves **2-4x speedup** over torch.einsum for most shapes. See `docs/performance.md` for details.
