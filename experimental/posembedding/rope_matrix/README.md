Rope-Matrix
Compile and test:

```bash
# How to run this sub-project$, please first read the reade at "experimental/npu_ops_transformer_ext"
path="path for experimental"
cd $path$/npu_ops_transformer_ext
python -m build --wheel -n
cd dist
pip install *.whl --force-reinstall --no-deps
cd $path$/posembedding/rope_matrix/tests
python ./test_rope.py # full code can be find at the last of readme.
```

Support limitation:

```bash
dtype=bf16, x=BNSD, y(matrix)=DD, cos/sin=11SD, D=128.
For example: x = [1, 24, 28800, 128], y = [128, 128], cos/sin = [1, 1, 28800, 128]
```

### Design

As the fig shows, we design a Mathematical Equivalence algorithm "Rope-Matrix(ROME)" to accelerate rope.
The key is to replace tensor rearrage operator by matrix.
![image.png](https://raw.gitcode.com/user-images/assets/7673863/75dd5e40-a660-4956-b052-95079eae8f8d/image.png 'image.png')
Besides, for 3D-ROPE situation, origin rope-fused need call three times, we just need to call once.
![image.png](https://raw.gitcode.com/user-images/assets/7673863/9a6d1aa0-9e73-4908-a0c7-c3cebfeda407/image.png 'image.png')

Designing detail：
By profiling, we find memory bound for both C and V on 910B. Thus, no need CV pipeline, we just simplify run V after C.
Using ```AscendC::CrossCoreSetFlag``` and ```AscendC::CrossCoreWaitFlag``` to control pipeline.

1. folder design:

```bash
  ${op_class}                                          # class
  ├── ${op_name}                                       # name
  │   ├── inc                                          # define aRopeMatrixTiling like TCubeTiling, which can be call by both op_device and op_host
  │   │   └── ${op_name}_extern.h
  │   ├── op_host                                      # Tiling、InferShape(If needed, custom operator may not need)...
  │   │   └── ${op_name}_tiling.h                      # Tiling
  │   ├── op_kernel                                    # kernel
  │   │   ├── ${op_name}.h                             # fused vector part
  │   │   └── ${op_name}_cube.h                        # matrix part
  │   ├── tests                                        # tests
  │   │   └── test_rope.py                             # test file: contain how to gen the [d, d] rope-matrix,
  │   ├── CMakeLists.txt                               # makefile
  │   ├── torch_interface.cpp                          # for torch calling
  │   └── README.md                                    # readme
```

2. design host and kernel: take 2 sub-project as reference:

```bash
from example: https://gitee.com/ascend/samples/blob/master/operator/ascendc/0_introduction/22_baremix_kernellaunch/BareMixInvocation/baremix_custom.cpp
from example: https://gitcode.com/cann/ops-transformer/blob/9.2.0/posembedding/rotary_position_embedding/rotate_half_bf16.h
```

we simplify the code from origin rope-fused operator to fit our cases.
Then use ```ASCEND_IS_AIC, ASCEND_IS_AIV``` to isolation cube/vector process and ```CrossCoreSetFlag, CrossCoreWaitFlag``` to control communication between cube and vector.

```bash
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);  // dry kernel type: KERNEL_TYPE_MIX_xxx
    TPipe tpipe;
    TCubeTiling tilingLocal;
    RopeMatrix::CopyTiling(&tilingLocal, tiling);
    if ASCEND_IS_AIC {
        ...
        AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(0x8); // set flag after matmul
    }
    if ASCEND_IS_AIV {
        AscendC::CrossCoreWaitFlag(0x8); // wait matmul outputs
        ...
    }
```

host tiling can be summary as follows:

```bash
split 'S' with vector core num, the last core may run less than other cores.
For example S=28799, vector_core=40: split to 720*39 + 719:
Then cube process 720*2, vector process 720, last vector process 719, last cube process (720+719)
```

3. fit project to torch:

```bash
A. design 'torch_interface.cpp':
1) design interface:
Add "m.def("rope_matrix(Tensor x, Tensor y, Tensor sin, Tensor cos) -> Tensor");"
into "TORCH_LIBRARY" function in "npu_ops_transformer_ext/npu_ops_def.cpp"

Then use TORCH_LIBRARY_IMPL to register your function, fuc must have same inputs and output,
speciel operator like '&' should both have or not.

// Register Ascend implementations for RopeMatrix
TORCH_LIBRARY_IMPL(npu_ops_transformer_ext, PrivateUse1, m)
{
    m.impl("rope_matrix", rope_matrix_npu);
}

// Register Meta Function for RopeMatrix
TORCH_LIBRARY_IMPL(npu_ops_transformer_ext, Meta, m)
{
    m.impl("rope_matrix", TORCH_FN(rope_matrix_meta));
}

// then design rope_matrix_npu
at::Tensor rope_matrix_kernel(at::Tensor &x, at::Tensor &y, at::Tensor &sin, at::Tensor &cos)
{
    ...

    auto acl_call = [=]() -> int {
        rope_matrix_kernel<bfloat16_t><<<blockDims, nullptr, aclstream>>>(
            (GM_ADDR)(x.storage().data()),
            (GM_ADDR)(y.storage().data()),
            (GM_ADDR)(sin.storage().data()),
            (GM_ADDR)(cos.storage().data()),
            (GM_ADDR)(output.storage().data()),
            (GM_ADDR)(workspaceTensor.storage().data()),
            ropeTiling,
            mmTilingData);
        return 0;
    };
}
```

may be you are confused by ```acl_call```, this is a standard user not need to care.

```bash
B. design compile files.
mix-core should be the same as our cmakefile
```

4. How to call

```bash
import torch
import torch_npu
import npu_ops_transformer_ext
x = torch.ops.npu_ops_transformer_ext.rope_matrix(x, mat, sin, cos)
```

or use default ```test_rope.py``` or full test code as follows:

```bash
import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_npu
from torch_npu.contrib import transfer_to_npu

from torch.profiler import ProfilerActivity, tensorboard_trace_handler
from torch.profiler import profile, schedule
from torch.autograd.profiler import record_function
from torch.cuda.amp import autocast

from einops import rearrange, repeat

os.environ["COMBINED_ENABLE"] = "1"  #
os.environ["INF_NAN_MODE_ENABLE"] = "1"
os.environ["PYTORCH_NPU_ALLOC_CONF"] = "expandable_segments:True"
torch.manual_seed(0)
torch.set_printoptions(precision=16)

torch_npu.npu.set_compile_mode(jit_compile=False)
torch_npu.npu.config.allow_internal_format = False
torch.npu.conv.allow_hf32 = False
torch.npu.matmul.allow_hf32 = False

print("import torch_npu success, training in ascend")

def get_interleave_matrix(n):
    matrix = torch.zeros(n, n, dtype=torch.bfloat16, device='npu')
    for i in range(0, n, 2):
        matrix[i + 0, i + 1] = 1
        matrix[i + 1, i + 0] = -1
    return matrix


def get_half_matrix(n):
    matrix = torch.zeros(n, n, dtype=torch.bfloat16)
    half = n // 2
    matrix[:half, half:] = torch.eye(half)
    matrix[half:, :half] = -torch.eye(half)
    return matrix.to('npu')


def compose_3matrix(matrix_a, matrix_b, matrix_c):
    total_rows = matrix_a.size(0) + matrix_b.size(0) + matrix_c.size(0)
    total_cols = matrix_a.size(1) + matrix_b.size(1) + matrix_c.size(1)

    result = torch.zeros((total_rows, total_cols), dtype=torch.bfloat16)

    result[:matrix_a.size(0), :matrix_a.size(1)] = matrix_a

    b_row_start = matrix_a.size(0)
    b_col_start = matrix_a.size(1)
    result[b_row_start:b_row_start + matrix_b.size(0),
           b_col_start:b_col_start + matrix_b.size(1)] = matrix_b

    c_row_start = matrix_a.size(0) + matrix_b.size(0)
    c_col_start = matrix_a.size(1) + matrix_b.size(1)
    result[c_row_start:c_row_start + matrix_c.size(0),
           c_col_start:c_col_start + matrix_c.size(1)] = matrix_c

    return result.to('npu')


# half
def rotate_half(x):
    x1, x2 = torch.chunk(x, 2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


# interleave
def rotate_every_two(x: torch.Tensor) -> torch.Tensor:
    x = rearrange(x, '... (d j) -> ... d j', j=2)
    x1, x2 = x.chunk(2, dim=-1)
    x = torch.cat((-x2, x1), dim=-1)
    return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')


def apply_rotary_pos_emb(tensor: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor, mode: str) -> torch.Tensor:
    sin = sin.unsqueeze(0).unsqueeze(1)
    cos = cos.unsqueeze(0).unsqueeze(1)
    if mode == 'interleave':
        return (tensor * cos) + (rotate_every_two(tensor) * sin)
    elif mode == 'half':
        return (tensor * cos) + (rotate_half(tensor) * sin)
    else:
        raise NotImplementedError("mode error, only support half or interleave")


def apply_3drotary_pos_v1(q, k, freqs_cis, mode):
    '''
    小算子实现
    '''
    sincos_h, sincos_w, sincos_t = freqs_cis

    sin_h, cos_h = sincos_h
    sin_w, cos_w = sincos_w
    sin_t, cos_t = sincos_t

    q1, q2, q3 = q.split(sin_h.shape[-1], dim=-1)
    k1, k2, k3 = k.split(sin_h.shape[-1], dim=-1)
    q1, q2, q3 = q1.contiguous(), q2.contiguous(), q3.contiguous()
    k1, k2, k3 = k1.contiguous(), k2.contiguous(), k3.contiguous()

    q1 = apply_rotary_pos_emb(q1, sin_h, cos_h, mode)
    k1 = apply_rotary_pos_emb(k1, sin_h, cos_h, mode)
    q2 = apply_rotary_pos_emb(q2, sin_w, cos_w, mode)
    k2 = apply_rotary_pos_emb(k2, sin_w, cos_w, mode)
    q3 = apply_rotary_pos_emb(q3, sin_t, cos_t, mode)
    k3 = apply_rotary_pos_emb(k3, sin_t, cos_t, mode)

    q = torch.concat([q1, q2, q3], dim=-1)
    k = torch.concat([k1, k2, k3], dim=-1)
    return q, k


def apply_3drotary_pos_v2(q, k, freqs_cis, mat1, mat2, mat3):
    '''
    rope_matrix, matrix不合一的实现
    '''
    sincos_h, sincos_w, sincos_t = freqs_cis

    sin_h, cos_h = sincos_h
    sin_w, cos_w = sincos_w
    sin_t, cos_t = sincos_t

    q1, q2, q3 = q.split(sin_h.shape[-1], dim=-1)
    k1, k2, k3 = k.split(sin_h.shape[-1], dim=-1)
    q1, q2, q3 = q1.contiguous(), q2.contiguous(), q3.contiguous()
    k1, k2, k3 = k1.contiguous(), k2.contiguous(), k3.contiguous()

    q1 = q1 * cos_h + (q1 @ mat1) * sin_h
    k1 = k1 * cos_h + (k1 @ mat1) * sin_h
    q2 = q2 * cos_w + (q2 @ mat2) * sin_w
    k2 = k2 * cos_w + (k2 @ mat2) * sin_w
    q3 = q3 * cos_t + (q3 @ mat3) * sin_t
    k3 = k3 * cos_t + (k3 @ mat3) * sin_t

    q = torch.concat([q1, q2, q3], dim=-1)
    k = torch.concat([k1, k2, k3], dim=-1)
    return q, k


def apply_3drotary_pos_v3(q, k, freqs_cis, mat, debug=None, high_precision=None):
    '''
    本方案:rope_matrix, matrix合一
    '''
    sincos_h, sincos_w, sincos_t = freqs_cis

    sin_h, cos_h = sincos_h
    sin_w, cos_w = sincos_w
    sin_t, cos_t = sincos_t

    sin = torch.cat((sin_h, sin_w, sin_t), dim=-1)
    cos = torch.cat((cos_h, cos_w, cos_t), dim=-1)

    if high_precision:
        q = q.float() * cos.float() + (q @ mat).float() * sin.float()
        k = k.float() * cos.float() + (k @ mat).float() * sin.float()
        q, k = q.to(torch.bfloat16), k.to(torch.bfloat16)
    else:
        q = q * cos + (q @ mat) * sin
        k = k * cos + (k @ mat) * sin

    return q, k


def apply_3drotary_pos_v4(q, k, freqs_cis, mode):
    '''
    当前库上融合算子实现
    '''
    sincos_h, sincos_w, sincos_t = freqs_cis

    sin_h, cos_h = sincos_h
    sin_w, cos_w = sincos_w
    sin_t, cos_t = sincos_t

    if mode == 'interleave':
        sin = torch.cat((sin_h, sin_w, sin_t), dim=-1)
        cos = torch.cat((cos_h, cos_w, cos_t), dim=-1)
        sin = sin.unsqueeze(0).unsqueeze(1)
        cos = cos.unsqueeze(0).unsqueeze(1)

        q = torch_npu.npu_rotary_mul(q, cos, sin, mode)
        k = torch_npu.npu_rotary_mul(k, cos, sin, mode)
    elif mode == 'half':
        q1, q2, q3 = q.split(sin_h.shape[-1], dim=-1)
        k1, k2, k3 = k.split(sin_h.shape[-1], dim=-1)
        q1, q2, q3 = q1.contiguous(), q2.contiguous(), q3.contiguous()
        k1, k2, k3 = k1.contiguous(), k2.contiguous(), k3.contiguous()
        sin_h = sin_h.unsqueeze(0).unsqueeze(1)
        cos_h = cos_h.unsqueeze(0).unsqueeze(1)
        sin_w = sin_w.unsqueeze(0).unsqueeze(1)
        cos_w = cos_w.unsqueeze(0).unsqueeze(1)
        sin_t = sin_t.unsqueeze(0).unsqueeze(1)
        cos_t = cos_t.unsqueeze(0).unsqueeze(1)

        q1 = torch_npu.npu_rotary_mul(q1, cos_h, sin_h, mode)
        k1 = torch_npu.npu_rotary_mul(k1, cos_h, sin_h, mode)
        q2 = torch_npu.npu_rotary_mul(q2, cos_w, sin_w, mode)
        k2 = torch_npu.npu_rotary_mul(k2, cos_w, sin_w, mode)
        q3 = torch_npu.npu_rotary_mul(q3, cos_t, sin_t, mode)
        k3 = torch_npu.npu_rotary_mul(k3, cos_t, sin_t, mode)

        q = torch.concat([q1, q2, q3], dim=-1)
        k = torch.concat([k1, k2, k3], dim=-1)
    return q, k


def apply_3drotary_pos_v5(q, k, freqs_cis, mat):
    '''
    本方案:自定义融合算子实现
    '''
    import npu_ops_transformer_ext
    sincos_h, sincos_w, sincos_t = freqs_cis

    sin_h, cos_h = sincos_h
    sin_w, cos_w = sincos_w
    sin_t, cos_t = sincos_t

    sin = torch.cat((sin_h, sin_w, sin_t), dim=-1)
    cos = torch.cat((cos_h, cos_w, cos_t), dim=-1)

    def rope(x, mat, cos, sin):
        x = torch.ops.npu_ops_transformer_ext.rope_matrix(x, mat, sin, cos)
        return x

    q = rope(q, mat, cos, sin)
    k = rope(k, mat, cos, sin)
    return q, k


class ROPE3D(nn.Module):
    def __init__(self, ):
        super().__init__()

    def forward(self, q, k, freqs_cis, mat):
        return apply_3drotary_pos_v3(q, k, freqs_cis, mat)


def main():
    # init
    B, N, S, D = 1, 24, 28800, 128
    shape_lists_1d = [128]
    shape_lists_2d = [64, 64]
    shape_lists_3d = [44, 44, 40]

    q = torch.randn((B, N, S, D), dtype=torch.bfloat16, device='npu')
    k = torch.randn((B, N, S, D), dtype=torch.bfloat16, device='npu')
    freqs_cis_1d = []
    for shape in shape_lists_1d:
        sincos = torch.randn((S, shape), dtype=torch.bfloat16, device='npu')
        freqs_cis_1d.append([sincos, sincos])
    freqs_cis_2d = []
    for shape in shape_lists_2d:
        sincos = torch.randn((S, shape), dtype=torch.bfloat16, device='npu')
        freqs_cis_2d.append([sincos, sincos])
    freqs_cis_3d = []
    for shape in shape_lists_3d:
        sincos = torch.randn((S, shape), dtype=torch.bfloat16, device='npu')
        freqs_cis_3d.append([sincos, sincos])

    inter_mat_128 = get_interleave_matrix(128)
    inter_mat_64 = get_interleave_matrix(64)
    inter_mat_44 = get_interleave_matrix(44)
    inter_mat_40 = get_interleave_matrix(40)

    half_mat_128 = get_half_matrix(128)
    half_mat_64 = get_half_matrix(64)
    half_mat_44 = get_half_matrix(44)
    half_mat_40 = get_half_matrix(40)
    half_mat_44_44_40 = compose_3matrix(half_mat_44, half_mat_44, half_mat_40)

    mode = 'half' # or 'interleave or None

    experimental_config = torch_npu.profiler._ExperimentalConfig(
        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        l2_cache=False,
    )

    op = ROPE3D()
    import torchair
    config = torchair.CompilerConfig()
    npu_backend = torchair.get_npu_backend(compiler_config=config)
    op = torch.compile(
        op,
        mode="default",
        backend=npu_backend,
        dynamic=False,
        fullgraph=False
    )

    with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(wait=0, warmup=0, active=1, repeat=1, skip_first=0),
            on_trace_ready=torch.profiler.tensorboard_trace_handler("profiling_rope_matrix"),
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
            with_flops=False,
            with_modules=False,
            experimental_config=experimental_config) as prof:

        if mode == 'interleave':
            with torch.no_grad():
                for i in range(10):
                    # 3D rope, if needed want know more call type can see the code in readme
                    XXX = torch.randn(3, 3).npu()
                    XXX = torch.pow(XXX, 2)
                    with record_function("3d rope v3"):
                        outq3, outk3 = apply_3drotary_pos_v3(q, k, freqs_cis_3d, inter_mat_128)
                        torch.cuda.synchronize()
        elif mode == 'half':
            with torch.no_grad():
                for i in range(10):
                    # 3D rope, if needed want know more can see the code in readme
                    XXX = torch.randn(3, 3).npu()
                    XXX = torch.pow(XXX, 2)
                    with record_function("3d rope v3 op"):
                        outq3, outk3 = op(q, k, freqs_cis_3d, half_mat_44_44_40)
                        torch.cuda.synchronize()

                    XXX = torch.randn(3, 3).npu()
                    XXX = torch.pow(XXX, 2)
                    with record_function("3d rope v5"):
                        outq5, outk5 = apply_3drotary_pos_v5(q, k, freqs_cis_3d, half_mat_44_44_40)
                        torch.cuda.synchronize()

                    prof.step()

    # test precision
    with record_function("3d rope v5"):
        outq5, outk5 = apply_3drotary_pos_v5(q, k, freqs_cis_3d, half_mat_44_44_40)
        torch.cuda.synchronize()

    with record_function("3d rope v3"):
        outq3, outk3 = apply_3drotary_pos_v3(q, k, freqs_cis_3d, half_mat_44_44_40, debug=None, high_precision=True)
        torch.cuda.synchronize()

    # 精度验证
    print(f"q={q[0][0][0][:10]}")
    print(f"outq3={outq3[0][0][0][:10]}")
    print(f"outq5={outq5[0][0][0][:10]}")
    print((outq3 - outq5).max())
    print(outq3.max())
    print(outq5.max())

    denominator = torch.maximum(torch.abs(outq3), torch.abs(outq5))
    abs_error = torch.abs(torch.abs(outq3) - torch.abs(outq5))
    error = abs_error / (denominator + 1e-8)
    print(torch.minimum(abs_error, error).max())
    print((torch.minimum(abs_error, error) > 0).sum())

if __name__ == '__main__':
    main()
```

you can add more test case as follows by using full test cases:
```bash
half mode:
outq1, outk1 = apply_3drotary_pos_v1(q, k, freqs_cis_3d, mode)
outq2, outk2 = apply_3drotary_pos_v2(q, k, freqs_cis_3d, inter_mat_44, inter_mat_44, inter_mat_40)
outq3, outk3 = apply_3drotary_pos_v3(q, k, freqs_cis_3d, inter_mat_128)
outq3, outk3 = op(q, k, freqs_cis_3d, inter_mat_128)
outq4, outk4 = apply_3drotary_pos_v4(q, k, freqs_cis_3d, mode)
```
```bash
interleave mode:
outq1, outk1 = apply_3drotary_pos_v1(q, k, freqs_cis_3d, mode)
outq2, outk2 = apply_3drotary_pos_v2(q, k, freqs_cis_3d, half_mat_44, half_mat_44, half_mat_40)
outq3, outk3 = apply_3drotary_pos_v3(q, k, freqs_cis_3d, half_mat_44_44_40)
outq3, outk3 = op(q, k, freqs_cis_3d, half_mat_44_44_40)
outq4, outk4 = apply_3drotary_pos_v4(q, k, freqs_cis_3d, mode)
```
