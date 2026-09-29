# IndexerPrologueK

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :---: |
| Ascend 950PR/Ascend 950DT | √ |
| Atlas A3 训练/推理系列产品 | × |
| Atlas A2 训练/推理系列产品 | × |

## 功能说明

`indexer_prologue_k` 融合完成 Indexer K 路的 BF16 投影、RMSNorm、尾部 RoPE、
MXFP4 量化和分页 cache 写入。算子原地更新 `k_cache`，mode 0 可同时原地更新
独立的 `k_scale_cache`，并返回传入的 `k_cache`。

Kernel 由配套的 net-ops wheel 提供，本目录只负责 PyTorch schema、
Meta/FakeTensor 和 PrivateUse1 dispatcher 适配。调用前必须安装包含
`ops.indexer_prologue_k` 的 net-ops wheel。

## 接口

```python
indexer_prologue_k(
    latent,
    wk,
    norm_weight,
    rope_sin,
    rope_cos,
    k_cache,
    k_scale_cache=None,
    *,
    cache_index,
    storage_mode,
    norm_eps,
    combined_block_size=-1,
) -> Tensor
```

| 参数 | dtype / format | shape |
| :--- | :--- | :--- |
| `latent` | BF16 / ND | `[T, H]` |
| `wk` | BF16 / FRACTAL_NZ | `[D, H]` |
| `norm_weight` | FP32 / ND | `[D]` |
| `rope_sin`, `rope_cos` | FP32 / ND | `[T, Dr]` |
| `cache_index` | INT64 / ND | `[T]`，`-1` 表示跳过 |
| `k_cache` (mode 0) | UINT8 / ND | `[Bn, Bs, 1, D/2]` |
| `k_scale_cache` (mode 0) | UINT8 / ND，可选 | `[Bn, Bs, 1, D/32]` |
| `k_cache` (mode 1) | UINT8 / ND | `[Bn, Bs/G, 1, G*(D/2+D/32)]` |

其中 `G=combined_block_size`。约束为 `T>0`、`H%16==0`、`D%32==0`、
`0<Dr<=D` 且 `Dr` 为偶数。mode 1 要求 `G>0`。

## 调用示例

```python
import torch_npu
import cann_ops_transformer.ops
from cann_ops_transformer.ops.ds41 import indexer_prologue_k

wk_nz = torch_npu.npu_format_cast(wk.npu(), torch_npu.Format.FRACTAL_NZ)
result = indexer_prologue_k(
    latent.npu(),
    wk_nz,
    norm_weight.npu(),
    rope_sin.npu(),
    rope_cos.npu(),
    k_cache,
    k_scale_cache,
    cache_index=cache_index.npu(),
    storage_mode=0,
    norm_eps=1e-6,
)
assert result is k_cache
```

也可通过 `cann_ops_transformer.ops.indexer_prologue_k` 或
`torch.ops.cann_ops_transformer.ds41.indexer_prologue_k` 调用。

## 构建与测试

先在 net-ops 工程的 `net/native_package` 目录构建并安装包含 Native provider
的 wheel：

```bash
cd net/native_package
./run-build.sh --operator indexer_prologue_k
python3 -m pip install \
  output/operators/indexer_prologue_k/wheels/*.whl \
  --force-reinstall --no-deps
```

该构建会打包 4 个 ACLGraph 验证布局的 AOT variant；其他合法 shape 在默认
Native `prefer` 模式下首次调用 JIT。随后构建本仓整包：

```bash
bash build.sh --torch_extension
python3 -m pip install build_out/*.whl --force-reinstall --no-deps
```

执行真机精度用例：

```bash
bash experimental/attention/indexer_prologue_k_dsl/tests/test.sh
```

若要确认 ACLGraph 用例命中已安装的 AOT variant，可在测试前设置
`CANNBOTDSL_NATIVE_BINARY_MODE=require`；此时未命中不会回退 JIT。

## ACLGraph

公开 Python 接口支持 `npugraph_ex` capture/replay。算子内部使用一个无
alias 返回值的 mutation-only schema，以满足 PyTorch AOT functionalization；
公开接口仍返回原地更新后的 `k_cache`。

`wk` 使用 FRACTAL_NZ 私有格式，ACLGraph 必须关闭输入 clone，否则
`empty_like` 生成的捕获输入会退化为 ND：

```python
import torchair
from torchair.configs.compiler_config import CompilerConfig

config = CompilerConfig()
config.mode = "npugraph_ex"
config.debug.aclgraph.clone_input = False
backend = torchair.get_npu_backend(compiler_config=config)

compiled_model = torch.compile(
    model.npu(), fullgraph=True, backend=backend, dynamic=False
)
```

图模式应调用 `cann_ops_transformer.ops.ds41.indexer_prologue_k` Python
接口。直接调用带 alias 返回值的底层
`torch.ops.cann_ops_transformer.ds41.indexer_prologue_k` 仅保留用于 eager
兼容，不适合 AOT functionalization。
