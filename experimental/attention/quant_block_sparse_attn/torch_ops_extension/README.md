# quant_block_sparse_attn — PyTorch 接入层 (TorchNPU/ torch_ops_extension)

将 `quant_block_sparse_attn` 算子桥接到 PyTorch，注册为 `torch.ops.custom.npu_quant_block_sparse_attn`
（同时挂载到 `TorchNPU.npu_quant_block_sparse_attn`），底层调用 ACLNN 接口 `aclnnQuantBlockSparseAttn`。

> 结构与配置对齐参考实现 `ops_adv_open/torch_ops_extension`：编译式 `NpuExtension`
> （`custom_ops.custom_ops_lib`）+ `TORCH_LIBRARY`/`TORCH_LIBRARY_IMPL` + `EXEC_NPU_CMD_V1`。
> `csrc/ops_common.h`、`csrc/ops_common.cpp` 为该参考的整文件拷贝（自包含）。

## 目录结构

```
torch_ops_extension/
├── setup.py                    # NpuExtension custom_ops.custom_ops_lib，编译 csrc/*.cpp
├── build_and_install.sh        # 构建 wheel 并 pip 安装
├── README.md
└── custom_ops/
    ├── __init__.py             # 导入 custom_ops_lib(.so) + converter，并挂载到 TorchNPU
    ├── csrc/
    │   ├── ops_def_registration.cpp     # TORCH_LIBRARY(custom): npu_quant_block_sparse_attn schema
    │   ├── ops_common.h / ops_common.cpp # EXEC_NPU_CMD_V1 / ConvertType 基础设施（拷贝）
    │   └── npu_quant_block_sparse_attn.cpp # NPU/Meta 前向实现 + TORCH_LIBRARY_IMPL
    └── converter/
        ├── __init__.py
        └── npu_quant_block_sparse_attn.py # torch.compile (GE) 成图 converter
```

## 前置条件

- Linux；Python 3.8+；GCC 9.4.0+
- PyTorch >= 2.6.0 与匹配版本的 `TorchNPU`
- Ascend CANN Toolkit（`ASCEND_HOME_PATH` 已设置）
- 已部署 `aclnnQuantBlockSparseAttn` 算子包，运行时可经
  `ASCEND_CUSTOM_OPP_PATH` / `ASCEND_OPP_PATH` 检索到 `libcust_opapi.so`

## 构建与安装

```sh
# 方式一：构建 wheel 并安装（推荐）
bash build_and_install.sh

# 方式二：就地编译（.so 生成在 custom_ops/custom_ops_lib*.so）
python3 setup.py build_ext --inplace
```

## 用法

### eager / 单算子

```python
import torch, torch_npu
import custom_ops   # 注册 torch.ops.custom.npu_quant_block_sparse_attn 并挂载到 torch_npu

attn_out, softmax_lse = torch.ops.custom.npu_quant_block_sparse_attn(
    query, key, value,                       # FP8_E4M3FN
    q_descale, k_descale, v_descale, p_scale,      # FLOAT32
    sparse_indices, sparse_seq_len, atten_mask,  # mask_mode=0 时可传 None
    softmax_scale, sparse_q_block_size, sparse_kv_block_size,
    layout_kv="PA_BNBD", layout_sparse_indices="B_N_Qb_Kb",
    block_table=block_table,                 # 可选：PagedAttention
    return_softmax_lse=False,
)
# 也可：torch_npu.npu_quant_block_sparse_attn(...)
```

### 与 QBSA pytest 集成

`tests/pytest/custom_ops.py` 通过 `QBSA_CUSTOM_OPS_PATH` 检索本扩展，既支持 glob
`custom_ops_lib*.so`（`torch.ops.load_library`），也支持 exec `custom_ops/__init__.py`：

```sh
export QBSA_CUSTOM_OPS_PATH=<repo>/experimental/attention/quant_block_sparse_attn/torch_ops_extension
cd <repo>/experimental/attention/quant_block_sparse_attn/tests/pytest
bash test_run.sh
```

注册成功后 `has_quant_block_sparse_attn_op()` 返回 True。

## 接口与 IR

- 入参/属性/输出与算子原型 `op_host/quant_block_sparse_attn_def.cpp` 一一对应。
- `paBlockStride`（组合 `PA_BNBD` cache 的物理块外步长，单位为 FP8 元素数）不再是接口参数，
  host tiling 从 `key.stride(0)` 推导；`torch.compile` converter 也不再构造该属性。
- `EXEC_NPU_CMD_V1` 实参按 IR 声明顺序传入（输入→属性→输出）；Python schema 为便于调用将必选张量前置，
  C++ 实现内部已按 IR 顺序重排。
- FP8 量化：q/k per-token-per-head、v per-head、p per-tensor 静态（用户传入）。
- `attention_out` 输出 BF16；`softmax_lse` FLOAT32（关闭时为空张量）。
- 与 FIA / sparse_flash_attention 一致：Tiling 阶段无法读取 tensor 数值，`sparse_indices` /
  `sparse_seq_len` / `cu_seqlens_*` / `block_table` 合法性由用户保证。
