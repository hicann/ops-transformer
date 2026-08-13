# stem_indexer — PyTorch 接入层 (TorchNPU/torch_ops_extension)

将 `stem_indexer` 算子桥接到 PyTorch，注册为 `torch.ops.custom.npu_stem_indexer`
（同时挂载到 `TorchNPU.npu_stem_indexer`），底层调用 ACLNN 接口 `aclnnStemIndexer`；
元数据接口 `npu_stem_indexer_metadata` 调用 `aclnnStemIndexerMetadata`。

> 结构与配置对齐参考实现 `quant_block_sparse_attn/torch_ops_extension`：编译式 `NpuExtension`
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
    │   ├── ops_def_registration.cpp     # TORCH_LIBRARY(custom): npu_stem_indexer(_metadata) schema
    │   ├── ops_common.h / ops_common.cpp # EXEC_NPU_CMD_V1 / ConvertType 基础设施（拷贝）
    │   ├── npu_stem_indexer.cpp         # NPU/Meta 前向实现 + TORCH_LIBRARY_IMPL
    │   └── npu_stem_indexer_metadata.cpp
    └── converter/
        ├── __init__.py
        ├── npu_stem_indexer.py          # torch.compile (GE) 成图 converter
        └── npu_stem_indexer_metadata.py
```

## 前置条件

- Linux；Python 3.8+；GCC 9.4.0+
- PyTorch >= 2.6.0 与匹配版本的 `TorchNPU`
- Ascend CANN Toolkit（`ASCEND_HOME_PATH` 已设置）
- 已部署 `aclnnStemIndexer` / `aclnnStemIndexerMetadata` 算子包，运行时可经
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
import custom_ops   # 注册 torch.ops.custom.npu_stem_indexer(_metadata) 并挂载到 torch_npu

metadata = torch.ops.custom.npu_stem_indexer_metadata(
    q_seq_lens, kv_seq_lens, q_heads, kv_heads,
    causal=True, stem_block_size=128, dim_qkflat=2048, window_size=4,
)
sparse_indices, sparse_seq_len = torch.ops.custom.npu_stem_indexer(
    qflat, kflat, vbias, q_seq_lens, kv_seq_lens,
    causal=True, stem_block_size=128, stem_stride=16, alpha=1.0,
    initial_blocks=4, window_size=4,
    k_block_num_rate_medium=0.2, k_block_num_bias_medium=30,
    k_block_num_rate_large=0.1, k_block_num_bias_large=30,
    topk_score_precision=1,
    num_prompt_tokens=num_prompt_tokens, metadata=metadata,
)
# 也可：TorchNPU.npu_stem_indexer(...)
```

### 与 stem_indexer pytest 集成

测试通过环境变量 `STEM_INDEXER_CUSTOM_OPS_PATH`（或默认相对路径
`<stem_indexer>/torch_ops_extension`）检索本扩展，既支持 glob
`custom_ops_lib*.so`（`torch.ops.load_library`），也支持 exec `custom_ops/__init__.py`。

## 接口与 IR

- 入参/属性/输出与算子原型 `op_host/stem_indexer_def.cpp` 一一对应。
- `EXEC_NPU_CMD_V1` 实参按 IR 声明顺序传入（输入→属性→输出）；Python schema 为便于调用将必选张量前置，
  C++ 实现内部已按 IR 顺序重排。
- `qflat` / `kflat` 为 BF16 `[B, N, Qb/Kb, stem_stride*D]`；`vbias` 为 FP32 `[B, kv_heads, Kb]`；
  其余 `q_seq_lens` / `kv_seq_lens` / `num_prompt_tokens` / `metadata` 为 INT32，
  `metadata` 长度按 `16 + B * kv_heads * (36 + 72) * 16` 计算并向上对齐到4096个INT32元素；
  输出 `sparse_indices` / `sparse_seq_len` 为 INT32。
- `num_prompt_tokens` / `metadata` 为可选张量（`c10::optional<at::Tensor>`），两条调用通路均将缺省状态传递给
  OpHost。未提供 `num_prompt_tokens` 时，TilingData记录复用标志，Kernel使用 `kv_seq_lens`；未提供
  `metadata` 时，OpHost Tiling按当前计算要求返回明确的参数错误。
