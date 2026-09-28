# K2qCsr

将 `q2k` 索引转为 k2q CSR（Compressed Sparse Row）。Host 按命名阶段串行 launch：

```text
Meta → Hist → RowPrefix → TilePrefix → Scatter
```

## 产品支持情况

| 产品 | 是否支持 |
| :----------------------------------------- | :------:|
| <term>Ascend 950PR&950DT系列产品</term> | √ |
| <term>Atlas A3系列产品</term> | √ |
| <term>Atlas A2系列产品</term> | √ |
| <term>Atlas 200I/500 A2推理产品</term> | × |
| <term>Atlas推理系列产品</term> | × |
| <term>Atlas训练系列产品</term> | × |

- <term>Ascend 950PR&950DT系列产品</term>：Hist / Scatter 支持 MC 与 SIMT（`use_simt=1`）。
- <term>Atlas A2 / A3</term>：仅 MC，`use_simt` 由 tiling 强制为 0。

## 功能说明

- 算子功能：根据每条 query 的 KV block 索引 `q2k`，以及 query / KV block 前缀和，构建按 KV 行为主序的 CSR：`row_ptr`、`q_ind`、`slot`。
- 主要计算过程：
  1. **Meta**：按 `order_method` 生成 `row_map` 与 `token_batch`，清零 `row_counts`。
  2. **Hist**：统计每个 tile / 行上的非空命中次数。
  3. **RowPrefix**：对 `row_counts` 做 exclusive scan，写出 `row_ptr`。
  4. **TilePrefix**：生成各 tile 的绝对写偏移 `abs_base`。
  5. **Scatter**：按 CSR 写出 `q_ind`（query 下标）与 `slot`（topk 槽位）；未命中位为 `-1`。

## 参数说明

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
| --- | --- | --- | --- | --- |
| q2k | 输入 | 每条 query 的 KV block 索引 | INT32 | ND |
| cu_seqlens | 输入 | query 侧前缀和，长度为 `B+1` | INT32 | ND |
| cu_block_lens | 输入 | KV block 前缀和，长度为 `B+1`，末元素即 `total_rows` | INT32 | ND |
| scratch | 内部 | 五阶段共享工作区（Host 申请） | INT32 | ND |
| order_method | 属性 | `0`：按 batch 内 block 顺序建 `row_map`；`1`：round-robin 交错 | INT64 | - |
| total_rows | 属性 | KV 总行数；Host 可从 `cu_block_lens` 末元素推导 | INT64 | - |
| max_kv | 属性 | 单 batch 最大 KV block 数 | INT64 | - |
| use_simt | 属性 | `1` 时 Hist/Scatter 走 SIMT（仅 950 生效） | INT64 | - |
| q_global_offset | 属性 | `1` 时 `q_ind` 为全局 Q 下标；`0` 为 batch-local | INT64 | - |
| row_ptr | 输出 | CSR 行指针，shape `[H, total_rows+1]` | INT32 | ND |
| q_ind | 输出 | 非空位为 query 下标，未写入位为 `-1`，shape `[H, T*topk]` | INT32 | ND |
| slot | 输出 | 对应 topk slot，未写入位为 `-1`，shape `[H, T*topk]` | INT32 | ND |

五阶段各自的 aclnn 接口为 `aclnnK2qCsrMeta` / `Hist` / `RowPrefix` / `TilePrefix` / `Scatter`。PyTorch 侧由 `cann_ops_transformer.k2q_csr` 编排一次调用。

## 约束说明

- `q2k` 必须为 3 维 `[H, T, topk]`，dtype 为 INT32，且位于 NPU。
- `cu_seqlens`、`cu_block_lens` 必须为 1 维 INT32 NPU Tensor。
- `order_method` 仅支持 `0` 或 `1`。
- 建议在已知 shape 时由 Host 显式传入 `total_rows` / `max_kv`，避免从 Device 回读。
- `q_ind` / `slot` 由 Host 预填 `-1`，与 TilePrefix 重叠；A5 SIMT 核内仍可再填。

## 编译与安装

在 **ops-transformer 仓库根目录** 编译 experimental 自定义算子包：

```bash
bash build.sh --pkg --experimental --soc=${soc_version} --ops=minimax_build_k2q_csr
```

- `--soc`：Atlas A2 使用 `ascend910b`，Atlas A3 使用 `ascend910_93`，Ascend 950 使用 `ascend950`。
- 成功后 run 包位于 `build_out/cann-ops-transformer-custom_linux-$(arch).run`。

安装：

```bash
./build_out/cann-ops-transformer-custom_linux-$(arch).run
# 或指定路径：
# ./build_out/cann-ops-transformer-custom_linux-$(arch).run --install-path=${install_path}
```

安装后激活 vendor：

```bash
export LD_LIBRARY_PATH=${ASCEND_HOME_PATH}/opp/vendors/custom_transformer/op_api/lib/:${LD_LIBRARY_PATH}
```

共享实现只维护在 `k2q_csr_common/`；各阶段 `op_kernel/common/` 由编译前 vendoring 生成，勿手改。

## torch 接口

需先安装 `cann_ops_transformer`（见 `torch_extension/README.md`），再调用：

```python
import torch
import torch_npu
from cann_ops_transformer.ops import k2q_csr

H, T, topk = 8, 128, 8
q2k = torch.randint(0, 16, (H, T, topk), dtype=torch.int32, device="npu")
cu_seqlens = torch.tensor([0, T], dtype=torch.int32, device="npu")
cu_block_lens = torch.tensor([0, 16], dtype=torch.int32, device="npu")

row_ptr, q_ind, slot = k2q_csr(
    q2k, cu_seqlens, cu_block_lens,
    order_method=0,
    total_rows=16,
    max_kv=16,
    use_simt=True,
)
```

接口说明见 [torchapi_k2q_csr](./docs/torchapi_k2q_csr.md)。

## 目录结构

```text
minimax_build_k2q_csr/
  CMakeLists.txt                 # 挂载五阶段子算子
  k2q_csr_common/                # 共享 tiling / MC / SIMT（无 OpDef）
  k2q_csr_{meta,hist,row_prefix,tile_prefix,scatter}/
    op_host/                     # def / infershape / tiling
    op_kernel/{op}.cpp           # ascend910b 入口
    op_kernel/{op}_apt.cpp       # ascend950 入口
  ```
