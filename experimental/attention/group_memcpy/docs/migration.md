# 原直调实现与迁移说明

## 来源

原始 Kernel 直调工程中的文件保持不变，SHA256 如下：

| 原文件说明 | 内容 | SHA256 |
| --- | --- | --- |
| 原 GroupMemcpy launch 实现 | 对外 launch、UB 容量判断、64 字节切分计算 | A27A1C0BDEC1D3DDC8FFA054FB48820E859CE5E2E397A99BDE753DBE5ADF4491 |
| 原 GroupMemcpy 单缓冲 Kernel 实现 | 单缓冲双队列 Kernel、CopyIn/Compute/CopyOut 和直调 launch | AEEF6DFE485504B7B388EE78183B2E3316AA7356E4E4960BF6FEE6DC0E569D7B |
| flash_attention.cc | 上层 K/V cache 地址收集和搬运调用示例 | 29A9872DA96D3367B456A02692DFC6ED8E3A562D282E482DBB20203CBBE128C3 |

原目录没有完整编译脚本、所需头文件及原始输入/输出 bin，因此无法在本机建立原直调基线；新增随机/规则数据测试不冒充原直调二进制一致性结果。`flash_attention.cc` 依赖的外部 `NewGptDenseCacheKV` 实现不在原目录中，没有将整个上层模块加入新算子构建。

## 接口映射

| 原接口信息 | 新接口 |
| --- | --- |
| 设备 `void** srcdstPair`，按 src0、dst0、src1、dst1 交错排列 | 两个 CPU 侧 aclTensorList，元素指向对应 Device Tensor |
| gbH | src 列表长度；与 dst 列表长度相同 |
| gbW（字节） | 每个 Tensor 的 numel × element_size，所有组相同 |
| 手工指定 blockDim | Tiling 使用 min(实际 AIV 核数, 任务数) |
| 手工 stream | aclnn 第二段显式传入；Torch 使用设备 guard 下的当前 NPU stream |
| 原 GroupMemcpy launch 接口 | aclnnGroupMemcpyGetWorkspaceSize + aclnnGroupMemcpy |

原调用的 `local_num_heads_kv * head_dim_ * sizeof(BFloat16)` 对应每个 BFLOAT16 视图包含 `local_num_heads_kv * head_dim_` 个元素。

迁移已有 C++ 调用方时，逐个取出源、目标内存片段，分别以 `aclCreateTensor` 描述 shape、stride、offset、存储容量和 Device 基址，再用 `aclCreateTensorList` 组装两组列表。只有裸 Device 指针时，可以用一维 UINT8 视图描述 `gbW` 个字节；调用方须知道真实分配容量与地址生命周期。不能直接把原设备指针表当作 aclTensorList。

已有 PyTorch K/V Tensor 时，用连续切片创建视图并调用：

~~~python
from cann_ops_transformer.ops.attention.group_memcpy import group_memcpy

# kv_src 和 kv_dst 是同一 NPU 上已有的连续 BFLOAT16 张量；
# 每个切片都覆盖恰好一组要搬运的数据，所有目标范围互不重叠。
src = [kv_src[start:start + width_elements] for start in source_offsets]
dst = [kv_dst[start:start + width_elements] for start in destination_offsets]
group_memcpy(src, dst)
~~~

此片段演示已有调用方如何传入视图；不重建外部的 cache 管理逻辑。

## Kernel 保留与修正

保留 184 KiB UB 上限、两份队列、单缓冲、64 字节 tile 对齐、任务按核号轮转，以及 `CopyIn → Compute(DataCopy) → CopyOut` 流程。若设备报告的 UB 更小，按实际容量裁减预算。tile 计算仍采用原来的 `ceil(W / maxTile)`、整数除法及 AlignUp64。

正式接入变化：

- 原交错裸地址表改为 `AscendC::ListTensorDesc`；TilingData 统一由 Host 定义，Kernel 用 `GET_TILING_DATA` 读取。
- 原代码在 `task_index < gbH` 前读取指针表；新代码只对满足范围的 task 解析地址，避免尾核越界。
- 原 `DataCopyPadExtParams` 未初始化；现在显式初始化 DMA padding。UB 内部仍按 64 字节复制；最后一个不足对齐长度的块先清零，再做 DMA，并以流水屏障保证初始化完成。仅真实字节写回 GM。
- 增加零字节处理，避免原分块路径除零；错误参数在 aclnn/ Torch 层拒绝。
- 不再暴露手工 blockDim；采用平台实际核数。同一 task 内分块搬运的有效字节范围不变。

## 实现依据

| 事实/选择 | 源码依据 |
| --- | --- |
| 仅字节复制、双队列、UB 上限 | 原 GroupMemcpy 单缓冲 Kernel 实现 |
| 原应用是 K/V cache 搬运 | 原 flash_attention.cc 的 GroupMemcpy launch 调用 |
| 工程组织 | experimental/moe/bias_sigmoid 的 README、tests 布局 |
| 手写 aclnn API 构建与排除自动生成接口 | experimental/attention/generic_block_sparse_attention/op_host/CMakeLists.txt |
| Tiling 注册及标准 Kernel 入口 | experimental/attention/chunk_local_cumsum |
| 动态列表、形状推导 | gmm/grouped_matmul |
| 动态输入/输出 binary 配置结构 | mc2/grouped_mat_mul_all_reduce/op_host/config/ascend910b |
| Torch 集成、选择性构建 | experimental/attention/chunk_gated_delta_rule_compute_wy/torch_extension、torch_extension/setup.py |
| aclnn 符号查找与描述符转换 | torch_extension/cann_ops_transformer/common/aclnn_common.h（复用，未修改） |

Torch 适配没有使用直调 Kernel 路线；它复用公共头中的符号查找和转换函数，以 `RunOpApiV2` 提交自有处理器。处理器持有 Tensor、TensorList 描述符及 workspace；对 NPU 缓存分配器记录 stream，避免临时对象在队列提交/设备执行完成前被回收复用。

接口约束见 [aclnn 文档](aclnnGroupMemcpy.md)，测试及原始 bin 回放见 [测试说明](../tests/README.md)。
