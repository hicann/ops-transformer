# LightningIndexer（Experimental）

本目录与主库 `attention/lightning_indexer` 的算子类型均为 `LightningIndexer`，但接口不同。
本版本提供 `kv_block_len`、`q_block_len`、`init_num`、`local_num`、`return_value` 等参数。
编译时用 `--experimental` 选择本目录；运行时使用本版本生成的自定义算子包。

## 功能与接口

计算 Query 与 Key 的点积、ReLU、逐头加权归约，然后按每个 Query token 选取 TopK key/block 索引，可选返回分值。支持 block 聚合、sink 和 local-window 参数。

```python
torch.ops.cann_ops_transformer.lightning_indexer(
    query, key, weights,
    cur_seq_lengths_query=None,
    cur_seq_lengths_key=None,
    block_table=None,
    layout_query="BSND",
    layout_key="PA_BSND",
    sparse_count=2048,
    kv_block_len=1,
    q_block_len=1,
    init_num=0,
    local_num=0,
    sparse_mode=3,
    pre_tokens=9223372036854775807,
    next_tokens=9223372036854775807,
    return_value=False,
)
```

返回 `(sparse_indices, sparse_values)`：indices 为 int32，values 与 query dtype 相同。

| 参数 | dtype / 布局 |
| --- | --- |
| query | FP16/BF16；`BSND` 或 `TND`，D=128 |
| key | 与 query dtype 相同；`BSND`、`TND` 或 `PA_BSND`，KV head 数为 1 |
| weights | FP32，shape 为 query.shape 去掉末维 |
| cur_seq_lengths_query / key | 可选 Tensor，Torch 层转为 int64；TND/PA 对应必填条件由 Tiling 检查 |
| block_table | PA 场景使用 int32 Tensor |
| sparse_indices / sparse_values | BSND：`[B,S1,N2,K]`；TND：`[T1,N2,K]` |

`kv_block_len` 对应 ACLNN/OpDef 中的 `block_len`，取值为 1、2、4、8、16；`q_block_len` 当前为 1。`sparse_count` 在 1～2048 时允许任意整数，2048～4096 范围需为 128 的倍数。其他参数约束由 Tiling 逻辑检查。

## 目录与实现

```text
lightning_indexer/
  CMakeLists.txt
  op_host/                  # OpDef、InferShape/InferDataType、Tiling
  op_kernel/                # Ascend C 内核与排序实现
  torch_extension/           # Python schema/Meta + C++ ACLNN 桥接
    csrc/lightning_indexer.cpp
  examples/compile_torch_extension.py
  tests/test_npu_lightning_indexer_TND.py
  tools/run_tnd.sh
  tools/build_torch.sh
  tools/build_standalone.sh
  tools/standalone/CMakeLists.txt
```

- Host 层包含算子定义、shape/dtype 推导和 Tiling；CMake 通过 `add_op_to_compiled_list`、`add_modules_sources` 接入仓库构建，生成 `aclnnLightningIndexer` API。
- Torch 层注册 schema、Meta 和 NPU 后端实现；可选序列长度仅在有值时转为 int64。
- C++ 桥接优先使用 CANN 头文件，避免 torch_npu 附带旧 ACL 头覆盖新类型定义。
- 独立打包脚本仅在临时构建副本中处理 vendor 包的相对导入，公共框架源码保持不变。

## 与主库 A2（arch22）LI 的排序 / TopK 对比

本节对比当前实验实现与主库的
[`attention/lightning_indexer/op_kernel/arch22`](../../../attention/lightning_indexer/op_kernel/arch22)，
不包含 `arch35` 的 TopK 实现。主库 [kernel 入口](../../../attention/lightning_indexer/op_kernel/lightning_indexer.cpp)（L20）
通过架构宏选择 `arch22` / `arch35`，A2 对应本节分析的 `arch22` 路径。

**主要变化：两者都沿 S2 流式处理，但主库 A2 通常每个基本块都排序并维护有序 TopK；
实验版在 `sparse_count <= 2048` 时维护无序候选，批量执行基于分数阈值的选择，
到局部结果输出前再精排。`sparse_count > 2048` 仍走逐块排序、归并。**

### 1. 主库 A2 已有的排序流程

主库 [ProcessVec](../../../attention/lightning_indexer/op_kernel/arch22/lightning_indexer_service_vector_arch22.h)（L263）
有两条路径，不能把它概括为“把整条 S2 一次性全排”：

| 条件 | 主库 A2 的实际处理 |
| --- | --- |
| `actS1Size > 4`，或 `sparse_count > 2048` | 对当前 S2 基本块执行 `SortAll`，再 `MergeSort` 到该 token 的有序候选中，保留 `virTopK` 项 |
| `actS1Size <= 4` 且 `sparse_count <= 2048` | 每块仍先排序，但可缓存四块 512 项；缓存满、S2 结束或本核任务结束时，通过 `MrgBasicBlock` 和 `SparseTopK` 合并 |

主库 [SortAll](../../../attention/lightning_indexer/op_kernel/arch22/lightning_indexer_vector.h)（L185）
先用 `Sort32` 生成 32 项有序段，再用 `MrgSort` 做多级四路归并；
[MergeSort](../../../attention/lightning_indexer/op_kernel/arch22/lightning_indexer_vector.h)（L267）
将新块与历史候选合并后截取前 `virTopK`。
在 `sparse_count <= 2048` 时，主库的 `virTopK` 同样固定为 2048。

因此，主库已经有流式 TopK 和 decode 场景的批量归并优化；实验版进一步减少的是
**中间阶段完整排序、反复维护内部顺序的工作**。

### 2. 无序候选与批量阈值筛选

对应代码：[InitBuffers](op_kernel/lightning_indexer_service_vector.h)（L178）、
[ProcessVec 的选择分支](op_kernel/lightning_indexer_service_vector.h)（L550）、
[ExecuteQS](op_kernel/lightning_indexer_service_vector.h)（L940）。

定义 `K = sparse_count`、`V = virTopK`、`B = 512 / block_len`。
当 `K <= 2048` 时，**V 固定为 2048，而不是 K**；例如测试的 K=128 仍维护 2048 项主体候选。

每个 token 的处理步骤是：

1. **先填候选，不排序。** 前 `V/B` 个 KV 基本块直接追加分数和索引；`block_len=1` 时为前四块。
2. **缓存新数据，合批筛选。** `QS_NUM_CACHE_SLOTS=2`。先暂存两块，第三块到达时，
   将“主体候选＋overflow＋两块缓存＋当前块”一起送入 `ExecuteQS`，随后清空缓存计数。
   正常持续流入时，每三个新基本块调用一次 QS。
3. **仅保留足够大的分数，不维护内部顺序。** QS 收敛后，保留数量落在 `[2048,2432]` 的候选。
4. **局部结果输出前统一排序。** 将主体、overflow 和尾部未处理缓存拼接，补齐到 4096 项，
   执行 `SortAll(4096)`，保留前 2048 项，后续按 K 输出或写入跨核合并工作区。

这里 QS 是函数 `QuickSelectPartition` 的代码命名，实际实现是**分数域阈值搜索＋向量压缩**，
没有采用传统 QuickSelect 的原地交换分区递归，也没有改成 radix sort 或堆排序。

### 3. 用 CompareScalar / GatherMask 替代每轮完整排序

[QuickSelectPartition](op_kernel/lightning_indexer_vector.h)（L464） 对一个阈值 `pivot` 执行：

```text
mask = (score >= pivot)
C(pivot) = mask 中满足条件的项数

C < 2048          → 阈值偏高，需要降低
C > 2048 + 384    → 阈值偏低，需要提高
2048 <= C <= 2432 → 接受本轮候选，用相同 mask 压缩 scores 和 indices
```

`CompareScalar` 生成 mask，探测阶段通过 `GatherMask` 的 `rsvdCnt` 获取计数；
收敛后才执行完整 FP32 score 和 uint32 index 的两次压缩。
实际探测仍有 GatherMask 写入工作区，不能理解成没有搬运开销。

中间候选采用分离存储 `[scores | indices]`，便于连续扫描 score，并对 index 应用同一个 mask。
主体候选前 2048 项此时**不一定是已经排好或最好的 2048 项**，高分项也可能位于 overflow，
因此后续筛选和最终排序都必须同时纳入两部分。

`Sort32 + MrgSort` 原语依然用于精排和回退；优化发生在调用频率和候选维护策略，
不是将这些排序原语本身替换成更快的实现。

### 4. 384 项宽容窗口：减少找阈值的迭代次数

`QS_OVF=384` 表示允许多保留的**元素个数**，不是分数误差阈值。
代码不要求每次筛选恰好得到 2048 项，而是接受 `[2048,2432]` 项，
多出的项放在独立 `overflowBuf_` 中参与下一轮筛选和最终排序。

按分数选择的逻辑是：若保留了所有 `score >= pivot` 的项，且数量至少为 2048，
则被丢弃的更低分项不会进入当前这批数据的前 2048。额外保留候选可放宽阈值收敛条件，
而无需在中间阶段决定这些候选的精确名次。最终仍用完整排序截断，设计上不是近似 TopK。
如果相同分数过多，计数可能跨过整个接受区间；这类情况依赖下述完整排序回退。
源码分析不等同于对所有同分输入的 index 次序作出保证。

### 5. 阈值搜索的加速：历史边界、哨兵过滤、割线插值

对应代码：[首次边界初始化](op_kernel/lightning_indexer_vector.h)（L484）、
[探测与阈值更新](op_kernel/lightning_indexer_vector.h)（L519）。

- **复用每个 token 的历史信息。** 首轮用主体候选的 `ReduceMin` 和过滤哨兵后的 `ReduceMax`
  初始化搜索范围；后续用 `lastThreshold_` 和 `sugMaxCarry_`，减少重复归约。
  切换到新的 S1 任务时重置这些状态。
- **过滤人工高分，只用于初始化上界。** sink/local 区域会被赋予接近 FP32 最大值的高分，
  对角位置赋予更高分。首次求最大值时排除这些哨兵，避免把搜索范围拉到极大值；
  实际筛选仍保留这些高分项，并没有丢弃它们。
- **上界不足时先扩界。** 正数阈值翻倍，负数阈值减半向零靠近，接近零时从 1 开始。
  因此 `128.0f` 是缺少 carry 时的建议起点，不是固定得分上限。
- **利用计数信息更新阈值。** 建立上下界后，用最近两次实测的 `(pivot, count)` 做割线外推，
  分母为零时用中点；越界的预测按区间宽度的 1/8 向边界内侧收回。
  相比仅用“选多/选少”做二分，这种方法还利用了计数距离目标有多远。
- **限制重试。** 每次 QS 最多 32 轮；浮点区间无法继续收窄或用完迭代次数时返回失败，转入全排。

这些机制旨在减少对候选数组的重复扫描，实际轮数依赖分数分布。不能将该实现直接描述为
保证单遍或保证线性时间的选择算法。

### 6. 正确性回退和输出格式衔接

[ExecuteQS 的失败分支](op_kernel/lightning_indexer_service_vector.h)（L991）
会**立即**把旧主体、旧 overflow、缓存及当前新块重新拼接，执行 `SortAll(4096)`，
再通过 `Extract` 写回前 2048+384 项的分离格式。失败时的新数据不会被直接丢弃。

当前最坏输入容量为：

```text
2048 主体 + 384 overflow + (2 缓存块 + 1 当前块) × 512 = 3968 < 4096
```

代码对 `QS_CAP_MAX <= 4096` 有编译期限制。
最终精排若仍有两块未处理缓存，最多拼入 `2048 + 384 + 2×512 = 3456` 项，
剩余位置用 `-inf/-1` 填充，再统一做 4096 项排序。

[输出前精排](op_kernel/lightning_indexer_service_vector.h)（L580）
在 `isS2End` **或本核任务结束**时发生，不一定整条序列只执行一次。
若一个 token 的 S2 被分给多个核，每个局部结果都先恢复成有序的 score/index 交织格式，
随后 [ProcessLD](op_kernel/lightning_indexer_service_vector.h)（L745）
仍使用四路 `MrgSort` 合并各核结果。这部分与主库的总体方法相同，并未被 QS 替代。

输出前还有对角索引位置交换，因此“最终精排”也不表示输出 API 在所有位置上
都保持普通分数降序排列。

### 7. 对比小结与适用边界

| 项目 | 主库 A2 LI | 本目录实验版 LI |
| --- | --- | --- |
| K<=2048 的主体容量 | 固定 2048 | 固定 2048，另保留最多 384 项 |
| 主体初始化 | 每块排序后归并，或 decode 场景先排序缓存 | 先直接追加到满 2048，无需排序 |
| 后续新块 | 普通场景逐块 SortAll+MergeSort；decode 可四块合批归并 | 两块缓存＋当前块合批 QS，未收敛时全排回退 |
| 中间候选 | 有序，score/index 交织 | 无序，score/index 分离，主体和 overflow 共同参与筛选 |
| 输出局部结果 | 直接从已有有序结果提取 | SortAll(4096) 后恢复有序交织格式，再提取或写工作区 |
| K>2048 | 逐块排序和归并 | 仍逐块排序和归并，不适用上述 QS 优化 |
| 跨核结果合并 | 四路 MrgSort | 仍使用四路 MrgSort |

作为调用次数示例，假设 K=2048、block_len=1，某个 token 的 8192 个有效 key
全部由一个核连续处理，无提前输出且 QS 均正常收敛：主库普通路径需要
16 次 512 项块排序和 16 次历史候选归并；实验版前四块直接填充，后十二块触发
4 次 QS，最后执行 1 次 4096 项精排。**这是指定条件下的调用计数，不是加速倍数。**
主库 decode 缓存路径、因果 mask、跨核切分或 QS 回退都会改变这个对比。

配套的资源取舍也发生了变化：[实验版 K<=2048 的 S1 基本块为 4](op_kernel/lightning_indexer_kernel.h)（L186），
[主库为 8](../../../attention/lightning_indexer/op_kernel/arch22/lightning_indexer_kernel_arch22.h)（L186）。
实验版为候选缓存、overflow 和 QS 工作区单独分配 UB，输入缓冲改为独立 `inQueue_`，
不再与排序工作区共用同一布局。因此不能只根据“排序次数减少”就推导整体 kernel 一定更快；
Query 分块、UB 占用、QS 迭代、拷贝和同步开销都要计入。

`block_len>1` 还会在选择前做 block 分数归约，减少进入选择的数据项，但这是新增的
block-indexer 语义，不应当当作同一个 token-level TopK 问题上的纯排序加速。
sink/local/对角必选也会改变选择规则，不能把两版输出直接视为完全等价。

阅读本版本时还需注意几处源码注释与执行逻辑不同：

- 缓存槽的实际常量是 **2**；少数注释仍写 3。加上当前块后，一次 QS 最多纳入三块新数据。
- 尾块若刚好遇到缓存已满，仍会先执行 QS，再执行输出前精排；不能按注释理解为尾块绝不调用 QS。
- 虽有注释写“未收敛保持旧候选”，实际 `else` 分支已实现立即全排回退。
- `carrySafeCnt` 当前按两块新数据估算，而实际触发可能带三块；据此不能保证复用上界后永不扩界。
  代码仍重新计数，并保留扩界和完整排序回退。此处只指出估算与实际批量的差异，未修改源码。

现有验证是实验版与自身 CPU golden 的逐 token index 对比：`seq_q=1024`、K=128、
`block_len=1`、`init_num=4`、`local_num=32`，`seq_k=4096/8192` 各 131072 个 index 及顺序均一致，value 未比较。
这些精度结果不等同于性能对照；两版实测见下节。短序列、局部分片很短、
分数大量相同以及 K>2048 时，不能仅据排序策略推断性能收益。

## 构建

在已准备好仓库全部依赖的环境，可使用仓库标准入口：

```bash
source /usr/local/Ascend/cann/set_env.sh
bash build.sh --pkg --experimental --ops=lightning_indexer --soc=ascend910b -j8
```

也可使用独立构建入口，直接调用已安装 CANN 的 `npu_op_*` CMake 接口编译本目录 Host 和 Kernel 源码：

```bash
source /usr/local/Ascend/cann/set_env.sh
bash experimental/attention/lightning_indexer/tools/build_standalone.sh \
  /workspace/lightning_indexer_build ascend910b
```

第二个位置参数也接受 `ascend910_93`，已验证的构建目标为 `ascend910b`。构建产物：

```text
<BUILD_DIR>/
  libcust_opapi.so
  libcust_opmaster_rt2.0.so
  libcust_opsproto_rt2.0.so
  li_kernel/binary/ascend910b/lightning_indexer/*.o
  stage/packages/vendors/experimental_lightning_indexer/
  package/custom_opp_*.run
```

独立构建直接部署到 `<BUILD_DIR>/stage`，不需要安装到系统 CANN 目录。后续若调用计算接口，先设置：

```bash
export ASCEND_CUSTOM_OPP_PATH=/workspace/lightning_indexer_build/stage/packages/vendors/experimental_lightning_indexer
export LD_LIBRARY_PATH="$ASCEND_CUSTOM_OPP_PATH/op_api/lib:$LD_LIBRARY_PATH"
```

## Torch wheel 与编译检查

在仓库根目录执行：

```bash
bash experimental/attention/lightning_indexer/tools/build_torch.sh \
  /workspace/lightning_indexer_python codex_li /workspace/lightning_indexer_wheels

export PYTHONPATH=/workspace/lightning_indexer_python:$PYTHONPATH
export TORCH_EXTENSIONS_DIR=/workspace/lightning_indexer_torch_cache
export CANN_OPS_TRANSFORMER_PACKAGE=cann_ops_transformer_codex_li
export MAX_JOBS=4
python3 experimental/attention/lightning_indexer/examples/compile_torch_extension.py
```

独立打包入口调用 `experimental/tools/build_torch_vendor.py`，在临时目录构建并清理副本。
wheel 保存在指定的 `/workspace/lightning_indexer_wheels`；不修改公共框架源码，
也不在源码仓库生成 build/dist、egg-info 或 vendor 链接。

该检查脚本验证 BSND/BSND、TND/TND、TND/PA_BSND 三种 Meta 输出 shape/dtype，并显式编译、加载 C++ 桥接 `.so`。它不执行 NPU 计算；实际运行使用下文的 `tools/run_tnd.sh`。

## 执行与逐 token index 检查

`tests/test_npu_lightning_indexer_TND.py` 使用 CPU FP32 golden 生成参考结果，
通过 `torch.ops.cann_ops_transformer.lightning_indexer` 获取 NPU 输出，再逐 token 比较 index。
默认执行 `seq_k=4096,8192`；测试脚本限制 `seq_k <= 8192`，不改变算子的长度支持范围。

在仓库根目录执行，路径与前面的独立构建、Torch 打包示例对应：

```bash
LI_NPU_DEVICE=11 LI_T2_LIST=4096,8192 bash \
  experimental/attention/lightning_indexer/tools/run_tnd.sh \
  /workspace/lightning_indexer_build \
  /workspace/lightning_indexer_python \
  cann_ops_transformer_codex_li
```

根据可用设备调整 `LI_NPU_DEVICE`。脚本的三个位置参数依次为算子构建目录、
Python 包安装目录、Python 包名；安装目录应为包目录的父目录。
脚本自动加载 CANN 环境并设置自定义 OPP、动态库和 Python 包路径。

测试配置：BF16、`seq_q=1024`、64 个 Query head、D=128、TopK=128、
`kv_block_len=q_block_len=1`、`init_num=4`、`local_num=32`。
实际布局为 **query=TND，key=PA_BSND**，key 打包为 page size 128 的缓存。

检查方式：

1. **严格数组相等**：逐 token、逐 TopK 位置比较整数 index，包含顺序；任一不同则用例失败。
2. **忽略顺序后的索引相等**：分别对每个 token 的索引排序后比较，用于区分选中索引不同和仅顺序不同。
   排序保留重复索引与 padding 数量。
3. 汇总不一致 token 数、位置数和仅顺序不一致 token 数；两个用例比较结束后以退出状态表示结果。

value 不参与比较，也不传回 CPU。以下配置已在 Ascend 910B2C、CANN 8.5.0、
PyTorch 2.9.0 / torch_npu 2.9.0 环境验证：

| seq_k | Query token 数 | 比较 index 总数 | index 数组不一致 token 数 | 忽略顺序后不一致 token 数 | 不一致位置数 |
| --- | --- | --- | --- | --- | --- |
| 4096 | 1024 | 131072 | 0 | 0 | 0 |
| 8192 | 1024 | 131072 | 0 | 0 | 0 |

两个用例逐 token 的 index 数组和顺序均一致，退出码为 0。
这些精度结果仅覆盖上述配置，完整测试集和图模式未验证。性能测试的配置与口径见“A2 长序列性能测试”。
