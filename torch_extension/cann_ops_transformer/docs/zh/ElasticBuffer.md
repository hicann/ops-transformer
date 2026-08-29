# ElasticBuffer

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR/Ascend 950DT</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：不支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：不支持
<!-- end id3 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2 推理产品</term>：不支持
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas 推理系列产品</term>：不支持
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas 训练系列产品</term>：不支持
<!-- end id6 -->

## 功能说明

ElasticBuffer类提供统一的分布式通信buffer管理能力：

- Engram存储接口用于分布式Engram存储管理，支持将本rank的表写入host pinned共享段，以及通过RDMA从远端rank抓取Engram数据。需与 [get_engram_storage_size_hint](#get_engram_storage_size_hint静态方法) 配套使用。
- Engram训练接口在推理接口的基础上，通过 `with_grad=True` 开启训练模式。前向 [engram_fetch](#engram_fetch) 在抓取数据的同时保存反向所需的通信元数据（封装为 [EngramFetchCtx](#engramfetchctx)），反向 [engram_fetch_grad](#engram_fetch_grad) 根据这些元数据将梯度沿前向路径反向交换并按local entry稀疏累加，产出稀疏梯度用于优化器更新。
- Dispatch/Combine接口用于MoE的Expert Parallelism（EP）并行部署，支持通过[dispatch](#dispatch)将token数据分发到对应专家卡，再通过[combine](#combine)将专家输出按原路由聚合回原始序列。需与[get_moe_ep_ccl_buffer_size](#get_moe_ep_ccl_buffer_size静态方法)配套使用。

## 函数原型

```python
class ElasticBuffer:
    def __init__(
        self,
        group: torch.distributed.ProcessGroup,
        *,
        num_cpu_bytes: int = 0,
        num_max_tokens_per_rank: Optional[int] = None,
        hidden: Optional[int] = None,
        num_topk: Optional[int] = None,
        with_grad: bool = False,
        explicitly_destroy: bool = False,
    )

    def engram_write(self, storage: torch.Tensor) -> None

    def engram_fetch(self, indices: torch.Tensor) -> Callable

    def engram_fetch_grad(
        self,
        grad_fetched: torch.Tensor,
        fetch_ctx: EngramFetchCtx,
    ) -> Tuple[torch.Tensor, torch.Tensor]

    def barrier(self, use_comm_stream: bool = True, with_cpu_sync: bool = False) -> None

    @staticmethod
    def get_engram_storage_size_hint(
        num_entries: int,
        hidden: int,
        dtype: torch.dtype = torch.bfloat16,
    ) -> int

    def dispatch(
        self,
        x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        *,
        topk_idx: Optional[torch.Tensor] = None,
        topk_weights: Optional[torch.Tensor] = None,
        handle: Optional[EPHandle] = None,
        num_experts: Optional[int] = None,
        num_max_tokens_per_rank: Optional[int] = None,
        expert_alignment: Optional[int] = None,
        do_cpu_sync: Optional[bool] = None,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
               Optional[torch.Tensor], Optional[torch.Tensor], EPHandle]

    def combine(
        self,
        x: torch.Tensor,
        handle: EPHandle,
        *,
        topk_weights: Optional[torch.Tensor] = None,
        bias: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], None] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]

    @staticmethod
    def get_moe_ep_ccl_buffer_size(
        world_size: int,
        num_max_tokens_per_rank: int,
        hidden: int,
        num_experts: int,
        topk: int,
    ) -> int

    def destroy(self) -> None
```

## EngramFetchCtx

前向 [engram_fetch](#engram_fetch) 在 `with_grad=True` 时返回的save-for-backward上下文，需原样传递给反向 [engram_fetch_grad](#engram_fetch_grad)。该对象由内部自动构造，无需手动创建。

| 属性 | 类型 | shape | 说明 |
|------|------|-------|------|
| `perm` | `Tensor` | `(num_tokens,)` int32 | 桶排列序索引，反向按此重排梯度 |
| `send_counts` | `Tensor` | `(world_size*8,)` int32 | 每rank发送的index数量（填充至32字节对齐，前`world_size`个元素有效），反向a2a的send量 |
| `recv_counts` | `Tensor` | `(world_size,)` int32 | 每rank接收的index数量，反向a2a的recv量 |
| `recv_local_entry` | `Tensor` | `(R_max,)` int32 | 前向a2a交换收到的全局索引，前 `num_recv` 个元素有效，其中 `R_max = num_max_tokens_per_rank × world_size` |
| `num_recv` | `Tensor` | `(1,)` int32 | 实际接收的全局索引数量 `num_recv`，其中`num_recv ≤ R_max` |

## 成员函数说明

### **init**

**功能**：构造ElasticBuffer实例，记录Engram和Dispatch/Combine所需配置。Engram运行时资源在首次调用 [engram_write](#engram_write) 时初始化；Dispatch/Combine通信上下文在首次调用 [dispatch](#dispatch) 或 [combine](#combine) 时初始化。

**输入参数**：

- **group** (`torch.distributed.ProcessGroup`)：必选参数，分布式进程组，用于跨rank通信和同步。
- <strong>*</strong>：其之前的变量是位置相关的；之后的变量是可选参数，需要使用键值对赋值，不赋值会使用默认值。
- **num_cpu_bytes** (`int`)：可选参数，CPU buffer大小（字节），用于host pinned存储区分配。默认值为0，且必须2MB对齐。
- **num_max_tokens_per_rank** (`int`)：可选参数，表示每张卡上的最大token数量上限。使用 [dispatch](#dispatch) 和 [combine](#combine) 时必须与 `hidden`、`num_topk` 一起指定；使用 `with_grad=True` 训练模式时必须指定且大于0。
- **hidden** (`int`)：可选参数，hidden size隐藏层大小。
- **num_topk** (`int`)：可选参数，表示选取topK个专家。
- **with_grad** (`bool`)：可选参数，是否开启训练。默认为 `False`（推理）。设为 `True` 时，前向 [engram_fetch](#engram_fetch) 会额外保存反向所需的通信元数据（封装为 [EngramFetchCtx](#engramfetchctx)），并可通过 [engram_fetch_grad](#engram_fetch_grad) 执行反向。
- **explicitly_destroy** (`bool`)：可选参数，是否需要显式调用 [destroy](#destroy) 释放资源。默认为 `False`，实例被垃圾回收时会自动调用 `destroy` 释放资源；设为 `True` 时，调用方需要显式调用 `destroy` 释放资源。推荐设置为 `True` 并在使用结束后手动调用 `destroy` 释放资源，以确保资源释放时机确定。

**输出**：无返回值，构造ElasticBuffer实例。

### engram_write

**功能**：将本rank的Engram表数据写入host pinned共享内存段，使其他rank可通过RDMA读取该数据。推理模式和训练模式均通过此接口写入storage。写入完成后，内部会自动获取通信上下文tensor、本地storage地址、通信buffer大小和rank数等信息，供后续 [engram_fetch](#engram_fetch) 使用。

> **host pinned共享内存段**：指通过 `aclrtMallocHost` 分配的页锁定（pinned）主机内存，并经 `aclrtHostRegisterV2` 映射后获得设备可访问地址。该内存段既可被本卡NPU直接访问，也可被远端rank的NPU通过RDMA读取，从而实现跨rank的零拷贝数据共享。其大小由构造函数的 `num_cpu_bytes` 参数指定。

**计算公式**：

$$HostPinnedBuf[0 : storage.nbytes()] \leftarrow storage.data()$$

即通过`memcpy_s`将`storage`的数据按字节拷贝到host pinned共享内存段起始位置。拷贝前后的两次 `Barrier` 保证所有rank写入完成且对彼此可见：

$$Barrier \rightarrow memcpy\_s(storage \rightarrow HostPinnedBuf) \rightarrow Barrier$$

其中 `storage.nbytes() = num_entries × hidden × dtype_size`，须满足 `storage.nbytes() ≤ num_cpu_bytes`。

**函数原型**：

```python
ElasticBuffer.engram_write(storage) -> None
```

**输入参数**：

- **storage** (`Tensor`)：必选参数，待写入的CPU tensor，shape为 `(num_entries, hidden)`，表示有 `num_entries` 个条目，每个条目维度为 `hidden`。要求2维、连续、`hidden`必须128对齐，数据类型支持 `bfloat16`、`float16`、`float32`。

**输出说明**：无返回值，数据写入host pinned内存。

### engram_fetch

**功能**：根据输入的全局索引，从对应rank抓取Engram数据。推理和训练的行为由构造时的 `with_grad` 决定：

- **推理**（`with_grad=False`）：通过RDMA one-sided read从远端rank的host pinned共享内存抓取数据。接口采用异步设计：调用后立即返回一个callable，执行该callable时阻塞等待RDMA传输完成并返回结果tensor。
- **训练**（`with_grad=True`）：通过all-to-all通信从远端rank抓取数据（桶排序 → a2a交换indices → 本地gather → a2a交换data → 还原），返回一个callable，执行该callable时等待传输完成并返回 `(fetched, fetch_ctx)` 二元组，其中 `fetch_ctx` 为save-for-backward上下文 [EngramFetchCtx](#engramfetchctx)。

**计算公式**：

每个全局索引按如下方式映射到目标rank和本地条目索引：

$$rank\_id = \lfloor global\_idx / num\_entries \rfloor$$

$$local\_idx = global\_idx \bmod num\_entries$$

$$fetched[i] = EngramTable[rank\_id][local\_idx]$$

其中各变量含义如下：

- $global\_idx$：输入索引张量 `indices` 的元素值，取值范围 $[0, world\_size \times num\_entries)$。
- $num\_entries$：[engram_write](#engram_write) 时各rank写入的条目数，即 `storage.size(0)`。
- $world\_size$：通信域中的rank总数。
- $rank\_id$：$global\_idx$ 映射到的目标rank编号，取值范围 $[0, world\_size)$。当 $rank\_id$ 为本rank时，数据直接从本地host pinned内存读取；当 $rank\_id$ 为远端rank时，通过RDMA跨卡读取。
- $local\_idx$：目标rank内的本地条目索引，取值范围 $[0, num\_entries)$。
- $EngramTable[rank\_id]$：目标rank通过 [engram_write](#engram_write) 写入host pinned共享内存的Engram表数据，shape为 `(num_entries, hidden)`，
$EngramTable[rank\_id][local\_idx]$ 表示其中第 $local\_idx$ 个条目。
- $fetched[i]$：输出张量中第 $i$ 个token对应的Engram数据，$i \in [0, num\_tokens)$，$num\_tokens$ 为 `indices` 的长度。

**训练**（`with_grad=True`）的索引映射与推理一致（见上述 $rank\_id$、$local\_idx$ 公式），区别在于通过all-to-all通信完成跨rank数据抓取，并保存反向上下文 [EngramFetchCtx](#engramfetchctx)。前向过程为：

1. 桶排序：按 $rank\_id_i = \lfloor indices[i] / num\_entries \rfloor$ 对 `indices` 分桶，记录桶排列序 $perm$ 与每rank发送量 $send\_counts$：

$$send\_counts[r] = |\{\,i \mid rank\_id_i = r\,\}|, \quad r \in [0, world\_size)$$

2. a2a交换indices：各rank将排序后的indices（全局索引）发送到目标rank，接收其他rank请求的全局索引，得到 $recv\_local\_entry$（均为映射到本rank的全局索引，即 `global_idx / num_entries == rank_id`）、$recv\_counts$，接收总量 $num\_recv = \sum_{r=0}^{world\_size-1} recv\_counts[r]$。

3. 本地gather：将 $recv\_local\_entry$ 中的全局索引按 $local\_idx = recv\_local\_entry[j] \bmod num\_entries$ 转换后，从本地 $EngramTable[rank\_id]$ 读取：

$$local\_data[j] = EngramTable[rank\_id][recv\_local\_entry[j] \bmod num\_entries], \quad j \in [0, num\_recv)$$

4. a2a交换data：将 $local\_data$ 沿原请求路径送回，得到 $recv\_data[i]$，$i \in [0, num\_tokens)$。

5. 还原：按 $perm$ 重排回原始token顺序：

$$fetched[perm[i]] = recv\_data[i], \quad i \in [0, num\_tokens)$$

6. 保存反向上下文 $fetch\_ctx$（即 [EngramFetchCtx](#engramfetchctx)），包含 $perm$、$send\_counts$、$recv\_counts$、$recv\_local\_entry$、$num\_recv$ 五个字段，供反向 [engram_fetch_grad](#engram_fetch_grad) 使用。

**函数原型**：

```python
ElasticBuffer.engram_fetch(indices) -> Callable
```

**输入参数**：

- **indices** (`Tensor`)：必选参数，查询索引的NPU tensor，shape为 `(num_tokens,)`，表示要抓取的条目全局索引。数据类型支持 `int32`，数据格式为 $ND$。元素取值范围需在 `[0, world_size × num_entries)`，若某一位置的元素取值超过了该范围，则返回值中该位置对应的数据为0。

**输出说明**：

- **wait_callable** (`Callable`)：返回一个callable，调用时根据模式返回不同结果。

**推理**调用 `wait_callable()` 返回：

- **fetched** (`Tensor`)：NPU tensor，shape为 `(num_tokens, hidden)`，数据类型与 [engram_write](#engram_write) 的 `storage.dtype` 相同，数据格式为 $ND$。

**训练**调用 `wait_callable()` 返回：

- **fetched** (`Tensor`)：NPU tensor，shape为 `(num_tokens, hidden)`，数据类型与 `storage.dtype` 相同，数据格式为 $ND$。
- **fetch_ctx** (`EngramFetchCtx`)：save-for-backward上下文，包含反向所需的5个通信元数据tensor（详见 [EngramFetchCtx](#engramfetchctx)）。需原样传递给 [engram_fetch_grad](#engram_fetch_grad)。

### engram_fetch_grad

**功能**：训练反向接口，需与训练模式的 [engram_fetch](#engram_fetch) 配套使用。根据前向保存的 [EngramFetchCtx](#engramfetchctx)，将 `grad_fetched` 沿前向路径反向交换并按local entry稀疏累加，产出稀疏梯度用于优化器更新。

**计算公式**：

反向沿前向路径逆序执行，$fetch\_ctx$ 各字段（$perm$、$send\_counts$、$recv\_counts$、$recv\_local\_entry$、$num\_recv$）均来自前向 [engram_fetch](#engram_fetch) 训练模式返回的 [EngramFetchCtx](#engramfetchctx)。依次为：

1. Unsort：按 $perm$ 将 $grad\_fetched$ 重排回桶排序顺序：

$$grad\_sorted[i] = grad\_fetched[perm[i]], \quad i \in [0, num\_tokens)$$

2. 反向a2a交换：将 $grad\_sorted$ 通过a2a反向交换（$send\_counts$ 为send量，$recv\_counts$ 为recv量），各rank收到对应 $recv\_local\_entry$ 的梯度：

$$grad\_recv[j] \leftrightarrow recv\_local\_entry[j], \quad j \in [0, num\_recv)$$

3. Unique：对 $recv\_local\_entry$ 按 $local\_idx = recv\_local\_entry[j] \bmod num\_entries$ 转换后去重，得到本rank local entry索引集合：

$$unique\_local\_entry = \text{unique}(\{\,recv\_local\_entry[j] \bmod num\_entries \mid j \in [0, num\_recv)\,\})$$

4. ScatterAdd：对每个unique entry，按FP32累加所有映射到该entry的梯度后转回输出dtype：

$$grad\_unique[k] = \text{Cast}_{dtype}\!\left(\sum_{\substack{j \in [0,num\_recv) \\ recv\_local\_entry[j] \bmod num\_entries = unique\_local\_entry[k]}} \text{Cast}_{fp32}(grad\_recv[j])\right)$$

其中 $K$ 为去重后的local entry数量（运行时决定，输出前 $K$ 行有效），$num\_tokens$ 为 `indices` 的长度，$num\_entries$ 为 [engram_write](#engram_write) 时各rank写入的条目数。`grad_unique` 与 `unique_local_entry` 可直接用于优化器稀疏更新。

**函数原型**：

```python
ElasticBuffer.engram_fetch_grad(grad_fetched, fetch_ctx) -> (Tensor, Tensor)
```

**输入参数**：

- **grad_fetched** (`Tensor`)：必选参数，前向 `fetched` 的梯度，NPU tensor，shape为 `(num_tokens, hidden)`，数据类型支持 `bfloat16`、`float16`、`float32`，数据格式为 $ND$。
- **fetch_ctx** (`EngramFetchCtx`)：必选参数，前向 [engram_fetch](#engram_fetch) 训练模式返回的上下文对象，包含 `perm`、`send_counts`、`recv_counts`、`recv_local_entry`、`num_recv` 五个字段。

**输出说明**：

- **grad_unique** (`Tensor`)：NPU tensor，shape为 `(K, hidden)`，数据类型与 `grad_fetched` 相同（内部按FP32累加后转回输出dtype），数据格式为 $ND$。其中 `K` 为去重后的local entry数量（运行时决定），前 `K` 行有效。
- **unique_local_entry** (`Tensor`)：NPU tensor，shape为 `(K,)`，数据类型为 `int32`，数据格式为 $ND$。表示 `grad_unique` 每行对应的local entry索引。

### barrier

**功能**：跨卡同步。

**函数原型**：

```python
ElasticBuffer.barrier(use_comm_stream=True, with_cpu_sync=False) -> None
```

**输入参数**：

- **use_comm_stream** (`bool`)：可选参数，表示是否使用专用通信stream执行barrier。默认值为 `True`，使用专用通信stream，barrier前后通过event同步计算流与通信流；设为 `False` 时使用当前计算stream。
- **with_cpu_sync** (`bool`)：可选参数，表示是否在barrier前后同步设备。默认值为 `False`。设为 `True` 时，在barrier前后各调用一次 `aclrtSynchronizeDevice`，确保设备侧操作完成。

**输出说明**：无返回值。

### get_engram_storage_size_hint（静态方法）

**功能**：计算Engram存储所需的CPU buffer大小。

**计算公式**：

```text
dtype_size = elementSize(dtype)              # 如bfloat16=2, float16=2, float32=4
hidden_size_bytes = hidden × dtype_size
num_bytes_per_entry = Align32(hidden_size_bytes)
num_cpu_bytes = Align2MB(num_bytes_per_entry × num_entries)
```

其中 `AlignX(value) = ((value + X - 1) / X) × X`，`/` 表示整除。`num_bytes_per_entry` 按32 字节对齐，最终结果按2MB对齐。

**函数原型**：

```python
ElasticBuffer.get_engram_storage_size_hint(
    num_entries, hidden, dtype=torch.bfloat16
) -> int
```

**输入参数**：

- **num_entries** (`int`)：必选参数，Engram storage的条目数，必须非负。
- **hidden** (`int`)：必选参数，每个条目的隐藏层维度，必须128 数量对齐且大于0。
- **dtype** (`torch.dtype`)：可选参数，数据类型，默认为 `torch.bfloat16`。仅在此处用于按dtype计算字节数。

**输出说明**：

- **num_cpu_bytes** (`int`)：CPU buffer大小（字节），用于engram_write的本地存储区，已2MB对齐。

### dispatch

**功能**：需与 [combine](#combine) 配套使用，完成MoE的Expert Parallelism（EP）并行部署下的token dispatch。该接口根据每个token的topK专家索引，将token数据通过EP域的alltoallv通信分发到对应的专家卡上。

- 支持cached模式，即第二次dispatch时可复用第一次的handle，跳过slot分配阶段，实现更低延迟。
- 支持指定 `num_max_tokens_per_rank` 控制接收buffer大小上限。

**计算公式**：

$$alltoall\_x\_out = alltoallv(x)$$

$$dst\_buffer\_slot\_idx = SlotAssignment(topk\_idx)$$

$$recv\_src\_metadata = MetadataAssignment(alltoall\_x\_out, topk\_idx)$$

**函数原型**：

```python
ElasticBuffer.dispatch(
    x,
    *,
    topk_idx=None,
    topk_weights=None,
    handle=None,
    num_experts=None,
    num_max_tokens_per_rank=None,
    expert_alignment=None,
    do_cpu_sync=None,
) -> (Tensor | Tuple[Tensor, Tensor], Tensor | None, Tensor | None, EPHandle)
```

**输入参数**：

- **x** (`Tensor` 或 `Tuple[Tensor, Tensor]`)：必选参数，表示计算使用的token数据，需根据 `topk_idx` 来发送给其他卡。当传入tuple时，第一个Tensor为token数据，第二个Tensor为scales。token要求为2 维张量，shape为 `(BS, H)`，数据类型支持 `bfloat16`、`float16`、`float8_e5m2`、`float8_e4m3fn`，数据格式为 $ND$。scales要求为2 维张量，shape为 `(BS, H1)`，数据类型支持`float32`或`float8_e8m0`，数据格式为 $ND$，只在token为`float8_e5m2`、`float8_e4m3fn`数据类型时传入。
- <strong>*</strong>：其之前的变量是位置相关的；之后的变量是可选参数，需要使用键值对赋值，不赋值会使用默认值。
- **topk_idx** (`Tensor`)：可选参数，表示每个token的topK个专家索引，决定每个token要发给哪些专家。要求为2 维张量，shape为 `(BS, K)`，数据类型支持 `int32`，数据格式为 $ND$。张量里value取值范围为 `[0, num_experts)`。非cached模式下为必选参数，cached模式下必须为 `None`。
- **topk_weights** (`Tensor`)：可选参数，表示每个token对应的topK专家权重。要求为2 维张量，shape为 `(BS, K)`，数据类型支持 `float32`，数据格式为 $ND$。非cached模式下为可选参数，cached模式下必须为 `None`。
- **handle** (`EPHandle`)：可选参数，表示上一次dispatch返回的handle对象，用于cached模式。传入handle时，`topk_idx` 和 `topk_weights` 必须为 `None`，`do_cpu_sync` 必须为 `False`。默认为 `None`，即非cached模式。
- **num_experts** (`int`)：可选参数，MoE专家总数量。取值范围 `[2, 2048]`，且满足 `num_experts % ep_world_size = 0`。非cached模式下为必选参数；cached模式下使用 `handle` 中的值，传入参数被忽略。
- **num_max_tokens_per_rank** (`int`)：可选参数，表示每张卡上的最大token数量上限，传入时覆盖ElasticBuffer初始化的值。默认使用初始化时传入的值。cached模式下使用 `handle` 中的值，传入参数被忽略。
- **expert_alignment** (`int`)：可选参数，表示专家对齐数。当前仅支持取值1；cached模式使用 `handle` 中的值。
- **do_cpu_sync** (`bool`)：可选参数，表示是否进行CPU同步等待。非cached模式默认为 `True`，cached模式下不能为 `True`。

**输出说明**：

- **recv_x** (`Tensor` 或 `Tuple[Tensor, Tensor]`)：表示本卡收到的token数据。Tensor shape为 `(A, H)`，数据类型与 `x` 一致，数据格式为 $ND$。经专家网络处理后，作为 [combine](#combine) 的 `x` 输入。当 `x` 输入包含scales时，输出为 `(recv_x, recv_scales)`。
- **recv_topk_idx** (`None`)：当前版本始终为 `None`，预留参数。
- **recv_topk_weights** (`Tensor | None`)：表示本卡收到的topK权重。仅当输入 `topk_weights` 不为 `None` 时返回，否则为 `None`。要求为1 维张量，shape为 `(A,)`，数据类型为 `float32`，数据格式为 $ND$，作为 [combine](#combine) 的 `topk_weights` 输入。
- **handle** (`EPHandle`)：表示dispatch阶段生成的handle对象，包含slot索引、元数据等信息，需传递给 [combine](#combine) 使用。handle的属性如下：
  - **dst_buffer_slot_idx** (`Tensor`)：slot索引，shape为 `(BS, K)`，dtype为 `int32`。
  - **recv_src_metadata** (`Tensor`)：接收元数据，shape为 `(A, 4)`，dtype为 `int32`。
  - **num_recv_tokens_per_rank** (`Tensor`)：各卡接收token数量，shape为 `(ep_world_size,)`，dtype为 `int32`。
  - **num_recv_tokens_per_expert** (`Tensor`)：每个本地专家接收的token数量，shape为 `(num_local_experts,)`，dtype为 `int64`。
  - **num_experts** (`int`)：专家总数量。
  - **expert_alignment** (`int`)：专家对齐数。
  - **num_max_tokens_per_rank** (`int`)：每张卡最大token数量上限。
  - **topk_idx** (`Tensor`)：原始topK索引。

### combine

**功能**：需与 [dispatch](#dispatch) 配套使用，相当于按dispatch算子收集数据的路径原路返回。该接口将专家处理后的token数据根据dispatch阶段记录的元数据信息，通过逆向路由和加权聚合，将token数据组合还原为原始序列顺序。

- 支持带 `topk_weights` 加权聚合和纯累加两种模式。
- 当前版本不支持bias参数。

**计算公式**：

当提供 `topk_weights` 时：

$$combined\_x_i = \sum_{k=0}^{K-1} topk\_weights_{i,k} \times x_{slot(i,k)}$$

当不提供 `topk_weights` 时（纯累加）：

$$combined\_x_i = \sum_{k=0}^{K-1} x_{slot(i,k)}$$

**函数原型**：

```python
ElasticBuffer.combine(x, handle, *, topk_weights=None, bias=None) -> (Tensor, Tensor?)
```

**输入参数**：

- **x** (`Tensor`)：必选参数，表示经过专家计算后的token数据，即 [dispatch](#dispatch) 输出的 `recv_x` 经过专家网络处理后的结果。要求为2 维张量，shape为 `(A, H)`，数据类型支持 `bfloat16`、`float16`，数据格式为 $ND$。
- **handle** (`EPHandle`)：必选参数，表示 [dispatch](#dispatch) 返回的handle对象，包含slot索引、接收元数据等信息。handle的属性参见 [dispatch](#dispatch) 输出说明。
- <strong>*</strong>：其之前的变量是位置相关的；之后的变量是可选参数，需要使用键值对赋值，不赋值会使用默认值。
- **topk_weights** (`Tensor`)：可选参数，表示每个token对应的topK专家权重，用于加权聚合。要求为1 维张量，shape为 `(A,)`，数据类型支持 `float32`，数据格式为 $ND$，对应 [dispatch](#dispatch) 的 `recv_topk_weights` 输出。若不提供，则进行纯累加combine，输出 `combined_topk_weights` 为 `None`。
- **bias** (`Tensor` 或 `Tuple[Tensor, Tensor]`)：可选参数，当前版本不支持bias，传入 `None` 即可。预留支持单个bias张量或 `bias_0`、`bias_1` 双张量模式。

**输出说明**：

- **combined_x** (`Tensor`)：表示combine后的token数据，还原为原始序列顺序。要求为2 维张量，shape为 `(BS, H)`，数据类型与 `x` 一致（`bfloat16` 或 `float16`），数据格式为 $ND$，不支持非连续的Tensor。
- **combined_topk_weights** (`Tensor | None`)：表示combine后的topK专家权重。当 `topk_weights` 输入不为 `None` 时，要求为2 维张量，shape为 `(BS, K)`，数据类型为 `float32`，数据格式为 $ND$；当 `topk_weights` 输入为 `None` 时，返回 `None`。

### get_moe_ep_ccl_buffer_size（静态方法）

**功能**：需与 [dispatch](#dispatch) 和 [combine](#combine) 配套使用，用于计算dispatch和combine算子所需的HCCL通信 `buffer_size` 大小（单位：MB）。该接口为静态方法，可在初始化ElasticBuffer前调用。

**函数原型**：

```python
ElasticBuffer.get_moe_ep_ccl_buffer_size(world_size, num_max_tokens_per_rank, hidden, num_experts, topk) -> int
```

**输入参数**：

- **world_size** (`int`)：必选参数，表示EP通信域的大小（即参与EP通信的卡数）。取值范围 `[2, 1024]`。
- **num_max_tokens_per_rank** (`int`)：必选参数，表示每张卡上的最大token数量上限。
- **hidden** (`int`)：必选参数，表示hidden size隐藏层大小。取值范围 `(0, 8192]`。
- **num_experts** (`int`)：必选参数，MoE专家总数量，取值范围 `[2, 2048]`，且满足 `num_experts % ep_world_size = 0`。
- **topk** (`int`)：必选参数，表示选取topK个专家，取值范围 `[1, 32]`。

**输出说明**：

- **ccl_buffer_size** (`int`)：计算得到的 `ccl_buffer_size` 大小，单位为MB。将该值设置为 `HCCL_BUFFSIZE` 环境变量即可满足通信域的内存需求。

**计算公式**：

```text
local_experts_num = num_experts // world_size

dispatch_count_size = world_size * Align512(local_experts_num * 4)
dispatch_notify_count = Align15000(num_max_tokens_per_rank) // 15000
dispatch_notify_size = world_size * 512 * (1 + dispatch_notify_count)
combine_state_size = num_max_tokens_per_rank * topk * 512 + world_size * 512
state_buffer_size = dispatch_count_size + dispatch_notify_size + combine_state_size

metadata_bytes = Align32(topk * 4)
hidden_align = Align32(hidden * 2)
dispatch_per_slot_bytes = Align512(hidden_align + metadata_bytes * 2 + 32)
combine_per_slot_bytes = Align512(hidden_align + 32)

dispatch_recv_buffer_size = world_size * num_max_tokens_per_rank * dispatch_per_slot_bytes
combine_recv_buffer_size = num_max_tokens_per_rank * topk * combine_per_slot_bytes
dispatch_send_buffer_size = dispatch_recv_buffer_size
direct_minimum_buffer_size =
    state_buffer_size
    + dispatch_recv_buffer_size
    + combine_recv_buffer_size
    + dispatch_send_buffer_size

combine_buffer_size = num_max_tokens_per_rank * topk * combine_per_slot_bytes
对 world_size 的每个因子 rnps（1 ≤ rnps ≤ world_size 且 world_size % rnps == 0）：
    scaleout_rank_count = world_size // rnps
    scaleout_per_slot_bytes = Align512(dispatch_per_slot_bytes + topk * 4)
    scaleout_recv_data_size = scaleout_rank_count * num_max_tokens_per_rank * scaleout_per_slot_bytes
    scaleout_recv_status_size = scaleout_rank_count * num_max_tokens_per_rank * 512
    payload_stash_size = num_max_tokens_per_rank * scaleout_per_slot_bytes
    dispatch_buffer_size(rnps) =
        scaleout_recv_data_size
        + dispatch_recv_buffer_size
        + scaleout_recv_status_size
        + payload_stash_size
hybrid_minimum_buffer_size =
    state_buffer_size
    + max(dispatch_buffer_size(rnps))
    + combine_buffer_size

minimum_buffer_size = max(direct_minimum_buffer_size, hybrid_minimum_buffer_size)

ccl_buffer_size = Align2(Align1MB(minimum_buffer_size) / 1MB) / 2
```

其中 `AlignX(value) = ((value + X - 1) / X) * X`，公式中的 `/` 表示整除。

由于运行时无法获知实际 scaleout 拓扑，hybrid 路径对 `world_size` 的所有合法因子 `rnps`（即 `rank_num_per_server`）枚举取最大值，按最大合法布局预留内存。通信窗口在 direct 拓扑下依次存放状态区、Dispatch接收区、Combine接收区和Dispatch发送区；hybrid 拓扑下布局为状态区、Scaleout Dispatch区、Combine区。Dispatch发送区与接收区均按
`dispatch_per_slot_bytes` 的最大2字节hidden规格预留；kernel实际读写和通信仍使用当前数据类型对应的
`per_slot_bytes`。Combine发送数据在Combine算子的workspace中暂存，不计入HCCL通信窗口大小。

### destroy

**功能**：释放ElasticBuffer资源，包括host pinned内存、Engram运行时资源和Dispatch/Combine通信上下文。训练模式下HCCL默认通信buffer由框架管理，无需手动释放。当构造时 `explicitly_destroy=False`（默认）时，实例被垃圾回收时会自动调用本方法；当 `explicitly_destroy=True` 时，需要由调用方显式调用。

**输入参数**：无参数。

**输出**：无返回值，资源释放完成。

## 约束说明

- **参数对齐约束**：
  - `num_cpu_bytes` 必须为2MB对齐（即能被 `2 × 1024 × 1024` 整除）。
  - `hidden` 必须为128 数量对齐。
  - [get_engram_storage_size_hint](#get_engram_storage_size_hint静态方法) 返回值自动满足2MB对齐。

- **Engram维度约束**：
  - `storage` 必须为2 维张量。
  - `indices` 必须为1 维张量。

- **Engram dtype约束**：
  - `storage.dtype` 仅支持 `bfloat16`、`float16`、`float32`。
  - `indices.dtype` 必须为 `int32`。

- **Engram设备约束**：
  - `storage` 必须在CPU上。
  - `indices` 必须在NPU上。

- **Engram调用顺序约束**：
  - 必须先调用 [engram_write](#engram_write) 至少一次，才能调用 [engram_fetch](#engram_fetch)。
  - 同一ElasticBuffer实例上不允许并发 [engram_fetch](#engram_fetch)（需等待上次fetch的callable执行完成）。
  - 训练模式下，必须先调用 [engram_fetch](#engram_fetch)（训练模式）获取 `fetch_ctx`，才能调用 [engram_fetch_grad](#engram_fetch_grad)。

- **Engram训练模式约束**：
  - `with_grad=True` 时，`num_max_tokens_per_rank` 必须为正整数。
  - `with_grad=True` 时，[engram_fetch](#engram_fetch) 返回的 `fetch_ctx` 为局部变量，多次fetch不会覆盖（每次调用返回独立的ctx）。
  - [engram_fetch_grad](#engram_fetch_grad) 的 `grad_fetched` shape 必须与前向 `fetched` 一致。
  - 训练模式下，通信buffer使用HCCL默认buffer（大小受 `HCCL_BUFFSIZE` 环境变量控制，默认200MB），由框架管理，无需手动申请或释放。

- **Engram数值约束**：
  - `num_cpu_bytes`、`num_entries`必须非负。
  - `hidden` 必须大于0。
  - `storage.nbytes()` 必须小于等于 `num_cpu_bytes`。`storage.nbytes()` 表示tensor实际占用的字节数，即 `num_entries × hidden × dtype_size`（其中
  `dtype_size` 为单个元素的字节大小，如 `bfloat16` 为2 字节、`float32` 为4 字节）。
  - 全局条目总数 `world_size × num_entries` 必须小于2^31（int32 最大值），保证indices索引不溢出。

- **Dispatch/Combine配套约束**：
  - [dispatch](#dispatch) 和 [combine](#combine) 必须配套使用。
  - 调用接口过程中使用的 `num_experts`、`num_max_tokens_per_rank` 参数取值所有卡需保持一致，且 [dispatch](#dispatch) 和 [combine](#combine) 对应参数也需保持一致。
  - 当前版本不支持bias参数，`bias` 必须传入 `None`。
  - cached模式下，`topk_idx` 和 `topk_weights` 必须为 `None`，`do_cpu_sync` 必须为 `False`；非cached模式下，`topk_idx` 为必选参数，`topk_weights` 为可选参数。

- **Dispatch/Combine Shape变量说明**：
  - `A`：表示本卡接收的最大token数量，`A = ep_world_size * num_max_tokens_per_rank * MIN(K, num_local_experts)`。
  - `H`：表示hidden size隐藏层大小。取值范围为 `(0, 8192]`。
  - `BS`：表示batch sequence size，即本卡的token数量。
  - `K`：表示选取topK个专家，取值范围为 `1 ≤ K ≤ min(32, num_experts)`。
  - `num_local_experts`：表示本卡专家数量，`num_local_experts = num_experts / ep_world_size`，其中 `num_experts` 取值范围为 `[2, 2048]` 且须被 `ep_world_size` 整除，即满足 `0 < num_local_experts * ep_world_size ≤ 2048`。

- **HCCL通信域缓存区大小**：
  - 调用 [dispatch](#dispatch) 或 [combine](#combine) 前需检查 `HCCL_BUFFSIZE` 环境变量取值是否合理，该环境变量配置单个通信域的 buffer 大小（单位MB，实际物理分配为 2 倍），不配置时默认为200MB。
  - 通信域缓存区大小可通过调用 [get_moe_ep_ccl_buffer_size](#get_moe_ep_ccl_buffer_size静态方法) 计算。
  - 计算得到的 `ccl_buffer_size` 需通过环境变量 `HCCL_BUFFSIZE` 设置，每个通信域独占一组 `2 * HCCL_BUFFSIZE` 大小的内存。

- **通信域约束**：
  - Engram通信域 `world_size` 范围 `[2, 1024]`，支持多卡分布式场景。
  - 一个模型中的 [dispatch](#dispatch) 和 [combine](#combine) 算子仅支持相同EP通信域，且该通信域中不允许有其他算子。

- **特殊场景处理**：
  - 支持 `num_entries = 0`。
  - 支持 `num_tokens = 0`。
  - 二进制一致：engram_write和engram_fetch全程纯数据搬运，输出与源必须逐字节相等，无任何容差。

## 调用示例

### Engram单算子模式调用（多卡分布式）

```python
import os
import torch
import torch_npu
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.multiprocessing import Process, Manager
from cann_ops_transformer import ElasticBuffer

num_entries = 10000
hidden = 4096
dtype = torch.bfloat16
world_size = 2


def set_device(rank):
    torch_npu.npu.set_device(rank)
    print(f"current device set: {torch_npu.npu.current_device()}")


def init_hccl_comm(rank, world_size):
    print(f"[INFO] device_{rank} create HCCL communication link")
    master_ip = "127.0.0.1"
    dist.init_process_group(
        backend="hccl",
        rank=rank,
        world_size=world_size,
        init_method=f"tcp://{master_ip}:50001",
    )
    print(f"device_{rank} init_process_group success")

    group = dist.new_group(backend="hccl", ranks=list(range(world_size)))
    return group


def run_elastic_buffer(queue, rank, world_size, storage, indices):
    print(f"{os.getpid()=}{rank=}")
    set_device(rank)
    group = init_hccl_comm(rank, world_size)

    num_cpu_bytes = ElasticBuffer.get_engram_storage_size_hint(
        num_entries, hidden, dtype
    )
    print(f"[INFO] device_{rank} num_cpu_bytes={num_cpu_bytes}")

    buffer = ElasticBuffer(group, num_cpu_bytes=num_cpu_bytes, explicitly_destroy=True)

    print(f"[INFO] device_{rank} run engram_write")
    buffer.engram_write(storage)

    print(f"[INFO] device_{rank} run engram_fetch")
    indices_npu = indices.npu()
    wait_callable = buffer.engram_fetch(indices_npu)
    fetched = wait_callable()

    torch.npu.synchronize()
    print(f"[INFO] device_{rank} fetched shape: {fetched.shape}")
    buffer.destroy()
    dist.destroy_process_group()

    queue.put([rank, fetched.cpu()])


if __name__ == "__main__":
    storage = torch.randn(num_entries, hidden, dtype=dtype)
    indices = torch.randint(
        0,
        num_entries * world_size,
        (1000,),
        dtype=torch.int32,
    )

    manager = Manager()
    result_queue = manager.Queue()
    mp.set_start_method("forkserver", force=True)

    proc_list = []
    for rank in range(world_size):
        p = Process(
            target=run_elastic_buffer,
            args=(result_queue, rank, world_size, storage.clone(), indices.clone()),
        )
        p.start()
        proc_list.append(p)

    results = [None] * world_size
    for _ in range(world_size):
        rank_id, fetched = result_queue.get()
        results[rank_id] = fetched
        print(f"[INFO] rank_{rank_id} result collected")

    for p in proc_list:
        p.join()

    if all(result is not None for result in results):
        print("All ranks finished successfully")
        for rank, result in enumerate(results):
            print(f"Rank {rank} fetched shape: {result.shape}")
    else:
        print("[ERROR] Task failed. Please check the detailed error logs.")
        exit(1)
```

### Engram训练模式调用（多卡分布式）

```python
import os
import torch
import torch_npu
import torch.distributed as dist
from torch.multiprocessing import Process
from cann_ops_transformer import ElasticBuffer

num_entries = 4
hidden = 128
dtype = torch.bfloat16
world_size = 2
num_tokens = 6


def set_device(rank):
    torch_npu.npu.set_device(rank)


def init_hccl_comm(rank, world_size):
    dist.init_process_group(
        backend="hccl",
        rank=rank,
        world_size=world_size,
        init_method=f"tcp://127.0.0.1:50003",
    )
    return dist.new_group(backend="hccl", ranks=list(range(world_size)))


def train_worker(rank):
    set_device(rank)
    group = init_hccl_comm(rank, world_size)
    device = f"npu:{rank}"

    # ==================== 生成 embedding 和 indices ====================
    torch.manual_seed(42 + rank)
    storage = torch.randn(num_entries, hidden, dtype=dtype)
    total_entries = num_entries * world_size

    torch.manual_seed(100 + rank)
    indices = torch.randint(0, total_entries, (num_tokens,), dtype=torch.int32)

    # ==================== 创建 ElasticBuffer（with_grad=True）====================
    num_cpu_bytes = ElasticBuffer.get_engram_storage_size_hint(
        num_entries, hidden, dtype
    )
    buffer = ElasticBuffer(
        group=group,
        num_cpu_bytes=num_cpu_bytes,
        num_max_tokens_per_rank=num_tokens,
        with_grad=True,
        explicitly_destroy=True,
    )

    # ==================== 写入 storage ====================
    buffer.engram_write(storage)
    dist.barrier()
    torch.npu.synchronize()

    # ==================== 训练前向 ====================
    indices_npu = indices.to(device)
    fetched, fetch_ctx = buffer.engram_fetch(indices_npu)()
    print(f"[rank {rank}] forward: fetched shape={fetched.shape}")

    # ==================== 训练反向 ====================
    # 假设 grad_fetched 来自 loss.backward()
    grad_fetched = fetched.clone()
    grad_unique, unique_local_entry = buffer.engram_fetch_grad(grad_fetched, fetch_ctx)
    print(f"[rank {rank}] backward: grad_unique shape={grad_unique.shape}, dtype={grad_unique.dtype}")
    print(f"[rank {rank}] backward: unique_local_entry={unique_local_entry.cpu().tolist()}")

    # ==================== 优化器更新（稀疏） ====================
    # storage[unique_local_entry] -= lr * grad_unique
    # 注意: grad_unique 与 grad_fetched 同 dtype, 需类型转换后更新 storage

    # ==================== 清理 ====================
    dist.barrier()
    torch.npu.synchronize()
    buffer.destroy()
    dist.destroy_process_group()


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("forkserver", force=True)
    proc_list = []
    for rank in range(world_size):
        proc = Process(target=train_worker, args=(rank,))
        proc.start()
        proc_list.append(proc)
    for proc in proc_list:
        proc.join()
    print("All processes finished.")
```

### Dispatch/Combine单算子模式调用（多卡分布式）

```python
import os
import torch
import torch_npu
import torch.distributed as dist
from torch.multiprocessing import Process
from cann_ops_transformer import ElasticBuffer

master_ip = "127.0.0.1"
world_size = 2
num_experts = world_size * 4
num_max_tokens_per_rank = 128
hidden = 4096
top_k = 4


def run_dispatch_combine(rank):
    torch_npu.npu.set_device(rank)
    dist.init_process_group(
        backend="hccl",
        rank=rank,
        world_size=world_size,
        init_method=f"tcp://{master_ip}:50002",
    )

    group = dist.new_group(backend="hccl")
    buffer = ElasticBuffer(
        group,
        num_max_tokens_per_rank=num_max_tokens_per_rank,
        hidden=hidden,
        num_topk=top_k,
        explicitly_destroy=True,
    )

    num_tokens = 64
    x = torch.randn(
        num_tokens,
        hidden,
        dtype=torch.bfloat16,
        device=f"npu:{rank}",
    )
    topk_idx = torch.randint(
        0,
        num_experts,
        (num_tokens, top_k),
        dtype=torch.int32,
        device=f"npu:{rank}",
    )
    topk_weights = torch.rand(
        num_tokens,
        top_k,
        dtype=torch.float32,
        device=f"npu:{rank}",
    )

    recv_x, _, recv_topk_weights, handle = buffer.dispatch(
        x,
        topk_idx=topk_idx,
        topk_weights=topk_weights,
        num_experts=num_experts,
    )
    torch.npu.synchronize()

    expert_output = recv_x
    combined_x, combined_topk_weights = buffer.combine(
        expert_output,
        handle,
        topk_weights=recv_topk_weights,
    )

    torch.npu.synchronize()
    print(f"[rank {rank}] combined_x shape={combined_x.shape}, expected ({num_tokens}, {hidden})")
    assert combined_x.shape == (num_tokens, hidden)
    print(f"[rank {rank}] dispatch_combine PASS")

    buffer.destroy()
    dist.destroy_process_group()


if __name__ == "__main__":
    os.environ["HCCL_WHITELIST_DISABLE"] = "1"
    os.environ["HCCL_BUFFSIZE"] = str(
        ElasticBuffer.get_moe_ep_ccl_buffer_size(
            world_size,
            num_max_tokens_per_rank,
            hidden,
            num_experts,
            top_k,
        )
    )

    processes = []
    for rank in range(world_size):
        p = Process(target=run_dispatch_combine, args=(rank,))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print("All processes finished.")
```
