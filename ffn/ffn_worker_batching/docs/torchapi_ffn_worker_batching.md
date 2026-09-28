# ffn_worker_batching

## 产品支持情况

本页的异步混层示例支持 Ascend 950PR&950DT系列产品。

## 功能说明

根据调度上下文中的专家号排序，重排 token、scale 和来源索引，并统计各专家的 token 数。算子只搬运数据，不执行 FFN 矩阵计算或量化计算。

异步接收（`need_schedule=1、sync_flag=True`）允许同一 micro batch 中已就绪的不同 session 属于不同层，使用跨层专家号：

```text
experts_per_layer = expert_num / layer_num
new_expert_id = layer_id * experts_per_layer + expert_id
```

`expert_num`为本卡参与层的专家总数；`expert_id`为层内本卡专家号。`layer_id`位于各 session 的`FfnDataDesc`描述符中。

## 函数原型

```python
cann_ops_transformer.ffn_worker_batching(
    schedule_context: torch.Tensor,
    expert_num: int,
    max_out_shape: List[int],
    *,
    token_dtype: int = 0,
    need_schedule: int = 0,
    layer_num: int = 0,
    sync_flag: bool = False,
    mx_output_uint8: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
           torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
```

## 参数说明

| 参数名 | 类型 | 必选/可选 | 描述 |
| --- | --- | --- | --- |
| schedule_context | Tensor | 必选 | 连续一维 NPU int8 张量，至少1024字节，按`ScheduleContext`布局保存维度及有效Device地址。 |
| expert_num | int | 必选 | 专家数，范围[1,8192]。异步混层模式表示本卡所有参与层的专家总数。 |
| max_out_shape | List[int] | 必选 | `[A, BS, K, H]`，分别为session容量、micro batch大小、每token的专家选择数、逻辑hidden维度。 |
| token_dtype | int | 可选 | 0：FP16；1：BF16；2：INT8；3：FP8 E5M2；4：FP8 E4M3FN；5：FP4 E2M1。示例使用0。 |
| need_schedule | int | 可选 | 0：NORM；1：RECV。示例使用1，直接消费接收描述符。 |
| layer_num | int | 可选 | 参与层数。异步RECV必须大于0且整除`expert_num`；示例使用2。 |
| sync_flag | bool | 可选 | False：同步接收，等待全部session；True：异步接收，处理ready快照选中的session。仅RECV解释该属性。 |
| mx_output_uint8 | bool | 可选 | 默认False。True时MX scale以uint8返回，FP4 y也以uint8返回，保持原始编码；FP16示例无需设置。 |

## 返回值说明

令`Y=A*BS*K`，返回值顺序如下。示例采用FP16，输出shape如下表。

| 返回值 | dtype | shape | 描述 |
| --- | --- | --- | --- |
| y | float16 | `[Y,H]` | 按跨层专家号排序后的token。 |
| group_list | int64 | `[expert_num,2]` | 每个非空专家占一行，内容为`[跨层专家号, token数]`，其余行补零。 |
| session_ids | int32 | `[Y]` | 原始session编号。 |
| micro_batch_ids | int32 | `[Y]` | 原始micro batch编号。 |
| token_ids | int32 | `[Y]` | 原始micro batch内的token编号。 |
| expert_offsets | int32 | `[Y]` | token的专家选择位置，范围[0,K)。 |
| dynamic_scale | float32 | `[Y]` | FP16/BF16下无有效含义。量化数据的scale布局见[V2接口文档](aclnnFfnWorkerBatchingV2.md)。 |
| actual_token_num | int64 | `[1]` | 本次有效输出行数。 |

`y`、来源索引及有意义的scale只读取前`actual_token_num`行，不读取未使用的输出容量。

## 约束说明

- 各层在本卡的专家数相同。`layer_id`为[0,layer_num)内的连续编号，输入专家号必须为层内局部编号，不能预先加层偏移。
- 非法层号使该已选中描述符的token全部被mask；专家号不在[0,experts_per_layer)内时被mask，包含原有mask值。
- 上下文中的地址必须来自Device分配，不能直接填CPU指针。上下文以及它引用的token、描述符缓冲区必须保持存活，直到执行完成。
- 接收过程会清理已选中描述符的flag及expert_ids，并推进轮询游标。重复调用或图replay前必须重新发布数据；没有ready数据时算子会继续等待。
- 示例在同一stream上先准备数据再调用消费者，以演示异步选择语义。实际跨stream或跨设备生产者应遵守“先完成数据写入，再发布ready”的接收协议。
- 同步RECV和NORM保留原专家编号规则，不使用该跨层公式。

## 调用示例

按[Torch扩展构建说明](../../../torch_extension/README.md)安装whl，并加载配套的自定义OPP环境后，从仓库根目录运行：

示例设置`expert_num=8、layer_num=2`，每层4个专家。session 0属于layer 1，session 1属于layer 0，两者使用相同的层内专家号`[[1,0],[3,1]]`。micro batch 0未就绪，micro batch 1仅这两个session就绪，session 2不应被消费。

核心调用为：

```python
y, group_list, session_ids, micro_batch_ids, token_ids, expert_offsets, dynamic_scale, actual_token_num = (
    ffn_worker_batching(
        schedule_context, 8, [3, 2, 2, 256],
        token_dtype=0, need_schedule=1, layer_num=2, sync_flag=True,
    )
)
```

预期有效token数为8，排序后的跨层专家号为`[0,1,1,3,4,5,5,7]`；`group_list`的非空行为：

```text
[[0,1], [1,2], [3,1], [4,1], [5,2], [7,1]]
```

脚本同时校验token搬运、来源索引、group_list、已消费描述符清理以及未选中描述符保持不变；失败时抛出异常，不跳过。

### Torch eager调用方式

```python
import importlib
import math
import os
import struct

import torch
import torch_npu  # noqa: F401: registers NPU and npugraph_ex

# ScheduleContext ABI: common/include/op_kernel/attention_ffn_schedule.h.
# The first 6 uint32 words describe dimensions and the byte stride of token data.
CONTEXT_BYTES = 1024
FFN_AREA_OFFSET = 384
POLLING_INDEX_OFFSET = FFN_AREA_OFFSET + 4 * 8
MASK_ID = torch.iinfo(torch.int32).max


class AsyncLayerExample:
    """Keep all device allocations alive while context contains their pointers."""

    def __init__(self, device):
        self.sessions, self.batches, self.bs, self.k, self.h = 3, 2, 2, 2, 256
        self.layers, self.experts = 2, 8
        self.selected_batch = 1
        self.selected_sessions = (0, 1)
        # Both sessions use the SAME layer-local expert IDs, but different layers.
        self.local_ids = torch.tensor([[1, 0], [3, 1]], dtype=torch.int32)
        self.layer_ids = (1, 0, 1)
        shape = (self.sessions, self.batches, self.bs, self.k, self.h)
        self.tokens_cpu = (torch.arange(math.prod(shape)) % 97).half().reshape(shape)
        self.tokens = self.tokens_cpu.to(device)
        self.descriptors_cpu = torch.zeros(
            (self.sessions, self.batches, 2 + self.bs * self.k), dtype=torch.int32
        )
        for session, layer in enumerate(self.layer_ids):
            # FfnDataDesc = [flag, layer_id, expert_ids[BS*K]].
            self.descriptors_cpu[session, :, 1] = layer
            self.descriptors_cpu[session, :, 2:] = self.local_ids.flatten()
        # Batch 0 is empty. Only two of three sessions are ready in batch 1.
        # The async receiver must skip batch 0 and must not wait for session 2.
        self.descriptors_cpu[list(self.selected_sessions), self.selected_batch, 0] = 1
        self.descriptors = self.descriptors_cpu.to(device)
        context = bytearray(CONTEXT_BYTES)
        struct.pack_into(
            "<6I",
            context,
            0,
            self.sessions,
            self.batches,
            self.bs,
            self.k,
            self.experts // self.layers,
            self.h * self.tokens.element_size(),
        )
        struct.pack_into(
            "<4Q",
            context,
            FFN_AREA_OFFSET,
            self.descriptors.data_ptr(),
            self.descriptors.numel() * self.descriptors.element_size(),
            self.tokens.data_ptr(),
            self.tokens.numel() * self.tokens.element_size(),
        )
        # common.expert_num is per-layer; the operator attribute expert_num is
        # the total across layers. RECV does not use the NORM layer_ids_buf.
        self.context_cpu = torch.frombuffer(context, dtype=torch.int8).clone()
        self.context = self.context_cpu.to(device)

    def publish(self):
        # RECV consumes flag/IDs and advances polling_index. Restore IN PLACE
        # before each replay, preserving every address captured by the graph.
        # Copies and the consumer run on the same stream, so publication is ordered.
        self.descriptors.copy_(self.descriptors_cpu)
        self.context.copy_(self.context_cpu)

    def check(self, outputs):
        y, groups, sessions, batches, tokens, offsets, _, count = [
            t.cpu() for t in outputs
        ]
        count = int(count.item())
        expected = {}
        for session in self.selected_sessions:
            for token in range(self.bs):
                for choice in range(self.k):
                    expert = int(self.local_ids[token, choice])
                    expected[(session, self.selected_batch, token, choice)] = (
                        self.layer_ids[session] * (self.experts // self.layers) + expert
                    )
        indices = list(
            zip(
                sessions[:count].tolist(),
                batches[:count].tolist(),
                tokens[:count].tolist(),
                offsets[:count].tolist(),
            )
        )
        assert count == len(expected) and len(set(indices)) == count
        assert set(indices) == set(expected), indices
        keys = [expected[index] for index in indices]
        assert keys == sorted(keys), keys
        # Do not assume a particular order among tokens with equal expert IDs.
        torch.testing.assert_close(
            y[:count],
            torch.stack([self.tokens_cpu[i] for i in indices]),
            rtol=0,
            atol=0,
        )
        expected_groups = torch.zeros((self.experts, 2), dtype=torch.int64)
        for row, expert in enumerate(sorted(set(expected.values()))):
            expected_groups[row] = torch.tensor(
                [expert, list(expected.values()).count(expert)]
            )
        torch.testing.assert_close(groups, expected_groups, rtol=0, atol=0)
        after = self.descriptors_cpu.clone()
        for session in self.selected_sessions:
            after[session, self.selected_batch, 0] = 0
            after[session, self.selected_batch, 2:] = MASK_ID
        torch.testing.assert_close(self.descriptors.cpu(), after, rtol=0, atol=0)
        context_bytes = self.context.cpu().numpy().tobytes()
        assert struct.unpack_from("<Q", context_bytes, POLLING_INDEX_OFFSET)[0] == 0
        print("actual_token_num:", count)
        print("sorted cross-layer expert IDs:", keys)
        print("group_list [expert_id, token_count]:", groups[groups[:, 1] > 0].tolist())
        print(
            "PASS: token reorder, group_list, descriptor consumption and unselected sessions"
        )


def main():
    torch.npu.set_device(0)
    # Explicit loading also works for selectively packaged vendor wheels.
    op = importlib.import_module(
        f"cann_ops_transformer.ops.ffn.ffn_worker_batching"
    ).ffn_worker_batching
    case = AsyncLayerExample("npu:0")

    def call(context):
        return op(
            context,
            case.experts,
            [case.sessions, case.bs, case.k, case.h],
            token_dtype=0,
            need_schedule=1,
            layer_num=case.layers,
            sync_flag=True,
        )

    with torch.inference_mode():
        # Publish fresh descriptors before the consuming call.
        case.publish()
        outputs = call(case.context)
        torch.npu.synchronize()
        case.check(outputs)


if __name__ == "__main__":
    main()
```

### TorchAir GE调用方式

```python
import importlib
import math
import os
import struct

import torch
import torch_npu  # noqa: F401: registers NPU and npugraph_ex

# ScheduleContext ABI: common/include/op_kernel/attention_ffn_schedule.h.
# The first 6 uint32 words describe dimensions and the byte stride of token data.
CONTEXT_BYTES = 1024
FFN_AREA_OFFSET = 384
POLLING_INDEX_OFFSET = FFN_AREA_OFFSET + 4 * 8
MASK_ID = torch.iinfo(torch.int32).max


class AsyncLayerExample:
    """Keep all device allocations alive while context contains their pointers."""

    def __init__(self, device):
        self.sessions, self.batches, self.bs, self.k, self.h = 3, 2, 2, 2, 256
        self.layers, self.experts = 2, 8
        self.selected_batch = 1
        self.selected_sessions = (0, 1)
        # Both sessions use the SAME layer-local expert IDs, but different layers.
        self.local_ids = torch.tensor([[1, 0], [3, 1]], dtype=torch.int32)
        self.layer_ids = (1, 0, 1)
        shape = (self.sessions, self.batches, self.bs, self.k, self.h)
        self.tokens_cpu = (torch.arange(math.prod(shape)) % 97).half().reshape(shape)
        self.tokens = self.tokens_cpu.to(device)
        self.descriptors_cpu = torch.zeros(
            (self.sessions, self.batches, 2 + self.bs * self.k), dtype=torch.int32
        )
        for session, layer in enumerate(self.layer_ids):
            # FfnDataDesc = [flag, layer_id, expert_ids[BS*K]].
            self.descriptors_cpu[session, :, 1] = layer
            self.descriptors_cpu[session, :, 2:] = self.local_ids.flatten()
        # Batch 0 is empty. Only two of three sessions are ready in batch 1.
        # The async receiver must skip batch 0 and must not wait for session 2.
        self.descriptors_cpu[list(self.selected_sessions), self.selected_batch, 0] = 1
        self.descriptors = self.descriptors_cpu.to(device)
        context = bytearray(CONTEXT_BYTES)
        struct.pack_into(
            "<6I",
            context,
            0,
            self.sessions,
            self.batches,
            self.bs,
            self.k,
            self.experts // self.layers,
            self.h * self.tokens.element_size(),
        )
        struct.pack_into(
            "<4Q",
            context,
            FFN_AREA_OFFSET,
            self.descriptors.data_ptr(),
            self.descriptors.numel() * self.descriptors.element_size(),
            self.tokens.data_ptr(),
            self.tokens.numel() * self.tokens.element_size(),
        )
        # common.expert_num is per-layer; the operator attribute expert_num is
        # the total across layers. RECV does not use the NORM layer_ids_buf.
        self.context_cpu = torch.frombuffer(context, dtype=torch.int8).clone()
        self.context = self.context_cpu.to(device)

    def publish(self):
        # RECV consumes flag/IDs and advances polling_index. Restore IN PLACE
        # before each replay, preserving every address captured by the graph.
        # Copies and the consumer run on the same stream, so publication is ordered.
        self.descriptors.copy_(self.descriptors_cpu)
        self.context.copy_(self.context_cpu)

    def check(self, outputs):
        y, groups, sessions, batches, tokens, offsets, _, count = [
            t.cpu() for t in outputs
        ]
        count = int(count.item())
        expected = {}
        for session in self.selected_sessions:
            for token in range(self.bs):
                for choice in range(self.k):
                    expert = int(self.local_ids[token, choice])
                    expected[(session, self.selected_batch, token, choice)] = (
                        self.layer_ids[session] * (self.experts // self.layers) + expert
                    )
        indices = list(
            zip(
                sessions[:count].tolist(),
                batches[:count].tolist(),
                tokens[:count].tolist(),
                offsets[:count].tolist(),
            )
        )
        assert count == len(expected) and len(set(indices)) == count
        assert set(indices) == set(expected), indices
        keys = [expected[index] for index in indices]
        assert keys == sorted(keys), keys
        # Do not assume a particular order among tokens with equal expert IDs.
        torch.testing.assert_close(
            y[:count],
            torch.stack([self.tokens_cpu[i] for i in indices]),
            rtol=0,
            atol=0,
        )
        expected_groups = torch.zeros((self.experts, 2), dtype=torch.int64)
        for row, expert in enumerate(sorted(set(expected.values()))):
            expected_groups[row] = torch.tensor(
                [expert, list(expected.values()).count(expert)]
            )
        torch.testing.assert_close(groups, expected_groups, rtol=0, atol=0)
        after = self.descriptors_cpu.clone()
        for session in self.selected_sessions:
            after[session, self.selected_batch, 0] = 0
            after[session, self.selected_batch, 2:] = MASK_ID
        torch.testing.assert_close(self.descriptors.cpu(), after, rtol=0, atol=0)
        context_bytes = self.context.cpu().numpy().tobytes()
        assert struct.unpack_from("<Q", context_bytes, POLLING_INDEX_OFFSET)[0] == 0
        print("actual_token_num:", count)
        print("sorted cross-layer expert IDs:", keys)
        print("group_list [expert_id, token_count]:", groups[groups[:, 1] > 0].tolist())
        print(
            "PASS: token reorder, group_list, descriptor consumption and unselected sessions"
        )


def main():
    torch.npu.set_device(0)
    # Explicit loading also works for selectively packaged vendor wheels.
    op = importlib.import_module(
        f"cann_ops_transformer.ops.ffn.ffn_worker_batching"
    ).ffn_worker_batching
    case = AsyncLayerExample("npu:0")

    def call(context):
        return op(
            context,
            case.experts,
            [case.sessions, case.bs, case.k, case.h],
            token_dtype=0,
            need_schedule=1,
            layer_num=case.layers,
            sync_flag=True,
        )

    import torchair
    from torchair.configs.compiler_config import CompilerConfig

    # Import the converter explicitly: optional package discovery must not
    # silently leave this custom operator without GE lowering.
    importlib.import_module(
        f"cann_ops_transformer.ops.ffn.ffn_worker_batching.graph_convert_ffn_worker_batching"
    )
    backend = torchair.get_npu_backend(compiler_config=CompilerConfig())
    call = torch.compile(call, backend=backend, fullgraph=True, dynamic=False)
    with torch.inference_mode():
        # The second graph call exercises replay. Never replay consumed input
        # without publishing descriptors again: the receiver would keep polling.
        for iteration in range(2):
            case.publish()
            outputs = call(case.context)
            torch.npu.synchronize()
            print(f"iteration {iteration + 1}")
            case.check(outputs)


if __name__ == "__main__":
    main()
```
