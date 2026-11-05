# group_memcpy Torch 接口

## 导入与调用

```python
from cann_ops_transformer.ops.attention.group_memcpy import group_memcpy

group_memcpy(src, dst) -> None
```

接口源码位于算子本地 `torch_extension/`，由仓库统一打包器自动收集。Python 包装通过 `torch.ops.cann_ops_transformer.group_memcpy.default(src, dst)` 分发到 NPU 实现；`OpBuilder` 在第一次 NPU 调用时编译 `csrc/group_memcpy.cpp`，再调用已安装算子包的 `aclnnGroupMemcpyGetWorkspaceSize` / `aclnnGroupMemcpy`。

## 参数与行为

| 参数 | 类型 | 约束 |
| --- | --- | --- |
| src | Sequence[torch.Tensor] | 非空源列表，连续 ND，位于同一 NPU |
| dst | Sequence[torch.Tensor] | 预分配目标列表，长度与 src 相同，逐对 shape 一致 |

所有 Tensor 的 dtype 和元素数一致。支持 int8、uint8、int16、int32、int64、float16、bfloat16、float32、bool。每个目标不得与任何源或其他目标的地址范围重叠；允许重复源。支持连续 offset 切片、标量和零元素 Tensor。返回 `None`，按当前 NPU stream 异步原地更新 dst，不分配替代输出，按原始字节复制。

不支持非连续 Tensor、NPU 私有分块格式或 requires_grad。Python 包装在空列表或列表长度不等时抛出 ValueError；后端拒绝不满足 dtype、shape、连续性、设备及地址重叠约束的输入。NPU/Meta 实现会更新普通目标 Tensor 的 version，以保持原地修改检测；不提供反向传播。

Meta 实现仅校验元数据，不触发 C++ 编译或 NPU 执行。Meta 注册不代表已经支持 torchair 图模式，本次未提供 GE converter。

## 示例

```python
import torch
import torch_npu
from cann_ops_transformer.ops.attention.group_memcpy import group_memcpy

src = [torch.arange(64, dtype=torch.int32, device="npu") for _ in range(3)]
dst = [torch.empty_like(x) for x in src]
group_memcpy(src, dst)
torch.npu.synchronize()
assert all(torch.equal(x.cpu(), y.cpu()) for x, y in zip(src, dst))
```

先构建并安装 aclnn 算子包，再安装统一 Torch 扩展。完整命令见 [构建说明](../README.md) 和 [测试说明](../tests/README.md)。
