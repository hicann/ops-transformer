# GroupMemcpy

将多个分散的连续 Tensor 成对按字节拷贝到预分配目标，可用于 K/V cache 切片搬运。实现由原 GroupMemcpy 直调算子迁移而来，调用链为：

~~~text
Python group_memcpy → Torch PrivateUse1 → aclnnGroupMemcpy 两段式接口
                   → GroupMemcpy OpDef / Tiling → Ascend C Kernel
~~~

对每个 `i ∈ [0, N)`，执行 `dst[i].bytes[0:W] = src[i].bytes[0:W]`。所有组的 `W` 相同；保留数据的原始位模式，不做类型转换，不分配替代输出。

## 产品与验证状态

<!-- npu="910b" id1 -->
- <term>Atlas A2系列产品</term>：已在 Ascend 910B3 / CANN 9.1 上完成编译与运行验证。
<!-- end id1 -->


## 参数

| 参数名 | 输入/输出 | 描述 | 数据类型 | 数据格式 |
| --- | --- | --- | --- | --- |
| src | 输入 | 非空动态 TensorList，N 个源 Tensor | INT8、UINT8、INT16、INT32、INT64、FLOAT16、BFLOAT16、FLOAT、BOOL | ND、连续 |
| dst | 输出 | N 个预分配目标 Tensor，由调用方持有 | 与 src 全部元素相同 | ND、连续 |

没有公开标量属性。`N` 与 `W` 从列表及 Tensor 元数据推导，代替原始 `gbH`、`gbW`。

- 列表长度相同且不为零，最长为 `INT32_MAX`；每对 shape 相同，所有 Tensor 的 dtype、元素数和设备一致。
- 每组完整拷贝，支持连续切片的 storage offset、标量和零元素 Tensor；不支持非连续或 NPU 私有分块格式。
- 所有目标之间、目标与任何源之间必须没有内存重叠；源可以重复或互相重叠。没有跨组读写先后关系，不具备 `memmove` 语义。
- 非空列表内全部为零元素 Tensor 时不发射 Kernel。空列表拒绝。
- 按当前 NPU stream 异步提交，不提供 autograd。C aclnn 调用方负责设备、地址有效性、分配容量和不重叠条件；Torch 额外检查设备和地址重叠。

## 目录

~~~text
group_memcpy/
├── CMakeLists.txt
├── group_memcpy.json                 # msopgen 接口描述；不是编译产物
├── torch_extension/                 # 独立 Torch 扩展（OpBuilder + aclnn）
│   ├── __init__.py
│   ├── group_memcpy.py               # schema、Meta、PrivateUse1、Python 包装
│   └── csrc/group_memcpy.cpp         # PyBind、aclnn 调用与 stream 生命周期
├── op_host/
│   ├── CMakeLists.txt
│   ├── group_memcpy_def.cpp
│   ├── group_memcpy_infershape.cpp
│   ├── group_memcpy_tiling.cpp
│   ├── group_memcpy_tiling.h
│   ├── group_memcpy_tiling_policy.h
│   ├── config/ascend910b/            # 9 组同 dtype 编译配置
│   └── op_api/aclnn_group_memcpy.{h,cpp}
├── op_kernel/group_memcpy.{h,cpp}
├── docs/                            # aclnn 接口、迁移说明、验证记录
└── tests/                           # aclnn C++、Torch pytest、CPU 检查
~~~

采用 `bias_sigmoid` 的“算子目录 + README + tests”组织方式；其本身使用 Kernel 直调，故注册框架参考仓库内 `generic_block_sparse_attention`、`chunk_local_cumsum` 以及动态 TensorList 算子。

## 构建与调用

在配套 CANN 的 Linux/NPU 环境进入仓库根目录，按实际安装位置载入 CANN 环境后运行：

~~~bash
source /usr/local/Ascend/cann/set_env.sh
bash build.sh --pkg --experimental --soc=ascend910b --ops=group_memcpy --vendor_name=group_memcpy
find build_out -type f -name '*.run'
~~~


~~~bash
# 在仓库根目录执行，生成包含 experimental 算子的统一扩展包。
bash build.sh --torch_extension --experimental
python3 -m pip install --force-reinstall --no-deps build_out/cann_ops_transformer-*.whl
~~~

~~~python
import torch
import torch_npu
from cann_ops_transformer.ops.attention.group_memcpy import group_memcpy

src = [torch.randn(128, dtype=torch.bfloat16, device="npu") for _ in range(3)]
dst = [torch.empty_like(x) for x in src]
group_memcpy(src, dst)
torch.npu.synchronize()
assert all(torch.equal(x.view(torch.uint8).cpu(), y.view(torch.uint8).cpu())
           for x, y in zip(src, dst))
~~~

直接 Torch 入口为 `torch.ops.cann_ops_transformer.group_memcpy.default(src, dst)`，返回 `None`，原地修改 `dst`。

## 说明文档

- [Torch 接口说明](docs/torchapi_group_memcpy.md)
- [aclnnGroupMemcpy 接口说明](docs/aclnnGroupMemcpy.md)
- [原直调调用方迁移及源码依据](docs/migration.md)
- [测试矩阵、运行命令与原始二进制回放](tests/README.md)
- [本次验证记录](docs/validation.md)
