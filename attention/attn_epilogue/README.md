# AttnEpilogue

## 产品支持情况

| 产品 | 是否支持 |
| ---- | :------: |
| <term>Ascend 950PR&950DT系列产品</term> | √ |
| <term>Atlas A3系列产品</term> | × |
| <term>Atlas A2系列产品</term> | × |
| <term>Atlas 200I/500 A2推理产品</term> | × |
| <term>Atlas推理系列产品</term> | × |
| <term>Atlas训练系列产品</term> | × |

## 功能说明

- 算子功能：Ascend 950PR/950DT（ARCH 3510）上的本地 attention 输出投影，将 inverse RoPE、两次 MXFP8 激活量化和两级 Matmul 融合为一个设备 kernel。实现位于 [attn_epilogue.py](op_dsl/attn_epilogue.py)，不包含通信或 AllReduce。

- 计算公式：

    记 $A$ 为完成 inverse RoPE 的 BF16 激活，$Q_{32}(\cdot)$ 为沿 K 轴每 32 个元素一个 E8M0 scale 的 MXFP8 量化，$W_{oA,g}$、$W_{oB}$ 为两级权重，则：

    $$
    A = \operatorname{BF16}\left(\operatorname{inverse\_rope}(x)\right)
    $$

    $$
    Y_g = \operatorname{BF16}\left(\operatorname{MXMatmul}\left(Q_{32}(A_g), W_{oA,g}^{T}\right)\right)
    $$

    $$
    out = \operatorname{BF16}\left(\operatorname{MXMatmul}\left(Q_{32}\left(\operatorname{concat}_{g}(Y_g)\right), W_{oB}^{T}\right)\right)
    $$

    inverse RoPE 仅处理每个 head 的最后 64 维，前 448 维保持原值；尾部按相邻元素配对，在 FP32 中计算 `even' = even * cos + odd * sin`、`odd' = odd * cos - even * sin`，再舍入为 BF16。sin/cos 表按 `[s0,s0,s1,s1,...]` 重复。NG 表示本算子处理的本地分组数，算子只计算这些分组的输出投影，不执行跨卡通信或结果聚合。

    支持任意 $T \ge 1$，受设备内存限制。普通执行返回具有独立存储的 BF16 `[T,5120]` 输出。

- 实现说明：

    MM1 在 FP32 中累加，并在 QuantY 前先舍入为 BF16。FP32 tile 通过 Fixpipe 直接交付配对 AIV，省去 BF16 中间 Y 的 GM 缓冲。内部只保留 Aq、A-scale、Yq、Y-scale 四个字节工作区。

    实现使用 DSL 0.8 的显式 layout、RingBuffer 和 copy_options，并显式开启 MX Matmul。构造接口的 layout/shape 差异在导入时识别一次，不在 forward 中反复检查。Fixpipe 按完整 tile 的两个半帧交付 AIV，每个 AIV 独立裁剪有效行；空分片仍消费接收槽，完成所有权交接，但不写入 GM。尾块不依赖按 T 或平台枚举的配置。

    M/N/K tiling 根据 T、矩阵维度、当前 stream 核配额及 ARCH 3510 的 L0/L1/UB 容量推导，没有按 T 或设备名称列举配置。MM1 的 N 分块对齐完整 MX scale 字节对；满足生产者覆盖条件时，QuantA 和 MM1 按 K slab 流水。Cube 缓冲复用和 QuantY → MM2 的数据/scale 就绪依赖保留显式同步。

    编译缓存命中后直接复用 handle，不再重复计算 tiling。普通执行的工作区按设备、stream、T、NG 缓存，图捕获使用图内存池；跨 stream 的依赖由调用方建立。首次调用包含编译开销，应先预热。当前提供前向实现，不提供反向。

## 参数说明

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
| :--- | :--- | :--- | :--- | :--- |
| x | 输入 | 待投影的 attention 输出激活，shape 为 `[T,NG*8,512]`。inverse RoPE 仅作用于每个 head 的后 64 维。 | BFLOAT16 | ND |
| woa | 输入 | 第一级 WoA 权重，ND 逻辑 shape 为 `[NG,1024,4096]`，NZ 逻辑 shape 为 `[NG,4096,1024]`。 | FLOAT8_E4M3FN | ND、私有NZ |
| wob | 输入 | 第二级 WoB 权重，ND 逻辑 shape 为 `[5120,NG*1024]`，NZ 逻辑 shape 为 `[NG*1024,5120]`。 | FLOAT8_E4M3FN | ND、私有NZ |
| descale_woa | 输入 | 第一级权重 MX scale，始终为 ND；ND 权重时为 `[NG,1024,64,2]`，NZ 权重时为 `[NG,64,1024,2]`。 | FLOAT8_E8M0FNU | ND |
| descale_wob | 输入 | 第二级权重 MX scale，始终为 ND；ND 权重时为 `[5120,NG*16,2]`，NZ 权重时为 `[NG*16,5120,2]`。 | FLOAT8_E8M0FNU | ND |
| rope_sin | 输入 | inverse RoPE 的 sin 表，shape 为 `[T,64]`，按 `[s0,s0,s1,s1,...]` 重复。 | FLOAT32 | ND |
| rope_cos | 输入 | inverse RoPE 的 cos 表，shape 为 `[T,64]`，按 `[c0,c0,c1,c1,...]` 重复。 | FLOAT32 | ND |
| out | 输出 | 输出投影结果，shape 为 `[T,5120]`。 | BFLOAT16 | ND |

每组含 8 个 head，固定 `F=4096`、`O_LORA=1024`、`DIM=5120`。三种分组场景如下：

| 场景 | 输入 x | 第二级 K |
| :--- | :--- | ---: |
| NG=8 | `[T,64,512]` | 8192 |
| NG=2 | `[T,16,512]` | 2048 |
| NG=1 | `[T,8,512]` | 1024 |

## 约束说明

- `T >= 1`，`NG` 仅支持 1、2、8，`x` 的 shape 必须为 `[T,NG*8,512]`。
- 所有输入须连续、位于同一 NPU，并符合参数表中的 shape、dtype；两级权重可以分别选择 ND/NZ，scale 始终为 ND。
- NZ 指通过 `torch_npu.npu_format_cast(weight_kn,29)` 生成的私有张量，保留逻辑 `(K,N)` 轴；不是显式增加打包维度的普通 ND 张量。
- 入口仅检查激活的有效维度，用权重逻辑轴识别布局，不逐项扫描张量的 dtype、device、stride 或私有格式。调用方应在加载权重时保证这些契约，不能依赖入口发现全部错误输入。
- 若从 uint8 文件加载 scale，使用 `.view(torch.float8_e8m0fnu)` 恢复位模式，不能做数值转换。
- 当前提供前向实现，不提供反向。

## 调用说明

| 调用方式 | 样例代码 | 说明 |
| :--- | :--- | :--- |
| PyTorch API | - | 通过[torchapi_attn_epilogue.md](docs/torchapi_attn_epilogue.md)接口调用AttnEpilogue算子，支持单算子与`torch.compile`（`npugraph_ex`）图模式，含完整接口原型、参数、返回值与调用示例。 |
| DSL 直接调用 | [attn_epilogue.py](op_dsl/attn_epilogue.py) | 将 `attention/attn_epilogue/op_dsl` 加入 `PYTHONPATH` 后，通过 `from attn_epilogue import attn_epilogue` 调用 DSL 实现。 |

### ND 权重预转换为私有 NZ

转换只在初始化时执行，不能放进被测 forward。按字节转换可避免部分 CANN 版本对 FP8/E8M0 设备转置的限制，且不会改变量化值。

```python
import torch
import torch_npu


def weight_to_nz(weight):
    """Preserve quantized bytes while converting logical (N,K) to private NZ (K,N)."""
    kn = weight.view(torch.uint8).cpu().transpose(-2, -1).contiguous()
    return torch_npu.npu_format_cast(kn.npu().view(torch.float8_e4m3fn), 29)


wa_nz = weight_to_nz(wa_nd)
wb_nz = weight_to_nz(wb_nd)
sa_kn = sa_nd.view(torch.uint8).cpu().permute(0, 2, 1, 3).contiguous().npu()
sb_kn = sb_nd.view(torch.uint8).cpu().permute(1, 0, 2).contiguous().npu()
out = attn_epilogue(x, wa_nz, wb_nz,
    sa_kn.view(torch.float8_e8m0fnu), sb_kn.view(torch.float8_e8m0fnu), sin, cos)
```

### 图模式调用

支持 `torch.npu.NPUGraph`，也支持 `torch.compile` 的 `npugraph_ex` 后端。模型中建议将量化权重及 scale 注册为固定的 buffer，保留其私有 NZ 存储：

```python
class Projection(torch.nn.Module):
    """Keep quantized weights as fixed graph parameters."""

    def __init__(self, woa, wob, sa, sb):
        """Retain the initialized NPU tensors without converting their formats."""
        super().__init__()
        for name, value in zip(("woa", "wob", "sa", "sb"), (woa, wob, sa, sb)):
            self.register_buffer(name, value)

    def forward(self, x, sin, cos):
        """Project the current activations with the fixed quantized weights."""
        return attn_epilogue(x, self.woa, self.wob, self.sa, self.sb, sin, cos)


model = torch.compile(
    Projection(woa, wob, descale_woa, descale_wob).eval(),
    backend="npugraph_ex", fullgraph=True, dynamic=False,
)
out = model(x, rope_sin, rope_cos)
```

同一模型跨 T 使用时可设 `dynamic=True`；已验证 T=3、7、16、17 之间的切换。图回放的输出缓冲遵循后端的静态复用语义；跨回放保留结果时，先用 `out.clone()` 保存。

若直接编译七参数函数，并把私有 NZ 权重作为调用实参，使用后端的 `clone_input=False`：

```python
compiled_op = torch.compile(
    attn_epilogue, backend="npugraph_ex", fullgraph=True, dynamic=False,
    options={"clone_input": False},
)
out = compiled_op(x, woa, wob, descale_woa, descale_wob, rope_sin, rope_cos)
```

此时使用固定输入缓冲区更新数据，保持 NZ 权重的存储地址不变；替换权重存储后重新建图。当前 torch_npu 后端默认克隆用户输入时可能把私有 NZ 转为 ND，随后尝试不支持的 FP8 私有格式复制。该过程位于算子外部；上述 buffer 参数方式和禁用输入克隆方式均已验证。ND 实参可使用后端默认配置。算子不修改全局格式开关，也不在 forward 中重新打包权重。

### 测试

仓内测试为 [test_attn_epilogue_dsl.py](tests/pytest/test_attn_epilogue_dsl.py)：6 个 NPU 精度用例覆盖三种 NG 和 ND/NZ、尾块、NPUGraph 和一个 npugraph_ex 回归，另有 2 个 Meta 用例。不包含性能采集、大 T 扫描、外部 `cann_ops_*` 依赖或全量编译任务，不修改公共 CI 配置。

```bash
# 使用仓库公共 marker 选择 CPU / NPU 阶段。
python3 -m pytest -q attention/attn_epilogue/tests/pytest -m 'not npu'
ASCEND_RT_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 \
  python3 -m pytest -q attention/attn_epilogue/tests/pytest -m npu
```

要求匹配的 CANN、torch_npu 及 DSL 0.8.0；NZ 路径使用 `TensorSpec(storage_format="nz")`。本次 0.8.0 适配在 950DT 上验证，使用 CANN 9.3.0、torch 2.12.0+cpu、torch_npu 2.12.0.post2。950PR 的历史验证记录使用 DSL 0.7.1。完整精度、压力、性能和原始 profiling 由仓外开发验证脚本维护，线上 CI 不收集这些脚本。

## 参考资源

- [PyTorch API 文档（中文）](docs/torchapi_attn_epilogue.md)
- [PyTorch API Document (English)](docs/torchapi_attn_epilogue_en.md)
- [算子单元测试](tests/pytest/test_attn_epilogue_dsl.py)

## 性能对比（DSL 0.8.0）

2026-11-02 在 Ascend950DT 上完成 16 个代表点复测，使用包含片上容量约束修正的实现、CANN 9.3.0、DSL 0.8.0、torch 2.12.0+cpu、torch_npu 2.12.0.post2，设备为 32 AIC / 64 AIV、128 MiB L2。本次为 16 个代表点复测，未重新执行 576 点全量扫描。

表中耗时为设备 kernel 时长中位数，单位为微秒；原生链路取五个 kernel 的时长之和。

| 场景 | 权重布局 | T | 原生链路（µs） | 融合算子（µs） | 加速比 |
| :--- | :--- | ---: | ---: | ---: | ---: |
| NG=8 | ND | 1 | 48.178 | 29.924 | 1.610× |
| NG=8 | ND | 72 | 56.465 | 41.449 | 1.362× |
| NG=8 | ND | 128 | 66.463 | 49.918 | 1.331× |
| NG=8 | ND | 256 | 87.355 | 73.464 | 1.189× |
| NG=8 | NZ | 1 | 45.078 | 28.239 | 1.596× |
| NG=8 | NZ | 72 | 55.338 | 40.445 | 1.368× |
| NG=8 | NZ | 128 | 63.409 | 49.397 | 1.284× |
| NG=8 | NZ | 256 | 96.873 | 73.363 | 1.320× |
| NG=2 | ND | 1 | 35.511 | 19.578 | 1.814× |
| NG=2 | ND | 16 | 37.096 | 21.400 | 1.733× |
| NG=2 | NZ | 1 | 29.942 | 18.431 | 1.625× |
| NG=2 | NZ | 16 | 31.898 | 21.635 | 1.474× |
| NG=1 | ND | 1 | 30.793 | 14.036 | 2.194× |
| NG=1 | ND | 16 | 34.368 | 17.791 | 1.932× |
| NG=1 | NZ | 1 | 26.480 | 13.266 | 1.996× |
| NG=1 | NZ | 16 | 30.030 | 16.925 | 1.774× |

原生五 kernel 链路与融合单 kernel 使用相同输入、相同 ND/NZ 权重，交错进行图回放；每点每侧 48 个样本，调用前在同流读取 256 MiB 缓冲区进行软件 L2 驱逐。计时排除驱逐、编译、输入恢复和主机调度；加速比为原生中位数 / 融合中位数。该口径不表示端到端请求时延。16 点全部获得收益，加速比为 1.189–2.194×，普通执行及每点 16 个图捕获输出均与原生参考逐位一致。
