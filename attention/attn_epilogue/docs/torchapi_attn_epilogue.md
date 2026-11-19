# attn_epilogue

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT系列产品</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3系列产品</term>：不支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2系列产品</term>：不支持
<!-- end id3 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2推理产品</term>：不支持
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas推理系列产品</term>：不支持
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas训练系列产品</term>：不支持
<!-- end id6 -->

## 功能说明

- **接口功能**：`cann_ops_transformer.attn_epilogue` 将 attention 输出投影中的 inverse RoPE、两次 MXFP8 激活量化和两级 Matmul 融合为一次 Torch 自定义算子调用。该接口直接调用 DSL 实现，不包含通信或 AllReduce。完整的分组、布局和调度说明见 [AttnEpilogue](../README.md)。

- **计算语义**：

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

    inverse RoPE 仅处理每个 head 的最后 64 维，前 448 维保持原值；尾部按相邻元素配对，在 FP32 中计算 `even' = even * cos + odd * sin`、`odd' = odd * cos - even * sin`，再舍入为 BF16。sin/cos 表按 `[s0,s0,s1,s1,...]` 重复。

- **分组说明**：NG 表示本算子处理的本地分组数，每组含 8 个 head，固定 `F=4096`、`O_LORA=1024`、`DIM=5120`。

    | 场景 | 输入 x | 第二级 K |
    | :--- | :--- | ---: |
    | NG=8 | `[T,64,512]` | 8192 |
    | NG=2 | `[T,16,512]` | 2048 |
    | NG=1 | `[T,8,512]` | 1024 |

## 函数原型

```python
import cann_ops_transformer

cann_ops_transformer.attn_epilogue(
    x: Tensor,
    woa: Tensor,
    wob: Tensor,
    descale_woa: Tensor,
    descale_wob: Tensor,
    rope_sin: Tensor,
    rope_cos: Tensor,
) -> Tensor
```

也可以使用 `from cann_ops_transformer.ops import attn_epilogue` 导入该接口。接口注册名为 `torch.ops.cann_ops_transformer.attn_epilogue`，支持单算子模式和 `torch.compile` 图模式调用。

## 参数说明

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度(shape) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| x | Tensor | 必选 | 待投影的 attention 输出激活。inverse RoPE 仅作用于每个 head 的后 64 维。数据格式为 ND。 | bfloat16 | (T, NG*8, 512) |
| woa | Tensor | 必选 | 第一级 WoA 权重。ND 权重对应 shape 为 (NG, 1024, 4096)，私有 NZ 权重对应逻辑 shape 为 (NG, 4096, 1024)。 | float8_e4m3fn | 见描述 |
| wob | Tensor | 必选 | 第二级 WoB 权重。ND 权重对应 shape 为 (5120, NG*1024)，私有 NZ 权重对应逻辑 shape 为 (NG*1024, 5120)。 | float8_e4m3fn | 见描述 |
| descale_woa | Tensor | 必选 | 第一级权重 MX scale，始终为 ND。ND 权重时为 (NG, 1024, 64, 2)，NZ 权重时为 (NG, 64, 1024, 2)。 | float8_e8m0fnu | 见描述 |
| descale_wob | Tensor | 必选 | 第二级权重 MX scale，始终为 ND。ND 权重时为 (5120, NG*16, 2)，NZ 权重时为 (NG*16, 5120, 2)。 | float8_e8m0fnu | 见描述 |
| rope_sin | Tensor | 必选 | inverse RoPE 的 sin 表，按 `[s0,s0,s1,s1,...]` 重复。数据格式为 ND。 | float32 | (T, 64) |
| rope_cos | Tensor | 必选 | inverse RoPE 的 cos 表，按 `[c0,c0,c1,c1,...]` 重复。数据格式为 ND。 | float32 | (T, 64) |

## 返回值说明

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度(shape) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| out | Tensor | 必选 | 输出投影结果。普通执行返回具有独立存储的 Tensor。数据格式为 ND。 | bfloat16 | (T, 5120) |

## 约束说明

- 该接口支持训练、推理场景下使用。
- 该接口支持单算子模式和 torch.compile（npugraph_ex）图模式调用。
- `T >= 1`，`NG` 仅支持 1、2、8。
- 所有输入须连续、位于同一 NPU，并符合参数表中的 shape、dtype；两级权重可以分别选择 ND/NZ，scale 始终为 ND。
- NZ 指通过 `torch_npu.npu_format_cast(weight_kn,29)` 生成的私有张量，保留逻辑 `(K,N)` 轴；不是显式增加打包维度的普通 ND 张量。
- 接口仅检查激活的有效维度，用权重逻辑轴识别布局，不逐项扫描张量的 dtype、device、stride 或私有格式。调用方应在加载权重时保证这些契约，不能依赖接口发现全部错误输入。
- 若从 uint8 文件加载 scale，使用 `.view(torch.float8_e8m0fnu)` 恢复位模式，不能做数值转换。
- 图回放的输出缓冲遵循后端的静态复用语义；跨回放保留结果时，先用 `out.clone()` 保存。
- 当前提供前向实现，不提供反向。

## 确定性/Batch一致性

- 确定性说明：默认支持确定性计算。
- Batch一致性说明：默认支持Batch一致性。

## 调用示例

- 单算子模式调用

    ```python
    import torch
    import torch_npu
    from cann_ops_transformer.ops import attn_epilogue

    torch.npu.set_device(0)
    device = torch.device("npu:0")
    t, ng = 72, 8

    # 构造激活与两级量化权重
    x = (torch.randn(t, ng * 8, 512, device=device) * 0.3).bfloat16()
    wa = (torch.randn(ng, 1024, 4096, device=device) * 0.3).bfloat16()
    wb = (torch.randn(5120, ng * 1024, device=device) * 0.2).bfloat16()

    woa, descale_woa = torch_npu.npu_dynamic_mx_quant(
        wa.reshape(-1, 4096), dst_type=292
    )
    wob, descale_wob = torch_npu.npu_dynamic_mx_quant(wb, dst_type=292)
    e8 = torch.float8_e8m0fnu
    woa = woa.reshape(ng, 1024, 4096)
    descale_woa = descale_woa.view(e8).reshape(ng, 1024, 64, 2)
    descale_wob = descale_wob.view(e8)

    # 生成成对重复的 inverse RoPE sin/cos 表
    angles = torch.outer(
        torch.arange(t).float() + 3,
        10000.0 ** (-torch.arange(0, 64, 2).float() / 64),
    )
    rope_sin = angles.sin().repeat_interleave(2, -1).to(device)
    rope_cos = angles.cos().repeat_interleave(2, -1).to(device)

    out = attn_epilogue(x, woa, wob, descale_woa, descale_wob, rope_sin, rope_cos)
    print(out.shape, out.dtype)
    ```

- 图模式调用

    ```python
    import torch
    import torch_npu
    from cann_ops_transformer.ops import attn_epilogue

    class AttnEpilogueModel(torch.nn.Module):
        def forward(self, x, woa, wob, descale_woa, descale_wob, rope_sin, rope_cos):
            return attn_epilogue(
                x, woa, wob, descale_woa, descale_wob, rope_sin, rope_cos
            )

    # 私有 NZ 权重作为实参时关闭输入克隆，保持其存储地址不变
    model = torch.compile(
        AttnEpilogueModel().eval(),
        backend="npugraph_ex", fullgraph=True, dynamic=False,
        options={"clone_input": False},
    )
    out = model(x, woa, wob, descale_woa, descale_wob, rope_sin, rope_cos)
    print(out.shape, out.dtype)
    ```

    使用私有 NZ 权重作为调用实参时，需设置 `clone_input=False`，保持 NZ 权重的存储地址不变；替换权重存储后重新建图。
