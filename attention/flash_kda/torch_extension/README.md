# FlashKDA PyTorch接口

## 接口功能

`flash_kda_metadata`生成调度信息；`flash_kda`返回前向结果、最终状态和chunk入口状态；`flash_kda_fwd`还返回训练反向所需的缓存。输入含义、类型和约束参见[FlashKDA](../README.md)。

## 接口原型

```python
def flash_kda(
    q, k, v, g, beta, metadata,
    initial_state=None, A_log=None, dt_bias=None,
    cu_seqlens=None, cu_seqlens_cpu=None, seqused=None, *,
    scale=128**(-0.5), output_final_state=False,
    return_intermediate_states=False, use_qk_l2norm_in_kernel=True,
    use_gate_in_kernel=True, use_beta_sigmoid_in_kernel=True,
    allow_neg_eigval=False, disable_recompute=False,
    lower_bound=None, layout="TND",
):
    ...
```

`flash_kda_fwd`具有相同的输入参数及默认值。

```python
def flash_kda_metadata(
    cu_seqlens=None, seqused=None, *,
    batch_size, nheads_qk, nheads_v, seqlens,
    headdim_qk=128, headdim_v=128, enable_grad=False, layout="TND",
):
    ...
```

metadata接口的`batch_size`、`nheads_qk`、`nheads_v`分别等于`B`、`Nqk`、`Nv`；`seqlens`为定长存储长度`S`或紧凑存储总长度`T`，不能用有效长度代替。维度参数均为正整数且不得超过INT32范围，两个head维度固定为128。`cu_seqlens`、`seqused`和`layout`与前向保持一致；`enable_grad`应设为`not torch.is_inference_mode_enabled()`。接口在当前NPU上生成一维INT32张量，长度由实现确定，调用方不得手工拼接或修改内容。

## 返回值

`flash_kda`固定返回`(out, final_state, chunk_states)`。`flash_kda_fwd`固定返回以下16项，不需要的项为`None`，不会从元组中删除。

| 索引 | 名称 | 数据类型 | 返回条件 |
| --- | --- | --- | --- |
| 0 | out | BFLOAT16 | 始终返回。 |
| 1 | final_state | FLOAT32 | `output_final_state=True`。 |
| 2 | chunk_states | BFLOAT16 | 推理时`return_intermediate_states=True`，或训练时`disable_recompute=True`。 |
| 3 | Aqk | BFLOAT16 | 训练模式，块内因果QK矩阵，已包含scale。 |
| 4 | Akk | BFLOAT16 | 训练模式，块内逆矩阵。 |
| 5 | w | BFLOAT16 | 训练且`disable_recompute=True`。 |
| 6 | u | BFLOAT16 | 训练且`disable_recompute=True`。 |
| 7 | qg | BFLOAT16 | 训练且`disable_recompute=True`。 |
| 8 | kg | BFLOAT16 | 训练且`disable_recompute=True`。 |
| 9 | v_new | BFLOAT16 | 训练且`disable_recompute=True`。 |
| 10 | g_cumsum | FLOAT32 | 训练且`disable_recompute=True`，chunk内累积log gate。 |
| 11 | q_norm | BFLOAT16 | 训练且`use_qk_l2norm_in_kernel=True`。 |
| 12 | q_rstd | FLOAT32 | 训练且`use_qk_l2norm_in_kernel=True`，Q归一化的逆范数。 |
| 13 | k_norm | BFLOAT16 | 训练且`use_qk_l2norm_in_kernel=True`。 |
| 14 | k_rstd | FLOAT32 | 训练且`use_qk_l2norm_in_kernel=True`，K归一化的逆范数。 |
| 15 | beta_activated | FLOAT32 | 训练且`use_beta_sigmoid_in_kernel=True`。 |

输出的形状约定如下，定长布局的`C=ceil(S/64)`。TND的`Ctotal`为实现分配的chunk状态容量，包含分块对齐所需空间；只消费有效序列对应的状态。

| 输出 | BNSD或BSND输入 | TND输入 |
| --- | --- | --- |
| out | `(B,S,Nv,128)` | `(T,Nv,128)` |
| final_state | `(B,Nv,128,128)` | `(B,Nv,128,128)` |
| chunk_states | `(B,C,Nv,128,128)` | `(Ctotal,Nv,128,128)` |
| Aqk、Akk | `(B,Nv,S,64)` | `(Nv,T,64)` |
| w、u、qg、kg、v_new、g_cumsum | `(B,Nv,S,128)` | `(Nv,T,128)` |
| q_norm、k_norm | 与q、k相同 | 与q、k相同 |
| q_rstd、k_rstd | `(B,Nqk,S)` | `(Nqk,T)` |
| beta_activated | `(B,Nv,S)` | `(Nv,T)` |

训练模式不会由输入的`requires_grad`或`torch.no_grad()`自动切换为推理模式。训练调用需要反向缓存时使用`flash_kda_fwd`，并与[FlashKDAGrad](../../flash_kda_grad/README.md)接口配套；三项返回值的`flash_kda`不提供这些额外缓存。

## 调用示例

在Linux NPU环境中安装匹配的CANN、PyTorch、torch_npu和CANNBot DSL，再按[扩展构建说明](../../../torch_extension/README.md)构建并安装包。以下示例使用专用包名`cann_ops_transformer_flash_kda`，对应构建参数`--ops=flash_kda --vendor_name=flash_kda`。包应已包含匹配设备的Native编译产物。

```python
import torch
import torch_npu
import cann_ops_transformer_flash_kda as ops

torch.npu.set_device(0)
batch, length, n_qk, n_v, dim = 1, 65, 2, 4, 128

# BSND输入；g和beta分别按(B,S,Nv,D)和(B,S,Nv)存储。
q = torch.randn(batch, length, n_qk, dim, device="npu", dtype=torch.bfloat16)
k = torch.randn_like(q)
v = torch.randn(batch, length, n_v, dim, device="npu", dtype=torch.bfloat16)
g = torch.randn(batch, length, n_v, dim, device="npu", dtype=torch.float32)
beta = torch.randn(batch, length, n_v, device="npu", dtype=torch.float32)
A_log = torch.zeros(n_v, device="npu", dtype=torch.float32)

# metadata和前向使用相同的存储规格及执行模式。
with torch.inference_mode():
    metadata = ops.flash_kda_metadata(
        batch_size=batch, nheads_qk=n_qk, nheads_v=n_v,
        seqlens=length, enable_grad=False, layout="BSND",
    )
    out, final_state, chunk_states = ops.flash_kda(
        q, k, v, g, beta, metadata, A_log=A_log,
        layout="BSND", output_final_state=True,
        return_intermediate_states=True,
    )

torch.npu.synchronize()
assert out.shape == (batch, length, n_v, dim)
assert final_state.shape == (batch, n_v, dim, dim)
assert chunk_states.shape == (batch, 2, n_v, dim, dim)
```
