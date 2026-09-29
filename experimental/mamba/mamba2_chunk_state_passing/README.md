# mamba2_chunk_state_passing算子说明

### 功能和实现说明

mamba2_chunk_state_passing用于在MambaV2 Prefill阶段进行跨chunk状态传递，将chunk内计算得到的状态按照时间顺序依次传递，并在各chunk之间执行指数衰减和新状态叠加，从而形成完整的跨chunk状态序列；同时返回最终的全局状态，用于下一阶段推理。该算子通常在chunk_state之后调用，用于将chunk内状态扩展为跨chunk的连续状态序列，并生成下一阶段使用的最终状态。同时，本算子在状态传递完成后，对重排后的状态张量与ct执行基于Cube的批量矩阵乘（states @ ct），实现类似inter-attention的跨chunk状态混合。
该算子实现为Vector+Cube融合算子，通过VC并行提升计算性能。

**计算流**

<img src="https://raw.atomgit.com/user-images/assets/7673863/49bacbe4-46b1-4780-88e6-2c6327bd97e1/image.png" height="300">

### Kernel输入输出（I/O）

**输入**

| Tensor | shape | dtype |
|-----|-----|-----|
| dacs   | BCLH   | FP32   |
| init_state   | BHZ   | FP32   |
| states   | BCHZ   | FP32   |
| ct   | BCLGN   | FP16   |

**输出**

| Tensor | shape | dtype |
|-----|-----|-----|
| inter_attn   | BCHLP   | FP32   |
| final_state  | BHNP    | FP32   |

**参数说明：**
B: batch size
C: number of chunks
L: chunk size
H: number of head
G: ngroups
N: state size
P: head dim
其中C*L为padding后的序列长度

**调用方式**

```
import npu_ops_transformer_ext

inter_attn, final_state = torch.ops.npu_ops_transformer_ext.mamba2_chunk_state_passing(dacs, init_state, states, ct)
```

**测试方法**

见当前目录tests/

```
python test_chunk_state_passing.py
```
