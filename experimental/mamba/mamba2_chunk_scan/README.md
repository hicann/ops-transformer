# mamba2_chunk_scan算子说明

### 功能和实现说明

mamba2_chunk_scan用于在MambaV2 Prefill阶段对chunk内状态执行selective scan运算：对来自前序chunk的传播状态、chunk内delta信息以及gating/bias进行结合，根据时间步顺序进行递推累计，生成当前chunk的有效输出。
该算子包含两个matmul和两部分vector计算，因此实现为CVCV融合算子，通过VC并行提升计算性能。

**计算流**

<img src="https://raw.atomgit.com/user-images/assets/7673863/246d0aa8-1ed9-45ff-97db-7ee759ca8334/image.png" height="300">

### Kernel输入输出（I/O）

**输入**

| Tensor | shape | dtype |
|-----|-----|-----|
| ct   | BCLGN   | FP16   |
| bt   | BCLGN   | FP16   |
| xt   | BCLHP   | FP16   |
| dt   | H   | FP16   |
| inter_attn   | BCHLP   | FP32   |
| dacs   | BCHL   | FP32   |
| dtout   | BCHL   | FP32   |

**输出**

| Tensor | shape | dtype |
|-----|-----|-----|
| final_attn   | BCLHP   | FP32   |

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

final_attn = torch.ops.npu_ops_transformer_ext.mamba2_chunk_scan(ct, bt, xt, dt, inter_attn, dacs, dtout)
```

**测试方法**

见当前目录tests/

```
python test_chunk_scan.py
```
