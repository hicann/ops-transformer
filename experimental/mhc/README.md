# mHC昇腾AscendC算子

面向昇腾NPU的mHC（Manifold-Constrained Hyper-Connections，流形约束超连接）算子AscendC实现，此算子由智子芯元KernelCAT智能体生成。

论文：[arXiv:2512.24880](https://arxiv.org/abs/2512.24880)

## 论文对应关系

mHC层更新公式:

```
x_{l+1} = H_res · x_l + H_post^T · F(H_pre · x_l, W_l)
```

| 公式项 | 算子 | 功能说明 |
|------|----------|----------|
| `H_pre · x` | mhc_pre | N路流 → 1 路（加权求和） |
| `H_post^T · y` | mhc_post | 1 路 → N路流（广播缩放） |
| `H_res · x` | mhc_res | N路 → N路（流间混合） |

以上算子仅处理张量收缩运算。训练逻辑（Sinkhorn投影、权重约束等）由上层框架负责。

## 算子公式

| 算子 | 公式 |
|----------|--------|
| **mhc_pre** | `out[b,s,d] = Σ_n h[n] × x[b*N+n,s,d]` |
| **mhc_post** | `out[b*N+n,s,d] = x[b,s,d] × h[n]` |
| **mhc_res** | `out[b*N+t,s,d] = Σ_r h[r,t] × x[b*N+r,s,d]` |

## 数据类型与精度

| 数据类型 | 精度（对比einsum） |
|-------|----------------------|
| fp32 | bit-exact |
| fp16 | atol=1e-4, rtol=1e-3 |
| bf16 | atol=1e-3, rtol=4e-3 |

## 实现范围

权重张量（`h_pre`、`h_post`、`h_res`）为**逐层静态参数**:

- 形状: `[num_streams]`或`[num_streams, num_streams]`
- 在batch与序列维度上共享
- 与 [tokenbender/mHC](https://github.com/tokenbender/mHC-manifold-constrained-hyper-connections)开源实现保持一致

逐样本或逐token的动态权重需要扩展接口。

## 完整 CANN 工程结构算子

除上述轻量原型算子外，本目录还包含完整 CANN 工程结构（op_host/op_kernel/aclnn/torch_extension）的算子：

| 算子 | 说明 |
|------|------|
| `mhc_pre_sinkhorn_premix` | 由正式算子 `mhc/mhc_pre_sinkhorn` 迁移并增强：新增 premix 可选输入（仅 Ascend 950PR&950DT系列产品），传入时 hin 使用 premix 加权；needBackward 时仍输出 hPre 等中间变量。构建需开启 `--experimental`，详见 [mhc_pre_sinkhorn_premix/README.md](mhc_pre_sinkhorn_premix/README.md) |

## 硬件要求

| 项目 | 规格 |
|------|------|
| **运行设备** | Atlas A2 系列 |
| **CANN版本** | 8.3.RC2 / 8.5 (已验证) |
| **驱动版本** | 25.0.rc1.2 |

## 编译构建

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh

cd mhc_pre  # 或mhc_post, mhc_res
mkdir -p build && cd build
cmake .. -DSOC_VERSION=ascend910b2 && make -j
cd .. && python setup.py build_ext --inplace
```

## 测试

```bash
# C++
cd build && LD_LIBRARY_PATH=./lib:$LD_LIBRARY_PATH ./test_multi_dtype

# Python
LD_LIBRARY_PATH=./build/lib:$LD_LIBRARY_PATH python mhc_pre_ops.py
```

## 使用方式

```python
import mhc_pre_ext

x = torch.randn(B * N, S, D, device='npu')  # [batch*streams, seq, dim]
h = torch.randn(N, device='npu')            # [streams]
out = mhc_pre_ext.forward(x, h)             # [batch, seq, dim]
```

## 性能（对比torch.einsum, Ascend 910B2）

| 算子 | 加速比 |
|----------|--------|
| mhc_pre | 24x ~ 52x |
| mhc_post | 2x ~ 5x |
| mhc_res | 24x ~ 50x |

## KernelCAT内测申请

KernelCAT限时免费内测中，欢迎体验：https://kernelcat.autokernel.cn

## 参考文献

本算子实现参考了mHC论文：

```bibtex
@article{xie2025mhc,
  title={mHC: Manifold-Constrained Hyper-Connections},
  author={Xie, Zhenda and Wei, Yixuan and Cao, Huanqi and Zhao, Chenggang and Deng, Chengqi and Li, Jiashi and Dai, Damai and Gao, Huazuo and Chang, Jiang and Yu, Kuai and Zhao, Liang and Zhou, Shangyan and Xu, Zhean and Zhang, Zhengyan and Zeng, Wangding and Hu, Shengding and Wang, Yuqing and Yuan, Jingyang and Wang, Lean and Liang, Wenfeng and others},
  journal={arXiv preprint arXiv:2512.24880},
  year={2025}
}
```

## 许可证

CANN Open Software License Agreement Version 2.0 - 详见 [LICENSE](LICENSE)文件。
