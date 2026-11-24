# DualLevelGroupedMatmul 示例

两个示例执行同一组双级量化 GMM。M=32、K=640、N=128，两个专家各处理 16 行，权重为非转置 FRACTAL_NZ，输出为 BF16。

- x 和 weight 的每个 FP4 元素均为 1，每字节为 `0x22`。两侧 E8M0 scale 的编码分别为 127 和 128，对应 1 和 2。
- 第一段 512 个 K 元素的两侧 level0 scale 均为 1。余下 128 个 K 元素的两侧 level0 scale 均为 2。两个专家分别使用 bias 16 和 32。
- 输出形状为 `(32,128)`，按行检查每个元素。矩阵乘结果为 `2 × (512 + 128 × 4) = 2048`，加 bias 后前 16 行为 2064，后 16 行为 2080。校验失败时程序返回非零状态。

## 构建并安装算子包

准备 Ascend 950 的 CANN 开发环境。构建前，将包含双级量化 Blaze 组件的配套 `ops-tensor` 源码放在 `ops-transformer` 同级目录，并在 `ops-tensor` 根目录运行 `git submodule update --init include/tensor_api`。

在 `ops-transformer` 根目录运行以下命令。将安装路径替换为实际目录，并按环境填写 CANN 路径和安装包文件名。

```bash
source /usr/local/Ascend/cann/set_env.sh
bash build.sh --pkg --experimental --soc=ascend950 \
  --ops=dual_level_grouped_matmul --vendor_name=dual_level_gmm -j4
bash build_out/cann-ops-transformer-dual_level_gmm_linux-x86_64.run \
  --quiet --nox11 --install-path=/absolute/install/path
source /absolute/install/path/vendors/dual_level_gmm_transformer/bin/set_env.bash
```

## 运行 ACLNN 示例

在加载上述环境后，从 `ops-transformer` 根目录执行仓库的[标准示例命令](../../../../docs/zh/invocation/quick_op_invocation.md)。

```bash
bash build.sh --experimental --run_example dual_level_grouped_matmul eager cust \
  --example_name=dual_level_grouped_matmul_weight_nz --vendor_name=dual_level_gmm --soc=ascend950
```

程序使用设备 0，在内存中构造张量，通过已安装算子包执行两段式 ACLNN 接口并同步 Stream。

## 运行 PyTorch 示例

在同一环境安装与 CANN 配套的 PyTorch、TorchNPU，以及编译 Torch 扩展所需的 C++ 编译器和 Ninja。从 `ops-transformer` 根目录执行以下命令。其中 `dlgmm` 为 Python 扩展的包名后缀。

```bash
TORCH_EXTENSION_EXPERIMENTAL=TRUE TORCH_EXTENSION_OPS=dual_level_grouped_matmul \
  TORCH_EXTENSION_VENDOR=dlgmm python3 -m pip install ./torch_extension --no-build-isolation
CANN_TRANSFORMER_PACKAGE=cann_ops_transformer_dlgmm \
  python3 experimental/gmm/dual_level_grouped_matmul/examples/test_torch_dual_level_grouped_matmul.py
```

首次在 NPU 上调用时，Torch 扩展会编译 C++ 适配层。转置、FP16、无 bias 和稀疏分组场景见 [ST 测试说明](../tests/README.md)。
