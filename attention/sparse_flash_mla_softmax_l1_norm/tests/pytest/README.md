# SparseFlashMlaSoftmaxL1Norm pytest 测试

本目录采用算子仓统一的 pytest 结构：

```text
pytest/
├── test_case.py   # 参数化用例
├── test_main.py   # pytest 入口
├── common.py      # 公共定义
├── cpu_impl.py    # CPU 标杆和输入生成
└── npu_impl.py    # PyTorch API/NPU 调用
```

在已经安装 custom ops 包并加载 CANN 环境的 NPU 机器上执行：

```bash
cd attention/sparse_flash_mla_softmax_l1_norm/tests/pytest
pytest -s
```

用例覆盖 A2/A3 arch22 的 FP16/BF16、ND、BSND/TND、D=256/512、稠密/稀疏、
mask_mode=0/3、非整块 head 和可选序列输入。CPU 实现生成 FP64 标杆，NPU
实现通过 `cann_ops_transformer.sparse_flash_mla_softmax_l1_norm` 调用算子；
输入和输出只在测试进程内存中存在，不提交二进制数据。
