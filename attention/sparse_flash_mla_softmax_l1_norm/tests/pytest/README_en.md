# SparseFlashMlaSoftmaxL1Norm pytest tests

This directory follows the common operator pytest layout:

```text
pytest/
├── test_case.py   # parameterized cases
├── test_main.py   # pytest entry point
├── common.py      # shared definitions
├── cpu_impl.py    # CPU reference and input generation
└── npu_impl.py    # PyTorch API/NPU invocation
```

Run on an NPU host after installing the custom ops package and loading the CANN
environment:

```bash
cd attention/sparse_flash_mla_softmax_l1_norm/tests/pytest
pytest -s
```

The cases cover FP16/BF16, ND, BSND/TND, D=256/512, dense/sparse inputs,
mask_mode=0/3, non-aligned head tails, and optional sequence inputs. The CPU
implementation produces the FP64 reference and the NPU implementation calls
`cann_ops_transformer.sparse_flash_mla_softmax_l1_norm`. Tensor data exists only
in the test process and is not stored in the repository.
