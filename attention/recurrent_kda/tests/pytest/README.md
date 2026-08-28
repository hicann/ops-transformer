# RecurrentKda 算子 pytest

## 功能说明

该目录用于验证 RecurrentKda 算子的 NPU 计算结果：

- 使用 PyTorch 实现生成 CPU golden。
- 通过 `cann_ops_transformer` 调用 NPU 算子。
- 对比输出和最终状态，并验证非法 state stride 的拦截行为。

## 文件说明

- `test_accuracy.py`：测试用例及执行入口。
- `recurrent_kda_reference.py`：CPU golden 实现。
- `utils.py`：精度对比工具。
- `test_run.sh`：pytest 执行脚本。

## 环境配置

在仓库根目录加载 CANN 和 Python 环境，并按照仓库
`torch_extension/README.md` 的规范构建、安装单算子 wheel：

```bash
cd <repo_root>
bash build.sh --torch_extension --ops=recurrent_kda --vendor_name=custom
python3 -m pip install build_out/*-1.0.0-py3-none-any.whl --force-reinstall --no-deps
export PYTHONPATH=$(pwd)/torch_extension:${PYTHONPATH}
```

`test_run.sh` 只负责启动 pytest，不执行 wheel 构建或安装。

如果使用安装到 CANN 目录下的 custom 算子包，还需要设置：

```bash
export ASCEND_CUSTOM_OPP_PATH=${ASCEND_OPP_PATH}/vendors/custom_transformer:${ASCEND_CUSTOM_OPP_PATH}
export LD_LIBRARY_PATH=${ASCEND_OPP_PATH}/vendors/custom_transformer/op_api/lib:${LD_LIBRARY_PATH}
```

## 运行测试

在 pytest 目录执行：

```bash
cd attention/recurrent_kda/tests/pytest
bash test_run.sh single
```

也可以直接执行 pytest：

```bash
python3 -m pytest -rA -s test_accuracy.py -v -m ci
```

默认使用 NPU 0，可通过环境变量指定其他设备：

```bash
TEST_DEVICE_ID=1 bash test_run.sh single
```
