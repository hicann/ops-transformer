# RecurrentKda Operator pytest

English | [简体中文](./README.md)

## Description

This directory verifies the NPU computation results of the RecurrentKda operator:

- Generate the CPU golden data with a PyTorch implementation.
- Call the NPU operator through `cann_ops_transformer`.
- Compare the output and the final state, and verify that invalid state strides are intercepted.

## File Description

- `test_accuracy.py`: test cases and the execution entry.
- `recurrent_kda_reference.py`: CPU golden implementation.
- `utils.py`: precision comparison tool.
- `test_run.sh`: pytest execution script.

## Environment Setup

Load the CANN and Python environments in the repository root directory, and build and install the single-operator wheel according to the specifications in
`torch_extension/README.md`:

```bash
cd <repo_root>
bash build.sh --torch_extension --ops=recurrent_kda --vendor_name=custom
python3 -m pip install build_out/*-1.0.0-py3-none-any.whl --force-reinstall --no-deps
export PYTHONPATH=$(pwd)/torch_extension:${PYTHONPATH}
```

`test_run.sh` only starts pytest and does not build or install the wheel.

If you use the custom operator package installed in the CANN directory, you also need to set:

```bash
export ASCEND_CUSTOM_OPP_PATH=${ASCEND_OPP_PATH}/vendors/custom_transformer:${ASCEND_CUSTOM_OPP_PATH}
export LD_LIBRARY_PATH=${ASCEND_OPP_PATH}/vendors/custom_transformer/op_api/lib:${LD_LIBRARY_PATH}
```

## Running Tests

Run the following commands in the pytest directory:

```bash
cd attention/recurrent_kda/tests/pytest
bash test_run.sh single
```

You can also run pytest directly:

```bash
python3 -m pytest -rA -s test_accuracy.py -v -m ci
```

NPU 0 is used by default; you can specify another device through an environment variable:

```bash
TEST_DEVICE_ID=1 bash test_run.sh single
```
