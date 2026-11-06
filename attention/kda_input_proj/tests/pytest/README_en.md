# KdaInputProj Operator Test Framework

English | [简体中文](./README.md)

## File Structure

```text
pytest/
├── README.md                          # Test framework description
├── pytest.ini                         # ci marker
├── test_run.sh                        # Local ST entry
├── test_kda_input_proj_accuracy.py    # End-to-end accuracy test cases for the four outputs
├── kda_input_proj_mx_quant_golden.py  # Stage1 DynamicMxQuant CPU golden
├── kda_input_proj_qkv_golden.py       # Stage2 QuantMatmul(qkv) CPU golden
└── kda_input_proj_bgg_golden.py       # beta / gate / g CPU golden
```

## Description

This framework uses pytest to verify the four outputs of KdaInputProj:

- **qkv**: On the CPU, `quant_x`/`x_scale` are generated according to the OCP MX quantization semantics of the kernel, and then MX matrix multiplication is performed to obtain the BF16 golden data.
- **beta**: FP32 `sigmoid(x @ w_beta)`.
- **gate / g**: BF16 matrix multiplication.

Only Ascend 950PR&950DT products are supported. Test cases are skipped on other devices.

You can specify the NPU through the environment variable `TEST_DEVICE_ID`. If it is not set, cards that are already occupied and cannot run `set_device` are skipped.

## Usage

Run the following command in the repository root directory:

```bash
python3 -m pytest -rA -s \
  attention/kda_input_proj/tests/pytest/test_kda_input_proj_accuracy.py \
  -v -m ci
```

Or run the following command in this directory:

```bash
bash test_run.sh single
# Specify the card ID: TEST_DEVICE_ID=5 bash test_run.sh single
```
