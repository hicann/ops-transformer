# MlaPrologV2 Operator Test Framework

English | [简体中文](./README.md)

## File Structure

pytest/

- test.py                      # Main program for running generalized pytest test cases
- testcases.py                    # Input parameter configuration for generalized test cases
- check_valid_param.py            # Input parameter check and precision comparison
- prologv2_no_quant_pa_bsnd.py       # CPU-side operator logic implementation for obtaining golden data, direct NPU operator invocation
- pytest.ini                      # Creates test markers for CI single-operator and graph modes

## Description

Based on the pytest test framework, this framework implements functional verification of the MlaPrologV2 operator:

- **CPU side**: Replicates the operator logic to generate golden data
- **NPU side**: Directly invokes the operator through TorchNPU to obtain actual data; entering the graph through torchair is not yet supported
- **Precision comparison**: Compares the precision of CPU and NPU results to verify the operator function

### Currently Implemented Scope

✅**Implemented**: Basic MlaPrologV2 non-quantized PA scenario
❌**Not implemented**: MlaPrologV2 partial-quantization and full-quantization scenarios

### Parameter Restrictions

- **Data format**: BF16
- **B**: Batch indicates the batch size of input samples, ranging from 1 to 65536.
- **S**: Seq-Length indicates the sequence length of input samples, ranging from 1 to 16.
- **He**: Head-Size indicates the size of the hidden layer, which is 7168.
- **Hcq**: Dimension of the low-rank q matrix, which is 1536.
- **N**: Head-Num indicates the number of heads, which can be 8, 16, 32, 64, or 128.
- **Hckv**: Dimension of the low-rank kv matrix, which is 512.
- **D**: Dimension of qk excluding positional encoding, which is 128.
- **Dr**: Dimension of qk positional encoding, which is 64.
- **Nkv**: Number of kv heads, which is 1.
- **BlockNum**: Number of blocks in the PagedAttention scenario. The value is obtained by rounding up B*Skv/BlockSize, where Skv indicates the sequence length of kv.
- **BlockSize**: Block size in the PagedAttention scenario, which can be 16 or 128.
- **T**: Size after the B and S axes are combined, ranging from 1 to 1048576.
- **cache_mode**: PA_BSND / PA_NZ. If the user does not specify a value, the default value "PA_BSND" is passed.

## Environment Setup

### Prerequisites

1. Ensure that TorchNPU is of the latest version
2. Source the CANN package

### Custom Package Invocation

Invoking the custom package is supported.

## Usage

Run the following commands in the pytest folder:

### Running Mode: Run Generalized Test Cases

Direct single-operator invocation + graph mode

```bash
python3 -m pytest -rA -s test.py
```

Direct single-operator invocation

```bash
python3 -m pytest -rA -s test.py -v -m ci
```

Graph mode

```bash
python3 -m pytest -rA -s test.py -v -m graph
```
