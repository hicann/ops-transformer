# FlashAttentionScoreGrad Operator Test Framework

English | [简体中文](./README.md)

## File Structure

pytest/

- test_case.py              # Test case set
- test_main.py              # Main execution program
- common.py                 # Common definitions file
- cpu_impl.py               # CPU implementation
- npu_impl.py               # NPU implementation

## Description

Based on the pytest test framework, this framework verifies the functions of the FA operator:

- **CPU side**: Reimplements the operator functions to generate golden data
- **NPU side**: Directly invokes the operator through TorchNPU to obtain actual data
- **Accuracy comparison**: Compares the accuracy of the CPU and NPU results to verify the operator functions

## Environment Configuration

### Prerequisites

1. Ensure that TorchNPU is the latest version
2. Source the CANN package environment variables

### Custom Package Invocation

Invocation of the custom package is supported.

## Usage

Run the following command in the pytest directory:

pytest -s
