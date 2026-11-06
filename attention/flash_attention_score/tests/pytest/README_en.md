# FlashAttentionScore Operator Test Framework

English | [简体中文](./README.md)

## File Structure

pytest/

- test_case.py              # Test case collection
- test_flash_attn.py        # Main execution program
- test_utils.py             # Utility methods
- cpu_impl.py               # cpu implementation
- npu_impl.py               # npu implementation

## Function Description

Based on the pytest test framework, this framework verifies the functions of the FA operator:

- **CPU side**: Reproduces the operator functions to generate golden data
- **NPU side**: Directly invokes the operator through TorchNPU to obtain actual data
- **Accuracy comparison**: Compares the accuracy of CPU and NPU results to verify operator functions

## Environment Setup

### Prerequisites

1. Confirm that TorchNPU is the latest version
2. Source the environment variables of the CANN package

### Custom Package Invocation

Custom package invocation is supported.

## Usage

Run the following command in the pytest folder directory:

pytest -s
