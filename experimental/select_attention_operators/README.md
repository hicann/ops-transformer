# Select Attention Operators

High performance Ascend-910B kernels for sparse attention pattern prediction for efficient LLM decoding. The kernels can be launched through python interface which we provide - see kernel usage examples in the [experiments](experiments) directory.

## Repo structure

```bash
.
|-- experiments - per kernel: test (functional correctness) and benchmark (time and bandwidth)
|   |-- 2_quest_prefill_metadata - constructing metadata after prefill
|   |-- 3_quest_block_select_paged - quest sparse mask predictor using metadata
|   |-- 4_quest_block_select_paged_w -  quest sparse mask predictor using metadata with extra sink+window features
|-- kernels - python packages, each having one or more ascendc kernels and a single torch interface
|   |-- select_attn_ops - predictor kernels (quest predictors of sparse pattern duting LLM decoding)
`-- scripts
    |-- build_kernels.sh - builds all kernels
    `-- init_cann.sh - initialize the environment and Ascend device version
```

## Requirements

Tested to work with:

- Ascend910B2, Ascend910B4
- CANN versions 8.0.RC3.beta1, 8.2.RC2, 8.3.RC1
- Python 3.11.10
- TorchNPU version 2.4.0, 2.5.1.post1
- see [requirements.txt](requirements.txt) for all other requirements

## Creating conda environment

create conda environment

```bash
conda create -n sa python=3.11.10 -y
conda activate sa
pip install -r requirements.txt
```

## Running (in conda environment)

activate conda and CANN environment, compile the operators and build their python api (as python packages):

```shell
source scripts/init_cann.sh Ascend910B4 # change Ascend910B4 to your card model
bash scripts/build_kernels.sh
```

Run all tests that are found in the experiments subdirectory

```bash
pytest -v experiments
```

## Usage 

Current best practise (proven at vllm-ascend) is to use `quest_prefill_metadata()` kernel for the creation of the metadata (after prefill) and every 128 tokens to update the metadata, and to use `quest_block_select_paged_in_out_w()` to predict important KV block indices given the current query vector of the token being decoded. For detailed usage examples refer to [experiments](experiments) directory.

The kernels are deployed with a very neat built in documentation:

```python
import torch_npu
from select_attn_ops import quest_block_select_paged_in_out_w
help(quest_block_select_paged_in_out_w)
```

<details>

<summary>Prints:</summary>

```
Help on built-in function quest_block_select_paged_in_out_w in module select_attn_ops:

quest_block_select_paged_in_out_w(...) method of builtins.PyCapsule instance
    quest_block_select_paged_in_out_w(query: torch.Tensor, maxblocks: torch.Tensor, minblocks: torch.Tensor, metadata_block_tables: torch.Tensor, seq_lens: torch.Tensor, tokens_since_metadata_update: int, selected_indices: torch.Tensor) -> None
    
    
    Alternative interface to the `quest_block_select_paged` kernel which predicts 
    the sparsity mask during decoding in the form of top-k important kv-block 
    indices for every KV-head in every request. The returned KV block ids 
    are not the indices in the KV-cache, but rather from their enumeration 
    from 0 to number of blocks in the sequence length being decoded.
    
    FEATURE 1) WITH PREALLOCATED OUTPUT TENSOR (selected_indices)
    
    FEATURE 2) "w" 2 stands for "window" i.e. the kernel decides whether to add local 
    window blocks ids to the selected indices based on the number of tokens 
    since the last update and based on the sequence length
    
    Args:
        query (torch.Tensor): Query vector of shape [B, H, D] (fp16 or bf16)
        maxblocks (torch.Tensor): Quest metadata with maximum vectors of 
                                every key-cache block of shape 
                                [num_meta_blocks, BLOCK_SIZE, N, D] (fp16 or bf16)
                                important: zeroes must be in place of metadata of non-existing kv blocks
        minblocks (torch.Tensor): Quest metadata with minimum vectors of 
                                every key-cache block of shape 
                                [num_meta_blocks, BLOCK_SIZE, N, D] (fp16 or bf16)
                                important: zeroes must be in place of metadata of non-existing kv blocks
        metadata_block_tables (torch.Tensor): Metadata block tables of 
                                            shape [B, MMBPR] (int32)
        seq_lens (torch.Tensor): Sequence length of each request in the batch
                            of shape [B] (int32)
        tokens_since_metadata_update (int) - number of tokens that were decoded 
                            since the last metadata update (note metadata update is 
                            done only on the multiple of BLOCK_SIZE tokens which is 
                            lower or equal to the sequence length at the moment of update)
                            set to -1 to disable selection of KV blocks for which the 
                            metadata doesn't exist.
        selected_indices (torch.Tensor): Selected indices vector of shape [B, N, k] (int32): 
                                Number of highest indices to return for every KV head
    
    Returns:
        <fills out the selected_indices tensor>
        
    
    Limitations: due to kernel's internal buffer design on 910B:
        D = 128
        BLOCK_SIZE = 128
        H / N <= BLOCK_SIZE
        MMBPR <= 6
        k % 8 == 0
```

</details>

## Development Workflow for a new kernel "OP"

1. Add new kernel implementations in the `kernels/` directory in one of 2 ways:
   1) under an existing python package e.g. `kernels/select_attn_ops/`. Then add your kernel code as new OP.cpp, add a compilation line to compile.sh, add a torch interface inside torch_interface.cpp
   2) as a new python package: `kernels/OP/`, with a OP.cpp kernel implementation; torch_interface.cpp, compile.sh, build.sh in it.
2. Create a dedicated experiment directory `experiments/5_OP` and implement in it the following programs:
    - *ref_OP.py* - start off by implementing a reference python model for correctness.
    - *gen_data_OP.py* - a function that produces a set of input tensors for your kernel.
    - *test_OP.py* with a smoke test (single run first) to validate correctness on a focused single input, then extend to automated pytesting across wide range of input shapes/data-types
    - *benchmark_OP.py* - measure performance (time, bandwidth)
