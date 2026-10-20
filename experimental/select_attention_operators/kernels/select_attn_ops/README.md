# Select Attention Operators Python Library

## 1 - Quest Prefill Metadata (initialization of metadata) operator

Operator `quest_prefill_metadata` computes initial metadata from the K cache. Based on the paper [Quest ICML2024 paper](https://arxiv.org/abs/2406.10774).

__Brief functionality description:__ in every K-cache block - a D-dimentional vector metadata-MAX-vector is computed (maximum along token dimension -> D to 1 reduction), and for BLOCK_SIZE of such blocks (16384 tokens) we obtain BLOCK_SIZE such metadata-MAX-vectors which are packed into a single metadata-MAX-block. This metadata-block is be written into a specially determined region _maxblocks_ in GM. Same procedure, but with minimum instead of maximum reduction, is performed to determine a metadata-MIN-block, which is stored in the _minblocks_ in the same specially predetermined block index. These special predetermined indices are an argument passed to the kernel "metadata_block_tables" - a 2D table with MMBPR indices per request.

__Operator prototype:__

```c++
/**
 * @brief Interface the `quest prefill metadata` kernel.  This is the interface
 *        function which is invoked from the python level. It handles
 *        1. resolving tensor shapes
 *        2. Passing the pointers of the input tensors to the kernel
 *        3. Invocation of the kernel
 *        4. Returning the pointer of the output
 *
 * @param [in] k_cache (num_kv_blocks, BLOCK_SIZE, N, D)
 * @param [in] block_tables (B, MKBPR) - for every request b: block_tables[b] is
                             a list of kv block indices
 * @param [in] seq_lens (B,) - sequence length (in token number) per request
 * @param [in] metadata_block_tables (B, MMBPR) - for every request b: block_tables[b] is
                             a list of metadata block indices
 * @param [out] maxblocks (num_meta_blocks, BLOCK_SIZE, N, D) - maxblock metadata, arranged in blocks of equivalent size to kv_cache blocks
 * @param [out] minblocks (num_meta_blocks, BLOCK_SIZE, N, D) - maxblock metadata, arranged in blocks of equivalent size to kv_cache blocks
 *
 * note: num_kv_blocks can be equal to num_kv_blocks and one may even pass maxblocks = k_cache, minblocks = v_cache to reuse the same page tables of vllm
 */
void quest_prefill_metadata(at::Tensor k_cache,
                            at::Tensor block_tables,
                            at::Tensor seq_lens,
                            at::Tensor metadata_block_tables,
                            at::Tensor maxblocks,
                            at::Tensor minblocks
)
```

Operator sizing parameters are:
 - `B` - batch size
 - `N` - number of KV heads
 - `BLOCK_SIZE` - number of tokens that fits in one maxblock and one minblock (default: 128)
 - `D` - head dimension (default: 128)
 * `MKBPR` - maximum number of blocks in every entry of the block_tables (num columns)
 * `MMBPR` - maximum number of blocks in every entry of the metadata_block_tables (num columns)

__Kernel Limitations__: BLOCK_SIZE=D=128

### Usage
Once the operator was built, it can be used in your python code as follows:
```python
import torch
import torch_npu
from select_attn_ops import quest_prefill_metadata

# === create dummy inputs ===
device = torch.device('npu:0')
dtype_ind, dtype_val = torch.int32, torch.float16
B, N, BLOCK_SIZE, D = 4, 8, 128, 128
MKBPR = 200 # number of kb blocks in every request
MMBPR = (MKBPR + BLOCK_SIZE - 1) // BLOCK_SIZE
num_kv_blocks = B * MKBPR
num_meta_blocks = B * MMBPR
max_seq_len_per_req = BLOCK_SIZE * MKBPR
seq_lens = torch.tensor([max_seq_len_per_req]*B, dtype=dtype_ind, device=device)
# k_cache:
k_cache = torch.randn(num_kv_blocks, BLOCK_SIZE, N, D, dtype=dtype_val, device=device)
# block tables and metadata block table - containing block ids:
perm_kv_blk_ids = torch.randperm(num_kv_blocks, device=device)[:num_kv_blocks]
block_tables = perm_kv_blk_ids.reshape((B, MKBPR)).to(dtype=dtype_ind, device=device)
perm_meta_blk_ids = torch.randperm(num_meta_blocks, device=device)[:num_meta_blocks]
metadata_block_tables = perm_meta_blk_ids.reshape((B, MMBPR)).to(dtype=dtype_ind, device=device)
# placeholder for kernel outputs:
maxblocks = torch.zeros(num_meta_blocks, BLOCK_SIZE, N, D, dtype=dtype_val, device=device)
minblocks = torch.zeros(num_meta_blocks, BLOCK_SIZE, N, D, dtype=dtype_val, device=device)

# === run the quest prefill metadata operator, the outputs will be filled into (maxblocks, minblocks) ===
quest_prefill_metadata(k_cache, seq_lens, block_tables, metadata_block_tables, maxblocks, minblocks)

```

### TODOs
1. Double check why bf16 passes even though the whole min/max processing happens in fp16.

### Good practices
1. If you only modify the operator file `quest_prefill_metadata.cpp`, you need to only re-run compile.sh to start observing the effect.
2. If you modify the `torch_interface.cpp`, you need to rebuild the select_attn_ops python package using build.sh.

## 2 - Quest Block Select - predict sparsity pattern for LLM decoding

Operator `quest_block_select_paged` is implemented in this directory. The operator allows to predict the top k important KV-blocks based on their metadata (maxblocks and minblocks, arranged in blocks of BLOCK_SIZExBLOCK_SIZE) and the incoming query. The implementation is a vector-only kernel that follows the [Quest ICML2024 paper](https://arxiv.org/abs/2406.10774). We extend the original implementation with GQA support: query heads are grouped and averaged such that the prediction of the important k blocks are returned per KV-head and not per attention head.

This operator works with arbitrary number of metadata blocks, which is essential for handling variable-length sequences efficiently in attention mechanisms. The metadata is managed in VLLM-compatible metadata blocks, which can be stored in the same blocks as the KV-cache.

```text
Input:
    query - fp16/bf16 tensor of shape (B, H, D)
    maxblocks - fp16/bf16 tensor of shape (num_meta_blocks, BLOCK_SIZE, N, D) - metadata with maximum vectors for each key-cache block
    minblocks - fp16/bf16 tensor of shape (num_meta_blocks, BLOCK_SIZE, N, D) - metadata with minimum vectors for each key-cache block
    metadata_block_tables - int32 tensor of shape (B, MMBPR) - block tables mapping sequences to metadata blocks
    seq_lens - int32 tensor of shape (B) - sequence length of each request in the batch
    k - integer specifying how many block-indices to return for each KV-head

Algorithm steps:
1. For every batch, reduce-mean the query tensor across H dimension
    such that every group of H/N vectors of shape D are reduced to one vector
    of shape D denoted as "grouped_query[b,n]".
2. For each batch b and KV-head n:
    2.1. Use metadata_block_tables[b] to locate the relevant metadata blocks for the sequence
    2.2. For each metadata block in the sequence:
        2.2.1. product_max, product_min = Elementwise-multiply grouped_query[b,n] with each maxblock and minblock vector
        2.2.2. channel_max_product = Elementwise-max between the two products (product_min, product_max)
        2.2.3. block_scores = Reduce-sum the last dimension of approx_attention (D to 1)
    2.3. selected_values, selected_indices = Find the top-k indices across all relevant blocks
3. Return selected_indices
```
The operator prototype is as follows.  
```c++
/** @brief Interface the `quest_block_select_paged` kernel.
 *
 * @param [in] query Tensor(fp16) containing the query vector [B,H,D]
 * @param [in] maxblocks Tensor(fp16) containing quest metadata with the maximum vectors of every key-cache block (num_meta_blocks, BLOCK_SIZE, N, D)
 * @param [in] minblocks Tensor(fp16) containing quest metadata with the minimum vectors of every key-cache block (num_meta_blocks, BLOCK_SIZE, N, D)
 * @param [in] metadata_block_tables Tensor(int32) containing the metadata block tables [B,MMBPR]
 * @param [in] seq_lens Tensor(int32) containing sequence length of each request in the batch [B]
 * @param [in] k natural number of highest indices to return for every KV head
 *
 * @returns selected_indices - Tensor(int32) where the kernel will output the selected indices vector. [B,N,k]
 */
at::Tensor quest_block_select_paged(at::Tensor query,
                                    at::Tensor maxblocks,
                                    at::Tensor minblocks,
                                    at::Tensor metadata_block_tables,
                                    at::Tensor seq_lens,
                                    int k
);



/** @brief Alternative interface the `quest_block_select_paged` kernel with PREALLOCATED OUTPUT
 * @param [in] query Tensor(fp16) the query vector [B,H,D]
 * @param [in] maxblocks Tensor(fp16) quest metadata with the maximum vectors of every key-cache block (num_meta_blocks, BLOCK_SIZE, N, D)
 * @param [in] minblocks Tensor(fp16) quest metadata with the minimum vectors of every key-cache block (num_meta_blocks, BLOCK_SIZE, N, D)
 * @param [in] metadata_block_tables Tensor(int32) the metadata block tables [B,MMBPR]
 * @param [in] seq_lens Tensor(int32) sequence length of each request in the batch [B]
 * @param [in] k natural number of highest indices to return for every KV head
 * @param [out] selected_indices - Tensor(int32) where the kernel will output the selected
 *          indices vector. [B,N,k]
 */
void quest_block_select_paged_in_out(at::Tensor query,
                                     at::Tensor maxblocks,
                                     at::Tensor minblocks,
                                     at::Tensor metadata_block_tables,
                                     at::Tensor seq_lens,
                                     at::Tensor selected_indices
);


/**
 * @brief Alternative Interface the `quest_block_select_paged` kernel which
 *        ** FEATURE 1 ** WITH PREALLOCATED OUTPUT TENSOR (selected_indices)
 *        ** FEATURE 2 ** "w" 2 stands for "window" i.e. the kernel decides
 *        whether to add local window blocks ids to the selected indices based
 *        on the number of tokens since the last update and based on the sequence
 *       length
 *
 * @param [in] query Tensor(fp16) the query vector [B,H,D]
 * @param [in] maxblocks Tensor(fp16) quest metadata with the maximum vectors
 *                       of every key-cache block (num_meta_blocks, BLOCK_SIZE, N, D)
 * @param [in] minblocks Tensor(fp16) quest metadata with the minimum vectors
 *                       of every key-cache block (num_meta_blocks, BLOCK_SIZE, N, D)
 * @param [in] metadata_block_tables Tensor(int32) the metadata block tables [B,MMBPR]
 * @param [in] seq_lens Tensor(int32) sequence length of each request in the batch [B]
 * @param [in] tokens_since_metadata_update
 * @param [out] selected_indices - Tensor(int32) where the kernel will output the
 *                                 selected indices vector. [B,N,k]
 *
 */
void quest_block_select_paged_in_out_w(at::Tensor query,
                              at::Tensor maxblocks,
                              at::Tensor minblocks,
                              at::Tensor metadata_block_tables,
                              at::Tensor seq_lens,
                              int tokens_since_metadata_update,
                              at::Tensor selected_indices
)
```

Operator sizing parameters are:
 - `B` - batch size
 - `N` - number of KV heads
 - `H` - number of attention (aka query) heads, must be a multiple of N
 - `num_meta_blocks` - total number of metadata blocks in the pool
 - `BLOCK_SIZE` - number of tokens that fits in one maxblock and one minblock (default: 128)
 - `D` - head dimension (default: 128)
 - `MMBPR` - maximum metadata blocks per request
 - `k` - number of important block indices to return for each KV head (top-k, default:8)

Implementation Notes
 - Performance limitations require *D* = 128 and *BLOCK_SIZE* = 128
 - Requires Grouped Query Attention (GQA) configuration where H % N == 0
 - When for a given request r: `k > seq_lens[r] // BLOCK_SIZE` (i.e. there is not yet k blocks in the kv-cache at all) the kernel will run nevertheless, providing top-k indices, although some of them garbage (computed on the filler part of the metadata block) - it is therefore the user's responsibility to use this kernel with a k that is larger than the number of the KV blocks.

### Build the operators and their torch extension

```bash
./build.sh
```

### Usage

Once the operator was built, it can be used in your python code as follows:

```python
import torch
import torch_npu
from select_attn_ops import quest_block_select_paged, quest_block_select_paged_in_out, quest_block_select_paged_in_out_w

# Create dummy inputs (ideally minblocks and maxblocks should be generated from KV cache to have proper 0 paddings)
B, H, N, BLOCK_SIZE, D = 20, 32, 8, 128, 128
MMBPR = 1
k = 8
dtype = torch.bfloat16
device = "npu:0"
num_meta_blocks = B * MMBPR
max_seq_len = MMBPR * BLOCK_SIZE * BLOCK_SIZE
query = torch.empty(B, H, D, dtype=dtype).uniform_(-1, 1).to(device).contiguous()
maxblocks = torch.empty(num_meta_blocks, BLOCK_SIZE, N, D, dtype=dtype).uniform_(-1, 1).to(device).contiguous()
minblocks = torch.empty(num_meta_blocks, BLOCK_SIZE, N, D, dtype=dtype).uniform_(-1, 1).to(device).contiguous()
metadata_block_tables = torch.randint(0, num_meta_blocks, (B, MMBPR), dtype=torch.int32).to(device).contiguous()
seq_lens = torch.randint(max_seq_len, max_seq_len+1, (B,), dtype=torch.int32).to(device).contiguous()

# Option 1: Run the quest paged operator and let it allocate the output
ids = quest_block_select_paged(query, maxblocks, minblocks, metadata_block_tables, seq_lens, k)
# Output shape: [B, N, k]

# Option 2: Run the quest predictor with a pre-allocated output tensor:
ids = torch.zeros((B, N, k), dtype=torch.int32, device=device) # k has to be rounded to neareast multiple of 8
quest_block_select_paged_in_out(query, maxblocks, minblocks, metadata_block_tables, seq_lens, ids)
# ids were filled with the predicted indices

# Option 3: Run the quest predictor with  preallocated output tensor and specify the number of tokens that were decoded since the most recent metadata update such that the sink (block 0) and local window blocks are forced to be chosen. This has good practical accuracy impact.
ids = torch.zeros((B, N, k), dtype=torch.int32, device=device) # k has to be rounded to neareast multiple of 8
tokens_since_metadata_update = 0
quest_block_select_paged_in_out_w(query, maxblocks, minblocks, metadata_block_tables, seq_lens, tokens_since_metadata_update, ids)
# ids some indices will be exceeding the sequence length in this output - reason is that metadata was not properly padded for non-existing meta-vectors. quest_prefill_metadata kernel should be used for creating metadata out of KV cache after prefill.
```

### TODOs

1. improve bf16 performance by tiling across block size dimension - to not allocate huge 64KB buffers for maxblock and minblock, and to enable full double buffering when loading the minblocks and the maxblocks
2. [quest_block_select_paged_in_out_w] check precisely what happens at boundary crossing from 1 metablock to 2nd
3. [quest_block_select_paged_in_out_w] - avoid summing up metadata vectors that are invalid, even when they are not zeroes - look at the seq_lens - tokens_since_metadata_update to determine that

## Good practices

1. If you only modify the operator file `quest_block_select_paged.cpp`, you need to only re-run compile.sh to start observing the effect.
2. If you modify the `torch_interface.cpp`, you need to rebuild the select_attn_ops python package using build.sh.
