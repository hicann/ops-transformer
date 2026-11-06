# Mock HCCL Server Framework

English | [简体中文](./README.md)

## Overview

This framework provides a **general HCCL Server-side mock implementation** that replaces the real CCU Server / HCCL communication backend in a single-device environment. It is independent of specific MC2 operators. Any operator that interacts with the HCCL Server through the workspace message protocol (such as AllGatherMatmul and MatmulReduceScatter) can use this framework for single-device testing.

### Background: hcclClient / Server Architecture

In a real multi-device environment, the communication flow of MC2 operators is as follows:

```
Device (kernel)                      CCU (HCCL Server)
  │                                    │
  │  1. Fill sendMsgs[slot]           │
  │  2. Write commitTurnCnt[slot]     │
  │  ──────── workspace ──────────→    │
  │                                    │  3. Read commitTurnCnt and find a valid entry
  │                                    │  4. Read sendMsgs and perform collective communication
  │                                    │  5. Write finishedTurnCnt[slot]
  │  ←──────── workspace ─────────     │
  │  6. Poll finishedTurnCnt and continue │
  │                                    │
```

The kernel submits communication requests to the server through the message protocol in the workspace. The server performs the actual cross-rank data movement and writes back the completion flag through the workspace. This **workspace message protocol is operator-independent**. Different MC2 operators (AllGather, ReduceScatter, and so on) use the same protocol format and differ only in the commType field.

**This framework simulates the behavior of this server with a polling thread on the host side.**

```
Directory structure:
tests/mc2_mock_test_frame/              # This directory: general mock framework
├── mock_framework.h                    # C++ core (MockContextBuilder + MultiRankMockContext + MockHcclServer)
├── mock_framework_test.cpp             # C++ unit tests (protocol interaction verification)
├── mock_framework.cpp                  # pybind11 torch extension (Python binding)
└── README.md                           # This document

tests/torch_extension_tests/mc2/all_gather_matmul_v3/  # V3 operator tests
├── conftest.py                             # Common utilities for mock tests
├── test_all_gather_matmul_v3_mock.py       # Single-rank mock test
├── test_all_gather_matmul_v3_mock_multirank.py  # Multi-rank multi-stream mock test
└── test_all_gather_matmul_v3.py            # Real multi-device test (torchrun)
```

---

## 1. Workspace Message Protocol

The core of what MockHcclServer simulates is the workspace message protocol. This is the kernel ↔ server communication interface defined by HCCL, and it is independent of specific operators.

### 1.1 Workspace Layout

The workspace must be 512B-aligned and contains 4 regions:

```
Workspace (512B-aligned):
┌──────────────────────────────────────┐
│ sendMsgs[64]      (each 112B)        │ +0x0000   Communication requests from kernel to server
├──────────────────────────────────────┤
│ recvMsgs[64]      (each 112B)        │ +0x1C00   Responses from server to kernel (reserved)
├──────────────────────────────────────┤
│ commitTurnCnt[64] (each 64B)         │ +0x5800   Kernel commit flags
├──────────────────────────────────────┤
│ finishedTurnCnt[64](each 64B)        │ +0x6800   Server completion flags
└──────────────────────────────────────┘
```

### 1.2 Message Structure

**HcclMsg (112 bytes)** — Communication request filled in by the kernel:

| Offset | Field | Description |
|------|------|------|
| +0x00 | commType (u32) | Communication type: AllGather=6, ReduceScatter=7 |
| +0x04 | opType (u32) | Reduce operation type |
| +0x08 | sendBuffer (u64) | Device source address |
| +0x10 | recvBuffer (u64) | Device destination address |
| +0x18 | dataCnt (u64) | Number of elements |
| +0x20 | strideCount (u64) | Stride between ranks in recvBuf |
| +0x28 | msgValid (u32) | HCCL_MSG_VALID_MASK (0x5CDF123A) |
| +0x2C | hcclDataType (u32) | Data type: FP32=0, FP16=1, BF16=5, ... |
| +0x30 | rest[64] | Remaining fields |

**TurnCnt (64 bytes)** — Commit/completion counters:

| Offset | Field | Description |
|------|------|------|
| +0x00 | valid (u64) | COMMIT_VALID_MASK (987654321) indicates validity |
| +0x08 | cnt (u64) | Commit/completion count |
| +0x10 | reserved[6] | Padding to cache line alignment |

### 1.3 Protocol Flow

**Regular communication (AllGather/ReduceScatter):**

```
kernel:
  1. Fill in sendMsgs[slot] (commType, sendBuf, recvBuf, dataCnt, ...)
  2. Write commitTurnCnt[slot].valid = COMMIT_VALID_MASK

server (MockHcclServer):
  3. Poll commitTurnCnt; when a valid entry is found, read sendMsgs[slot]
  4. Perform the operation according to commType (single device: D2D memcpy sendBuf → recvBuf)
  5. Write finishedTurnCnt[slot].cnt = commitCnt
  6. Clear commitTurnCnt[slot].valid

kernel:
  7. Poll until finishedTurnCnt[slot].cnt >= the expected value, then continue
```

**Finalize (communication-end handshake):**

```
kernel Finalize():
  1. Submit the finalize commit (the last message)

server:
  2. Detect the finalize commit
  3. Write finishedTurnCnt[slot].cnt = FINALIZE_FINISH_CNT (1234567899999999999)

kernel:
  4. Detect FINALIZE_FINISH_CNT → the communication flow ends
```

---

## 2. Framework Components

### 2.1 MockHcclServer — General HCCL Server Mock

**Core responsibility:** Starts a polling thread on the host side, listens for communication requests in the workspace, and uses the passed-in input tensors of each rank to simulate real collective communication semantics.

During construction, you must pass in the input tensor of each rank (device pointers). When the server processes a communication request, it reads these tensors to simulate fetching data from remote ranks.

**Supported communication types:**

| commType | Operation | Mock Behavior |
|----------|------|-----------|
| 6 (AllGather) | Concatenation of chunks from all ranks | rankInputs[r] → recvBuf[r * stride]; the local device uses sendBuf |
| 7 (ReduceScatter) | reduce + scatter | Read all ranks through D2H → element-wise reduce on the host → write back chunk[localRank] through H2D |
| 2 (AllReduce) | Full reduce | Read all ranks through D2H → element-wise reduce on the host → write back the full result through H2D |
| 12 (AlltoAll) | Block rearrangement | block[localRank] of each rank → recvBuf[r] |

**Reduce operations:** SUM(0) / PROD(1) / MAX(2) / MIN(3) are supported. The host side performs the computation after FP16/BF16 ↔ float conversion.

### 2.2 MockContextBuilder — Single-Rank HCCL Context Construction

Constructs the `HcclA2CombineOpParam` structure (that is, commContext) required by the kernel on the device side. The `windowsIn[i]` of all ranks point to the same device memory, which makes the flag synchronization mechanism of the 910B AIV mode self-consistent.

### 2.3 MultiRankMockContext — Multi-Rank Context Construction

Allocates N independent windows (184MB each) plus N contexts. The windowsIn[] entries in the contexts cross-reference each other. Together with concurrent kernels on multiple streams, this implements real multi-rank communication.

### 2.4 Python Binding (mock_framework.cpp)

Exposed to Python through pybind11:

```python
import mock_hccl_ext

# Single-rank test
ctx = mock_hccl_ext.MockContext(rank_num=2, rank_id=0, device_id=0)
ctx.build()
server = mock_hccl_ext.MockServer(workspace_ptr=ctx.workspace_ptr(),
                                   rank_inputs=[t0, t1], local_rank_id=0)
server.start()
# ... Call the operator ...
server.wait_for_finalize(slot=0, timeout_ms=10000)
server.stop()

# Multi-rank test
mctx = mock_hccl_ext.MultiRankContext(rank_num=2, device_id=0)
mctx.build()
ctx_r0 = mctx.context_tensor(0)
ctx_r1 = mctx.context_tensor(1)
# Launch on separate streams...
```

---

## 3. Unit Tests

`mock_framework_test.cpp` verifies the protocol correctness of the mock server (independent of specific operators):

| Test | Verification Point |
|------|--------|
| TestContextBuilder | Context H2D/D2H; field readback is correct |
| TestServerProtocol | Manually write commitTurnCnt → the server responds with finishedTurnCnt |
| TestServerDataCopy | AllGather msg → the server performs the D2D copy → the recvBuf content is correct |
| TestFinalizeProtocol | regular msg + finalize commit → FINALIZE_FINISH_CNT response |

Compile and run:

```bash
ASCEND_HOME=~/Ascend/ascend-toolkit/latest
g++ -std=c++17 -O2 -I${ASCEND_HOME}/include \
  mock_framework_test.cpp \
  -L${ASCEND_HOME}/lib64 -lascendcl -lpthread \
  -Wl,-rpath,${ASCEND_HOME}/lib64 \
  -o /tmp/mock_framework_test
/tmp/mock_framework_test
```

---

## 4. How to Add Tests for a New Operator

This framework is operator-independent. Adding tests for a new MC2 operator requires only the following steps:

1. **Build the mock context** — `MockContextBuilder.Build(rankNum, rankId)`
2. **Start the mock server** — `MockHcclServer(workspace).Start()`
3. **Call the operator** — pass the mock context to the operator as the commContext input
4. **Wait for completion** — `WaitForFinalize()` waits for the Finalize handshake of the kernel
5. **Verify the result** — check whether the computation output of the operator is correct

---

## 5. Known Limitations

| Limitation | Description |
|------|------|
| Single device only | Multiple devices require real HCCL or a multi-device mock |
| D2D memcpy replaces collective communication | A single device has no cross-rank data, so AllGather/ReduceScatter degrade to memcpy |
| No RDMA in the mock context | IbVerbsData / aiRMAInfo are 0; cross-node scenarios are not supported |
| Server polling latency | Host D2H/H2D polling has a latency of about 50 us and does not reflect real communication timing |
| Flag self-resolution does not verify timing | On a single-device mock, a flag is readable immediately after being written, so multi-device race conditions cannot be verified |
