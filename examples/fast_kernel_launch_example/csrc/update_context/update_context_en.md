# MC2 Context Creation Flow

## Context Structure

```cpp
struct Mc2ContextStru {
    uint64_t epRankId;           // Current rank ID
    uint64_t kfcContextAddr;     // Address of the KFC communication context
    uint64_t epHcclBuffer[1024]; // Array of HCCL Buffer addresses of all ranks
};
```

## Creation Flow

```
update_context(group_ep, ep_world_size)
    │
    ▼
GetMc2Context(mc2ContextHost, epWorldSize, bufferSize, groupEpStr)
    │
    ├─► HcclKfcAllocOpArgs(&opArgs)           // 1. Allocate the communication configuration object
    │
    ├─► HcclKfcOpArgsSetCommEngine(opArgs, AIV) // 2. Set the communication engine to AIV
    │
    ├─► CreateHcclContext(comm, opArgs, worldSize, groupName)
    │       │
    │       ├─► HcclKfcOpArgsSetAlgConfig(opArgs, algConfig)  // Set the algorithm configuration
    │       ├─► HcclCommGetHandleWithName(groupName, &comm)   // Obtain the communication handle
    │       ├─► HcclCreateOpResCtx(comm, opType, opArgs)      // Create the operation resource context
    │       ├─► HcclGetRankSize(comm, &worldSize)             // Obtain the world size
    │       └─► HcclGetRankId(comm, &rankId)                  // Obtain the rank ID
    │
    ├─► CreatMc2Context(comm, worldSize, bufferSize, mc2Context)
    │       │
    │       ├─► HcclGetRankId(comm, &rankId)   // Obtain the current rank
    │       │
    │       └─► for (remoteRankId = 0; remoteRankId < worldSize; remoteRankId++)
    │               │
    │               ├─► rankId == remoteRankId:
    │               │   HcclGetHcclBuffer(comm, &addr, &size)     // Local buffer
    │               │
    │               └─► rankId != remoteRankId:
    │                   HcclGetRemoteIpcHcclBuf(comm, remoteRankId, &addr, &size)  // Remote IPC buffer
    │               │
    │               └─► mc2Context.epHcclBuffer[remoteRankId] = addr  // Store it in the struct
    │
    ├─► HcclKfcFreeOpArgs(opArgs)             // 3. Release the communication configuration object
    │
    ▼
at::Tensor output = copy mc2ContextHost to NPU
return (output, bufferSize)
```

### Two-Stage Function Description

The creation flow consists of two core functions:

#### CreateHcclContext - Establishing the HCCL Communication Framework

**Responsibility**: creates the HCCL operation resource context and establishes the communication framework.

| API | Function |
|-----|------|
| HcclKfcOpArgsSetAlgConfig | Sets the All-to-All algorithm configuration |
| HcclCommGetHandleWithName | Obtains the communication handle based on the group name |
| HcclCreateOpResCtx | Creates the operation resource context |
| HcclGetRankSize / HcclGetRankId | Obtain the world size and rank ID, and validate parameters |

**Output**: `HcclComm commHandle` - a valid communication handle

#### CreatMc2Context - Filling In the Concrete Communication Addresses

**Responsibility**: fills in the Mc2ContextStru struct based on the established communication framework.

| API | Function |
|-----|------|
| HcclGetRankId | Obtains the current rank ID → `mc2Context.epRankId` |
| HcclGetHcclBuffer | Obtains the local buffer address and size |
| HcclGetRemoteIpcHcclBuf | Obtains the IPC buffer address of a remote rank |

**Output**: the complete `Mc2ContextStru`, which contains the buffer addresses of all ranks

#### Logical Relationship

- **Sequential dependency**: `CreateHcclContext` runs first to establish the communication framework and produce `commHandle`; `CreatMc2Context` runs afterwards and depends on `commHandle` to query the buffer addresses.

## Caching Mechanism

```cpp
static bool isInit = false;
static std::string last_group_ep_str = "";

// If the group is unchanged, return the cached result directly to accommodate repeated invocations
if (isInit && new_group_ep_str == last_group_ep_str) {
    return std::make_tuple(output, bufferSize);
}
```

## Functions of the Context

1. **Pre-fetching communication addresses**: obtains the HCCL buffer addresses of all ranks at once during initialization, avoiding repeated HCCL API queries at each kernel invocation.
2. **Zero-copy support**: the kernel directly accesses the buffer address of the target rank through `epHcclBuffer[rank_id]`.
3. **Communication resource caching**: when the same group is invoked repeatedly, the already created context is returned directly.
