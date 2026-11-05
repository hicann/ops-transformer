"""NPU integration tests; run after installing the operator and Torch extension."""

import pytest

torch = pytest.importorskip("torch")
torch_npu = pytest.importorskip("torch_npu")
pytestmark = pytest.mark.skipif(
    not torch.npu.is_available(), reason="requires an Ascend NPU"
)


@pytest.fixture(scope="module")
def copy_op():
    # Do not skip missing registration on an available NPU: a missing build is a failure.
    from cann_ops_transformer.ops.attention.group_memcpy import group_memcpy

    return group_memcpy


DTYPES = [
    torch.int8,
    torch.uint8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.bool,
]


def raw_bytes(tensor):
    return tensor.reshape(-1).view(torch.uint8).cpu()


def pattern(elements, dtype, seed):
    if dtype == torch.bool:
        return (torch.arange(elements, dtype=torch.int64) + seed).remainder(2).to(dtype)
    item_bytes = torch.empty((), dtype=dtype).element_size()
    # Arbitrary bit patterns include floating point NaNs/infinities/signed zeros.
    raw = (
        (torch.arange(elements * item_bytes, dtype=torch.int64) * 73 + seed * 19) % 256
    ).to(torch.uint8)
    return raw.view(dtype)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    "elements", [0, 1, 15, 16, 17, 31, 32, 33, 63, 64, 65, 94207, 94208, 94209, 188417]
)
def test_exact_bytes_and_guards(copy_op, dtype, elements):
    src = [pattern(elements, dtype, i).to("npu") for i in range(3)]
    backing = [torch.full((elements + 4,), 1, dtype=dtype, device="npu") for _ in src]
    dst = [item[2 : 2 + elements] for item in backing]
    before = [raw_bytes(item) for item in backing]
    addresses = [item.data_ptr() for item in dst]
    assert copy_op(src, dst) is None
    torch.npu.synchronize()
    for i, (x, y, storage) in enumerate(zip(src, dst, backing)):
        assert y.data_ptr() == addresses[i]
        assert torch.equal(raw_bytes(x), raw_bytes(y))
        element_bytes = x.element_size()
        actual = raw_bytes(storage)
        assert torch.equal(actual[: 2 * element_bytes], before[i][: 2 * element_bytes])
        assert torch.equal(
            actual[(elements + 2) * element_bytes :],
            before[i][(elements + 2) * element_bytes :],
        )


@pytest.mark.parametrize("tasks", [1, 39, 40, 41, 81, 257])
def test_task_distribution_and_repeated_sources(copy_op, tasks):
    x = torch.arange(65, dtype=torch.int32, device="npu")
    dst = [torch.empty_like(x) for _ in range(tasks)]
    copy_op([x] * tasks, dst)
    torch.npu.synchronize()
    for item in dst:
        assert torch.equal(item.cpu(), x.cpu())


def test_views_same_storage_and_heterogeneous_shapes(copy_op):
    # Offsets and disjoint slices of a shared KV-like allocation are supported.
    src_storage = torch.arange(1024, dtype=torch.int32, device="npu")
    dst_storage = torch.full_like(src_storage, -1)
    src = [src_storage[1:65].view(8, 8), src_storage[101:165].view(1, 4, 4, 4)]
    dst = [dst_storage[3:67].view(8, 8), dst_storage[203:267].view(1, 4, 4, 4)]
    copy_op(src, dst)
    torch.npu.synchronize()
    expected = torch.full((1024,), -1, dtype=torch.int32)
    expected[3:67] = src_storage[1:65].cpu()
    expected[203:267] = src_storage[101:165].cpu()
    assert torch.equal(dst_storage.cpu(), expected)


def test_current_stream_and_lifetimes(copy_op):
    stream = torch.npu.Stream()
    done = torch.npu.Event()
    with torch.npu.stream(stream):
        src = [
            torch.empty((188417,), dtype=torch.uint8, device="npu") for _ in range(41)
        ]
        for i, tensor in enumerate(src):
            tensor.fill_(i + 1)
        dst = [torch.empty_like(tensor) for tensor in src]
        copy_op(src, dst)
        # Dropping inputs and reusing allocations stresses queued capture lifetimes.
        del src
        trash = [torch.full_like(dst[0], 255) for _ in range(64)]
        chained = [tensor.to(torch.int32).sum() for tensor in dst]
        done.record()
    done.synchronize()
    for i, result in enumerate(chained):
        assert result.item() == (i + 1) * 188417
    assert len(trash) == 64


def test_reject_invalid_inputs(copy_op):
    x = torch.empty((64,), device="npu")
    y = torch.empty_like(x)
    with pytest.raises(ValueError, match="nonzero list length"):
        copy_op([], [])
    with pytest.raises(ValueError, match="nonzero list length"):
        copy_op([x], [])
    bad_cases = [
        ([x], [y[:32]]),
        ([x], [y.to(torch.float16)]),
        ([x], [torch.empty((128,), device="npu")[::2]]),
        ([x], [torch.empty_like(x, device="cpu")]),
        ([x], [x]),
        ([x, x], [y, y]),
        ([x, y], [y, torch.empty_like(x)]),
        ([x.detach().clone().requires_grad_(True)], [y]),
        ([x], [y.view(8, 8)]),
        ([x], [y.detach().clone().requires_grad_(True)]),
    ]
    for inputs, outputs in bad_cases:
        with pytest.raises((RuntimeError, ValueError)):
            copy_op(inputs, outputs)


def test_reject_partial_overlaps(copy_op):
    storage = torch.empty((256,), device="npu")
    other = torch.empty((64,), device="npu")
    with pytest.raises(RuntimeError, match="overlap"):
        copy_op([storage[:64]], [storage[32:96]])
    with pytest.raises(RuntimeError, match="overlap"):
        copy_op([other, other], [storage[:64], storage[32:96]])


def test_mutation_version_and_saved_tensor_detection(copy_op):
    x = torch.arange(64, dtype=torch.float32, device="npu")
    dst = torch.ones_like(x)
    version = dst._version
    copy_op([x], [dst])
    assert dst._version > version
    torch.npu.synchronize()
    # A non-grad destination can still have been saved by another grad operation.
    leaf = torch.ones_like(dst, requires_grad=True)
    loss = (leaf * dst).sum()
    copy_op([x], [dst])
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        loss.backward()


def test_inference_mode(copy_op):
    with torch.inference_mode():
        x = torch.arange(65, dtype=torch.int32, device="npu")
        dst = torch.empty_like(x)
        copy_op([x], [dst])
        torch.npu.synchronize()
        assert torch.equal(x.cpu(), dst.cpu())


def test_reject_mixed_widths(copy_op):
    src = [torch.empty((16,), device="npu"), torch.empty((32,), device="npu")]
    with pytest.raises(RuntimeError, match="byte counts"):
        copy_op(src, [torch.empty_like(item) for item in src])


def test_meta_and_schema(copy_op):
    x = torch.empty((2, 32), device="meta", dtype=torch.float16)
    y = torch.empty_like(x)
    assert copy_op([x], [y]) is None
    assert y.shape == x.shape and y.device.type == "meta"
    schema = str(torch.ops.cann_ops_transformer.group_memcpy.default._schema)
    assert "Tensor(a!)[] dst" in schema
    unsupported = torch.empty((64,), device="meta", dtype=torch.float64)
    with pytest.raises(RuntimeError, match="unsupported dtype"):
        copy_op([unsupported], [torch.empty_like(unsupported)])


def test_nondefault_device_if_present(copy_op):
    if torch.npu.device_count() < 2:
        pytest.skip("requires two NPU devices")
    x = torch.arange(65, device="npu:1", dtype=torch.int32)
    y = torch.empty_like(x)
    with torch.npu.device(0):
        copy_op([x], [y])
    torch.npu.synchronize(1)
    assert torch.equal(y.cpu(), x.cpu())
    with pytest.raises(RuntimeError, match="same device"):
        copy_op([x], [torch.empty_like(x, device="npu:0")])
