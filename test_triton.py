import os
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

from torch.utils._triton import has_triton

import _ishmem_triton as ishmem_triton


HAS_TRITON_API_NAMES = [
    "GridCallableWithExtern",
    "put",
    "putmem_block_extern_wrapper",
    "get",
    "getmem_block_extern_wrapper",
    "get_nbi",
    "getmem_nbi_block_extern_wrapper",
    "putmem_signal_block",
    "putmem_signal_block_extern_wrapper",
    "wait_until",
    "wait_until_extern_wrapper",
    "signal_wait_until",
    "signal_wait_until_extern_wrapper",
    "signal_op",
    "fence",
    "quiet",
    "my_pe",
    "n_pes",
    "barrier_all",
    "sync_all",
    "alltoall",
    "alltoallmem_block_extern_wrapper",
    "broadcast",
    "broadcastmem_block_extern_wrapper",
    "reduce",
    "reduce_extern_wrapper",
]


IMPORTED_HAS_TRITON_APIS = {}


if has_triton():
    import triton
    import triton.language as tl

    from _ishmem_triton import (
        GridCallableWithExtern,
        alltoall,
        alltoallmem_block_extern_wrapper,
        barrier_all,
        broadcast,
        broadcastmem_block_extern_wrapper,
        fence,
        get,
        get_nbi,
        getmem_block_extern_wrapper,
        getmem_nbi_block_extern_wrapper,
        my_pe,
        n_pes,
        put,
        putmem_block_extern_wrapper,
        putmem_signal_block,
        putmem_signal_block_extern_wrapper,
        quiet,
        reduce,
        reduce_extern_wrapper,
        signal_op,
        signal_wait_until,
        signal_wait_until_extern_wrapper,
        sync_all,
        wait_until,
        wait_until_extern_wrapper,
    )

    IMPORTED_HAS_TRITON_APIS = {
        "GridCallableWithExtern": GridCallableWithExtern,
        "put": put,
        "putmem_block_extern_wrapper": putmem_block_extern_wrapper,
        "get": get,
        "getmem_block_extern_wrapper": getmem_block_extern_wrapper,
        "get_nbi": get_nbi,
        "getmem_nbi_block_extern_wrapper": getmem_nbi_block_extern_wrapper,
        "putmem_signal_block": putmem_signal_block,
        "putmem_signal_block_extern_wrapper": putmem_signal_block_extern_wrapper,
        "wait_until": wait_until,
        "wait_until_extern_wrapper": wait_until_extern_wrapper,
        "signal_wait_until": signal_wait_until,
        "signal_wait_until_extern_wrapper": signal_wait_until_extern_wrapper,
        "signal_op": signal_op,
        "fence": fence,
        "quiet": quiet,
        "my_pe": my_pe,
        "n_pes": n_pes,
        "barrier_all": barrier_all,
        "sync_all": sync_all,
        "alltoall": alltoall,
        "alltoallmem_block_extern_wrapper": alltoallmem_block_extern_wrapper,
        "broadcast": broadcast,
        "broadcastmem_block_extern_wrapper": broadcastmem_block_extern_wrapper,
        "reduce": reduce,
        "reduce_extern_wrapper": reduce_extern_wrapper,
    }

    @triton.jit
    def _kernel_put(dest, src, nelems, pe):
        ishmem_triton.put(dest, src, nelems, pe)


    @triton.jit
    def _kernel_get(dest, src, nelems, pe, nbi: tl.constexpr):
        if nbi:
            ishmem_triton.get_nbi(dest, src, nelems, pe)
            ishmem_triton.quiet()
        else:
            ishmem_triton.get(dest, src, nelems, pe)


    @triton.jit
    def _kernel_putmem_signal_block(dst, src, size_bytes, signal, sig_val, sig_op, pe):
        ishmem_triton.putmem_signal_block(dst, src, size_bytes, signal, sig_val, sig_op, pe)


    @triton.jit
    def _kernel_wait_until(ivar, cmp_op, cmp_val):
        ishmem_triton.wait_until(ivar, cmp_op, cmp_val)


    @triton.jit
    def _kernel_signal_wait_until(signal, cmp_op, cmp_val):
        ishmem_triton.signal_wait_until(signal, cmp_op, cmp_val)


    @triton.jit
    def _kernel_signal_op(sig_addr, signal, sig_op, pe):
        ishmem_triton.signal_op(sig_addr, signal, sig_op, pe)


    @triton.jit
    def _kernel_fence():
        ishmem_triton.fence()


    @triton.jit
    def _kernel_quiet():
        ishmem_triton.quiet()


    @triton.jit
    def _kernel_pe_info_and_barrier():
        pe = ishmem_triton.my_pe()
        npes = ishmem_triton.n_pes()
        if pe < npes:
            ishmem_triton.barrier_all()


    @triton.jit
    def _kernel_sync_all():
        ishmem_triton.sync_all()


    @triton.jit
    def _kernel_alltoall(team, dest, src, nelems_per_pe):
        ishmem_triton.alltoall(team, dest, src, nelems_per_pe)


    @triton.jit
    def _kernel_broadcast(team, dest, src, nelems, pe_root):
        ishmem_triton.broadcast(team, dest, src, nelems, pe_root)


    @triton.jit
    def _kernel_reduce(team, dest, src, nreduce, operation: tl.constexpr):
        ishmem_triton.reduce(team, dest, src, nreduce, operation)


def test_has_triton_api_exports() -> None:
    if not has_triton():
        return

    missing = [name for name in HAS_TRITON_API_NAMES if not hasattr(ishmem_triton, name)]
    assert not missing, f"Missing has_triton APIs in _ishmem_triton: {missing}"

    missing_imports = [
        name for name in HAS_TRITON_API_NAMES if name not in IMPORTED_HAS_TRITON_APIS
    ]
    assert not missing_imports, f"Missing imports in test_triton.py: {missing_imports}"

    mismatched = [
        name
        for name in HAS_TRITON_API_NAMES
        if IMPORTED_HAS_TRITON_APIS[name] is not getattr(ishmem_triton, name)
    ]
    assert not mismatched, f"Imported APIs do not match module exports: {mismatched}"


def test_nvshmem_kernel_registry_round_trip() -> None:
    registry = ishmem_triton.NvshmemKernelRegistry
    previous = dict(registry._to_init)

    try:
        registry._to_init = {}
        registry.register("kernel_a")
        assert registry.has("kernel_a")

        registry.register("kernel_a")
        assert registry.has("kernel_a")

        registry.deregister("kernel_a")
        assert not registry.has("kernel_a")
    finally:
        registry._to_init = previous


def test_find_device_library_uses_env_var() -> None:
    finder = ishmem_triton.NvshmemLibFinder
    previous = finder.found_device_lib_path

    try:
        finder.found_device_lib_path = None
        with tempfile.TemporaryDirectory() as tmpdir:
            lib_path = os.path.join(tmpdir, "libnvshmem_device.bc")
            with open(lib_path, "w", encoding="utf-8") as handle:
                handle.write("test")

            with patch.dict(os.environ, {"NVSHMEM_LIB_DIR": tmpdir}, clear=False):
                found = finder.find_device_library()
                assert found == lib_path
                assert finder.found_device_lib_path == lib_path
    finally:
        finder.found_device_lib_path = previous


def test_find_device_library_raises_for_missing_env_target() -> None:
    finder = ishmem_triton.NvshmemLibFinder
    previous = finder.found_device_lib_path

    try:
        finder.found_device_lib_path = None
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"NVSHMEM_LIB_DIR": tmpdir}, clear=False):
                try:
                    finder.find_device_library()
                except RuntimeError as exc:
                    assert "NVSHMEM device library not found at specified path" in str(exc)
                else:
                    assert False, "Expected RuntimeError for missing NVSHMEM device library"
    finally:
        finder.found_device_lib_path = previous


def test_grid_callable_with_extern_forwards_extern_libs() -> None:
    if not has_triton():
        return

    calls = []

    class FakeJitFunc:
        def run(self, *args, **kwargs):
            calls.append((args, kwargs))
            return "ok"

    extern_libs = {"libnvshmem_device": "/tmp/libnvshmem_device.bc"}
    wrapper = ishmem_triton.GridCallableWithExtern(FakeJitFunc(), extern_libs)

    result = wrapper.run(1, 2, keyword="value")

    assert result == "ok"
    assert calls == [
        (
            (1, 2),
            {"keyword": "value", "extern_libs": extern_libs},
        )
    ]


def test_requires_nvshmem_rejects_non_jitfunction() -> None:
    if not has_triton():
        return

    import triton

    class FakeJITFunction:
        pass

    with patch.object(triton.runtime.jit, "JITFunction", FakeJITFunction):
        try:
            ishmem_triton.requires_nvshmem(object())
        except TypeError as exc:
            assert "Expected a JITFunction" in str(exc)
        else:
            assert False, "Expected TypeError for non-JITFunction input"


def test_requires_nvshmem_registers_kernel_and_sets_hook() -> None:
    if not has_triton():
        return

    import triton

    class FakeJITFunction:
        def __init__(self):
            self.fn = SimpleNamespace(__name__="fake_kernel")
            self.run_calls = []

        def run(self, *args, **kwargs):
            self.run_calls.append((args, kwargs))
            return "wrapped"

    fake_jit = FakeJITFunction()
    previous_hook = triton.knobs.runtime.jit_post_compile_hook

    try:
        with (
            patch.object(triton.runtime.jit, "JITFunction", FakeJITFunction),
            patch.object(
                ishmem_triton.NvshmemLibFinder,
                "find_device_library",
                return_value="/tmp/libnvshmem_device.bc",
            ),
            patch.object(ishmem_triton.NvshmemKernelRegistry, "register") as register_mock,
        ):
            triton.knobs.runtime.jit_post_compile_hook = None
            wrapped = ishmem_triton.requires_nvshmem(fake_jit)

            assert isinstance(wrapped, ishmem_triton.GridCallableWithExtern)
            register_mock.assert_called_once_with("fake_kernel")
            assert triton.knobs.runtime.jit_post_compile_hook is ishmem_triton._nvshmem_init_hook

            result = wrapped.run("arg", flag=True)
            assert result == "wrapped"
            assert fake_jit.run_calls == [
                (
                    ("arg",),
                    {
                        "flag": True,
                        "extern_libs": {
                            "libnvshmem_device": "/tmp/libnvshmem_device.bc"
                        },
                    },
                )
            ]
    finally:
        triton.knobs.runtime.jit_post_compile_hook = previous_hook


def test_put_kernel_calls_put_api() -> None:
    if not has_triton():
        return

    with patch.object(ishmem_triton, "put") as mock_put:
        _kernel_put.fn("dest", "src", 6, 3)

    mock_put.assert_called_once_with("dest", "src", 6, 3)


def test_get_kernel_calls_blocking_get_api() -> None:
    if not has_triton():
        return

    with (
        patch.object(ishmem_triton, "get") as mock_get,
        patch.object(ishmem_triton, "get_nbi") as mock_get_nbi,
        patch.object(ishmem_triton, "quiet") as mock_quiet,
    ):
        _kernel_get.fn("dest", "src", 9, 1, False)

    mock_get.assert_called_once_with("dest", "src", 9, 1)
    mock_get_nbi.assert_not_called()
    mock_quiet.assert_not_called()


def test_get_kernel_calls_get_nbi_and_quiet_for_nonblocking_path() -> None:
    if not has_triton():
        return

    with (
        patch.object(ishmem_triton, "get") as mock_get,
        patch.object(ishmem_triton, "get_nbi") as mock_get_nbi,
        patch.object(ishmem_triton, "quiet") as mock_quiet,
    ):
        _kernel_get.fn("dest", "src", 4, 2, True)

    mock_get.assert_not_called()
    mock_get_nbi.assert_called_once_with("dest", "src", 4, 2)
    mock_quiet.assert_called_once_with()


def test_putmem_signal_block_kernel_calls_api() -> None:
    if not has_triton():
        return

    with patch.object(ishmem_triton, "putmem_signal_block") as mock_signal:
        _kernel_putmem_signal_block.fn("dst", "src", 128, "signal", 7, 5, 9)

    mock_signal.assert_called_once_with(
        "dst",
        "src",
        128,
        "signal",
        7,
        5,
        9,
    )


def test_wait_until_kernel_calls_api() -> None:
    if not has_triton():
        return

    with patch.object(ishmem_triton, "wait_until") as mock_wait:
        _kernel_wait_until.fn("ivar", 0, 11)

    mock_wait.assert_called_once_with("ivar", 0, 11)


def test_signal_wait_until_kernel_calls_api() -> None:
    if not has_triton():
        return

    with patch.object(ishmem_triton, "signal_wait_until") as mock_signal_wait:
        _kernel_signal_wait_until.fn("signal", 3, 42)

    mock_signal_wait.assert_called_once_with("signal", 3, 42)


def test_signal_op_kernel_calls_api() -> None:
    if not has_triton():
        return

    with patch.object(ishmem_triton, "signal_op") as mock_signal_op:
        _kernel_signal_op.fn("sig_addr", 1, 5, 2)

    mock_signal_op.assert_called_once_with("sig_addr", 1, 5, 2)


def test_fence_kernel_calls_api() -> None:
    if not has_triton():
        return

    with patch.object(ishmem_triton, "fence") as mock_fence:
        _kernel_fence.fn()

    mock_fence.assert_called_once_with()


def test_quiet_kernel_calls_api() -> None:
    if not has_triton():
        return

    with patch.object(ishmem_triton, "quiet") as mock_quiet:
        _kernel_quiet.fn()

    mock_quiet.assert_called_once_with()


def test_pe_info_kernel_calls_my_pe_n_pes_and_barrier_all() -> None:
    if not has_triton():
        return

    with (
        patch.object(ishmem_triton, "my_pe", return_value=1) as mock_my_pe,
        patch.object(ishmem_triton, "n_pes", return_value=4) as mock_n_pes,
        patch.object(ishmem_triton, "barrier_all") as mock_barrier,
    ):
        _kernel_pe_info_and_barrier.fn()

    mock_my_pe.assert_called_once_with()
    mock_n_pes.assert_called_once_with()
    mock_barrier.assert_called_once_with()


def test_sync_all_kernel_calls_api() -> None:
    if not has_triton():
        return

    with patch.object(ishmem_triton, "sync_all") as mock_sync_all:
        _kernel_sync_all.fn()

    mock_sync_all.assert_called_once_with()


def test_alltoall_kernel_calls_api() -> None:
    if not has_triton():
        return

    with patch.object(ishmem_triton, "alltoall") as mock_alltoall:
        _kernel_alltoall.fn(0, "dest", "src", 5)

    mock_alltoall.assert_called_once_with(0, "dest", "src", 5)


def test_broadcast_kernel_calls_api() -> None:
    if not has_triton():
        return

    with patch.object(ishmem_triton, "broadcast") as mock_broadcast:
        _kernel_broadcast.fn(1, "dest", "src", 16, 7)

    mock_broadcast.assert_called_once_with(1, "dest", "src", 16, 7)


def test_reduce_kernel_calls_api() -> None:
    if not has_triton():
        return

    with patch.object(ishmem_triton, "reduce") as mock_reduce:
        _kernel_reduce.fn(0, "dest", "src", 12, "sum")

    mock_reduce.assert_called_once_with(0, "dest", "src", 12, "sum")
