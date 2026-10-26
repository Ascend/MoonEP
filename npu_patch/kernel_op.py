"""
KernelOp base: the ascend_deepep "op owns its symmetric workspace" pattern.

Mirrors ``ShmemWorkspaceOp`` in ``ascend_deepep/csrc/torch_classes.cpp``: the
op may allocate its own workspace or borrow a registered view of a Buffer
arena. Owned workspace is released at destroy; a borrowed arena remains
owned by the Buffer. Ops must be created and destroyed in the same order
on every rank.

Buffer-backed path only: the workspace is CANN VMM physical memory
(``create_vmm_physical``, registered with the buffer-backed shmem init); its
symmetric address is resolved lazily via ``translate_tensor`` in
``sym_workspace()``. Teardown of owned workspace reverses the VMM steps
(unmap -> release -> free_physical).
"""

import torch

from ._common import SymBuffer
from .buffer_c import create_vmm_physical, translate_tensor


class KernelOp:
    """Base for npu_patch kernel operators.

    The migrated AscendC kernels consume the local symmetric bases from ``ctx``
    (the communication chunks — ``hidden_buf`` / ``weights_buf`` /
    ``barrier_buf`` — addressed via their shmem addresses) plus this op's
    symmetric workspace; they must never assume a cross-rank contiguous VA.
    """

    def __init__(self, ws_bytes: int = 0, num_sms: int = 32, *,
                 ws_tensor: torch.Tensor | None = None):
        # No 2 MiB alignment requirement on Ascend: any positive size is OK
        # (VMM granularity alignment is handled by create_vmm_physical).
        self._num_sms = num_sms
        self._ws_bytes = ws_bytes
        self._ws_tensor = None
        self._ws = None
        self._ws_translated_tensor = None
        self._ws_allocation = None
        self._owns_workspace = False
        if ws_tensor is not None:
            if not isinstance(ws_tensor, torch.Tensor):
                raise TypeError("ws_tensor must be an NPU tensor")
            if ws_tensor.dtype != torch.uint8 or not ws_tensor.is_contiguous():
                raise ValueError("ws_tensor must be contiguous uint8")
            if ws_tensor.device.type != "npu":
                raise ValueError("ws_tensor must be on an NPU")
            if ws_tensor.device.index != torch.npu.current_device():
                raise ValueError("ws_tensor must be on the current NPU")
            tensor_bytes = int(ws_tensor.numel()) * int(ws_tensor.element_size())
            if ws_bytes <= 0 or tensor_bytes < ws_bytes:
                raise ValueError("ws_tensor must cover a positive ws_bytes range")
            allocation = getattr(ws_tensor, "_vmm_allocation", None)
            if allocation is None or getattr(allocation, "_destroyed", False):
                raise ValueError("ws_tensor must have a live VMM allocation")
            if not getattr(allocation, "_registered", False):
                raise ValueError("ws_tensor VMM allocation must be registered")
            if getattr(allocation, "device_id", None) != ws_tensor.device.index:
                raise ValueError("ws_tensor VMM allocation must be on the same NPU")
            source_offset = int(ws_tensor.data_ptr()) - int(allocation.source_va)
            offset = int(getattr(ws_tensor, "_vmm_offset_bytes", source_offset))
            view_bytes = int(getattr(ws_tensor, "_vmm_view_bytes", tensor_bytes))
            if offset != source_offset:
                raise ValueError("ws_tensor must be a source view with matching VMM offset")
            if (offset < 0 or view_bytes < tensor_bytes or
                    offset + view_bytes > int(allocation.size)):
                raise ValueError("ws_tensor view exceeds its VMM allocation")
            # Keep the shared allocation alive, without taking ownership or
            # registering a second descriptor for the same physical arena.
            self._ws_tensor = ws_tensor
            self._ws_allocation = allocation
        elif ws_bytes > 0:
            # VMM physical allocation; symmetric address resolved lazily in sym_workspace().
            self._ws_tensor, _, _ = create_vmm_physical(
                [ws_bytes], torch.uint8, torch.npu.current_device())
            self._ws_allocation = getattr(self._ws_tensor, '_vmm_allocation', None)
            self._owns_workspace = True
        # Collective: all ranks allocate the same workspaces in the same order.
        self._destroyed = False

    @property
    def num_sms(self) -> int:
        return self._num_sms

    def sym_workspace(self) -> SymBuffer | None:
        """The op's symmetric workspace (device-RMA scratch), if any.

        Returns the translated local shmem symmetric address (requires the
        buffer-backed shmem heap initialized and this op's workspace registered).
        """
        if self._ws_tensor is None:
            return None
        if self._ws_allocation is not None and getattr(self._ws_allocation, '_destroyed', False):
            raise RuntimeError("op workspace backing has been destroyed")
        if self._ws is None:
            # translate_tensor maps the registered arena base and adds the
            # source view's byte offset; an interior pointer is not itself a
            # SHMEM descriptor and must not be mapped as a separate buffer.
            self._ws_translated_tensor = translate_tensor(self._ws_tensor, 'op_workspace')
            self._ws = SymBuffer(self._ws_translated_tensor.data_ptr(), self._ws_bytes)
        return self._ws

    def destroy(self) -> None:
        """Collective: all ranks must destroy ops in the same order."""
        if not self._destroyed:
            if self._ws_tensor is not None:
                if self._owns_workspace and self._ws_allocation is not None:
                    # Reverse VMM steps (unmap -> release -> free_physical).
                    self._ws_allocation.destroy()
                self._ws_tensor = None
                self._ws = None
                self._ws_translated_tensor = None
                self._ws_allocation = None
            self._destroyed = True
