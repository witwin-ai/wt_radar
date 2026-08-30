"""Bound host-side result allocations; never silently reduce a simulation."""
import ctypes
import os

MAX_RESULT_BYTES = 1536 * 1024**2


def available_memory_bytes():
    if os.name == "nt":
        class MemoryStatus(ctypes.Structure):
            _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong),
                        *[(name, ctypes.c_ulonglong) for name in
                          ("total", "available", "page_total", "page_available", "virtual_total", "virtual_available", "extended")]]
        status = MemoryStatus()
        status.length = ctypes.sizeof(status)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            raise RuntimeError("Cannot check free host memory for this result.")
        return status.available
    return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")


def require_result_memory(size):
    if size > MAX_RESULT_BYTES:
        raise ValueError("Animation exceeds 1536 MiB raw-cube budget; no settings changed.")
    # Leave room for native scratch, processing and compressed export. The cube
    # is preallocated once, not retained as a list plus a second stacked copy.
    needed = max(512 * 1024**2, 3 * size)
    if available_memory_bytes() < needed:
        raise ValueError(f"Insufficient free host memory: need {needed / 1024**3:.2f} GiB headroom; no settings changed.")
