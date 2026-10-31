# MY_ENV_OP_DUMP hook loader.  This directory is prepended to PYTHONPATH only
# for the diagnostic job; production scripts never import it.
#
# Do NOT import torch here: vLLM's worker startup sets OMP/thread state before
# importing torch, and importing it from sitecustomize makes the forked worker
# fail in c10::set_num_threads.  Instead, install the hook after the
# application's own first ``import torch`` has completed.
import builtins
import os
import sys

if os.environ.get("MY_ENV_OP_DUMP") == "1":
    _orig_import = builtins.__import__
    _installed = False

    def _install_once():
        global _installed
        if _installed:
            return
        _installed = True
        try:
            import my_op_dump_hook

            my_op_dump_hook.install()
        except Exception:  # noqa: BLE001 - never break the worker on a debug hook
            import traceback

            print("[MY_DEBUG_LOG opdump] sitecustomize install failed", file=sys.stderr, flush=True)
            traceback.print_exc()

    def _import_hook(name, globals=None, locals=None, fromlist=(), level=0):
        module = _orig_import(name, globals, locals, fromlist, level)
        if (
            not _installed
            and level == 0
            and (name == "torch" or name.startswith("torch."))
        ):
            torch_mod = sys.modules.get("torch")
            # During torch's own initialization ``Tensor`` is not defined yet;
            # wait until the top-level module is complete.
            if torch_mod is not None and hasattr(torch_mod, "Tensor") and hasattr(torch_mod, "nn"):
                _install_once()
        return module

    builtins.__import__ = _import_hook
