# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch

from verl.utils.fsdp_utils import fsdp2_load_full_state_dict

from ..base import EngineRegistry
from .transformer_impl import FSDPEngineWithLMHead


@EngineRegistry.register(model_type="language_model", backend="fsdp_turbo", device=["cuda", "npu"])
class FSDPTurboEngineWithLMHead(FSDPEngineWithLMHead):
    def _init_device_mesh(self):
        super()._init_device_mesh()
        self._init_parallel_state()

    def _init_parallel_state(self):
        from fsdp_turbo.distributed.parallel_state import get_parallel_state, init_parallel_state
        from fsdp_turbo.fsdp_turbo_config import FSDPTurboConfig, _dict_to_dataclass

        self.fsdp_turbo_config = _dict_to_dataclass(FSDPTurboConfig, self.engine_config.turbo_config)
        self.fsdp_turbo_config.distributed.fsdp_plan.cpu_offload = self.engine_config.offload_policy
        init_parallel_state(self.fsdp_turbo_config)
        self._parallel_state = get_parallel_state()
        if self._is_ulysses_enabled():
            self._process_ulysses_config()

    def _build_module(self):
        # Do not let verl's Qwen VLM monkey patch slice the text model before
        # FSDP-Turbo's post-fusion model patch does the CP split.
        cp_size = self.ulysses_sequence_parallel_size
        self.ulysses_sequence_parallel_size = 1
        try:
            return super()._build_module()
        finally:
            self.ulysses_sequence_parallel_size = cp_size

    def _build_fsdp_module(self, module):
        from fsdp_turbo.fsdp_turbo import FSDPTurbo

        full_state = module.state_dict()
        module = FSDPTurbo(self.fsdp_turbo_config, module).model
        offload_policy = None
        if self.engine_config.offload_policy or self.engine_config.forward_only:
            self._is_offload_param = False
            self._is_offload_optimizer = False
            offload_policy = True
            self._uses_fsdp2_cpu_offload_policy = True
        fsdp2_load_full_state_dict(module, full_state, None, offload_policy)

        return module

    def _is_ulysses_enabled(self):
        return self._parallel_state.is_group_enable("ulysses")

    def _process_ulysses_config(self):
        if self.ulysses_sequence_parallel_size > 1:
            raise ValueError(
                "Do not enable both FSDP-Turbo CP and verl Ulysses SP. "
                "Use fsdp_kwargs.distributed.ulysses_parallel_size for Turbo CP "
                "and set ulysses_sequence_parallel_size=1."
            )

        self.model_config.hf_config._attn_implementation = "eager"
        self.ulysses_sequence_parallel_size = self._parallel_state.get_ulysses_group_size()
        self.ulysses_parallel_group = self._parallel_state.get_ulysses_group()
        self.use_ulysses_sp = True

    def get_data_parallel_rank(self):
        if not hasattr(self, "_parallel_state"):
            return super().get_data_parallel_rank()
        return self._parallel_state.get_data_rank()

    def get_data_parallel_size(self):
        if not hasattr(self, "_parallel_state"):
            return super().get_data_parallel_size()
        return self._parallel_state.get_data_group_size()

    def get_data_parallel_group(self):
        if not hasattr(self, "_parallel_state"):
            return super().get_data_parallel_group()
        return self._parallel_state.get_data_group()

    def is_mp_src_rank_with_outputs(self):
        if not hasattr(self, "_parallel_state"):
            return super().is_mp_src_rank_with_outputs()
        if self._is_ulysses_enabled():
            is_collect = self._parallel_state.get_ulysses_rank() == 0
        else:
            is_collect = True
        # [1102 CP] KV-AllGather CP 下，CP 组内每个 rank 拿同一份 micro-batch
        # （dispatch 的 dp_rank 相同），且模型侧已把输出拼回整条序列，所以
        # collect 只能保留 CP rank 0，否则 16 个 worker 的输出会在
        # BatchData.concat 里拼出重复样本，KVBatchMeta 报 "Got duplicated keys"
        # （g1-5 实测：transferqueue_utils.py:333）。
        if self._parallel_state.is_group_enable("kvallgather"):
            is_collect = is_collect and self._parallel_state.get_kvallgather_rank() == 0
        return is_collect

    def optimizer_step(self):
        """
        Clip gradients, skip update if non-finite, and step optimizer.

        Returns:
            grad_norm (float): Norm of gradients before clipping.
        """
        assert self.optimizer_config.clip_grad is not None

        # getattr fallback: some subclasses (e.g. VeOmniEngine) bypass FSDPEngine.__init__.
        scaler = getattr(self, "scaler", None)

        # Unscale gradients before clip so the clip threshold is applied to true gradient
        # magnitudes, not scaled ones. scaler.step() will skip the update if any grad is inf/nan.
        if scaler is not None:
            scaler.unscale_(self.optimizer)

        from fsdp_turbo.training.clip_grads import clip_grad_norm

        grad_norm_value = clip_grad_norm(model=self.module, max_norm=self.optimizer_config.clip_grad)
        grad_norm = torch.tensor(grad_norm_value, device=next(self.module.parameters()).device, dtype=torch.float32)

        if scaler is not None:
            # scaler handles inf/nan skipping internally via _check_inf_per_device.
            scaler.step(self.optimizer)
            scaler.update()
        else:
            # if grad_norm is not finite, skip the update
            if not torch.isfinite(grad_norm):
                print(f"WARN: grad_norm is not finite: {grad_norm}")
                self.optimizer.zero_grad()
            else:
                self._dsv41_prepare_optimizer_step()
                self.optimizer.step()

        if self._qat_enabled:
            from verl.utils.qat.core import invalidate_all_scales

            invalidate_all_scales(self.module)

        return grad_norm.item()


    @staticmethod
    def _dsv41_probe_status() -> dict:
        """读 /proc/self/status 的 VmPin / VmRSS（MiB）。VmPin 是内核记账的锁页量。"""
        out = {}
        try:
            with open("/proc/self/status") as fh:
                for line in fh:
                    key = line.split(":", 1)[0]
                    if key in ("VmPin", "VmRSS", "VmSize"):
                        out[key] = int(line.split(":", 1)[1].split()[0]) // 1024
        except Exception:
            pass
        return out

    @staticmethod
    def _dsv41_probe_pinned(p) -> object:
        """DTensor 必须解包到 local 张量再判 is_pinned，否则永远拿到 False。"""
        try:
            t = p.to_local() if hasattr(p, "to_local") else p
            return bool(t.is_pinned())
        except Exception:
            return None

    def _dsv41_probe_optimizer_state(self):
        """[1106 P0-C3] 逐参数建 Adam 状态，同时记 wall / cpu / 带宽 / 锁页量。

        为什么需要：云道 E1 里这个循环单 rank 从 1 s 恶化到 1812 s，但老钩子只用
        time.time() 记 wall —— 分不清「在算」还是「在等」。本探针并列
        perf_counter(wall) 与 process_time(cpu)：
          wall ~= cpu  -> 真的在分配/清零，查 allocator / memset
          wall >> cpu  -> 在等（驱动锁 / cgroup 节流 / 缺页），zeros_like 无罪
        再配 MB/s：<10 MB/s 且 RSS 不涨就是「等」的可打印证据
        （141 MB 要 224 s = 0.6 MB/s，不可能是 memset）。

        反向思考：本探针会**替代** torch 的惰性建状态，因此 step/exp_avg/exp_avg_sq 的
        dtype 与 device 必须逐字沿用 torch/optim/adam.py 的口径，否则就是静默数值偏离。
        这也是它只在诊断时打开的原因。
        """
        import faulthandler
        import os
        import time

        opt = self.optimizer
        groups = opt.param_groups
        total = sum(len(g["params"]) for g in groups)
        dump_sec = float(os.environ.get("MY_ENV_DSV41_OPT_STATE_DUMP_SEC", "30"))
        dump_dir = os.environ.get("MY_ENV_DSV41_OPT_STATE_PROBE_DIR") or os.environ.get(
            "VERL_DSV41_OPT_STATE_DEBUG_DIR"
        )
        try:
            import torch.distributed as _dist

            _rank = _dist.get_rank() if _dist.is_initialized() else -1
        except Exception:
            _rank = -1

        def _log(msg):
            line = f"[rt-probe][rank{_rank}] {msg}"
            print(line, flush=True)
            if dump_dir:
                try:
                    os.makedirs(dump_dir, exist_ok=True)
                    with open(os.path.join(dump_dir, f"rtprobe_rank{_rank}_pid{os.getpid()}.log"), "a") as fh:
                        fh.write(f"{time.strftime('%H:%M:%S')} {msg}\n")
                except Exception:
                    pass

        st0 = self._dsv41_probe_status()
        _log(
            f"start optimizer={type(opt).__name__} groups={len(groups)} params={total} "
            f"VmPin={st0.get('VmPin')}MiB VmRSS={st0.get('VmRSS')}MiB"
        )
        t_start_wall = time.perf_counter()
        t_start_cpu = time.process_time()
        created = 0
        skipped_nograd = 0
        buckets = {"<0.01": 0, "0.01-0.1": 0, "0.1-1": 0, "1-5": 0, "5-30": 0, "30-120": 0, ">=120": 0}
        slow = []

        with torch.no_grad():
            for gi, group in enumerate(groups):
                for p in group["params"]:
                    if p.grad is None:
                        skipped_nograd += 1
                        continue
                    st = opt.state[p]
                    if len(st) != 0:
                        continue
                    t_wall0 = time.perf_counter()
                    t_cpu0 = time.process_time()
                    if group.get("capturable", False) or group.get("fused", False):
                        st["step"] = torch.zeros((), dtype=torch.float, device=p.device)
                    else:
                        st["step"] = torch.tensor(0.0)
                    st["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    st["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if group.get("amsgrad", False):
                        st["max_exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    t_wall = time.perf_counter() - t_wall0
                    t_cpu = time.process_time() - t_cpu0
                    created += 1

                    nbytes = p.numel() * p.element_size() * 2  # exp_avg + exp_avg_sq
                    mb = nbytes / (1024 * 1024)
                    mbps = (mb / t_wall) if t_wall > 0 else float("inf")
                    if t_wall < 0.01:
                        buckets["<0.01"] += 1
                    elif t_wall < 0.1:
                        buckets["0.01-0.1"] += 1
                    elif t_wall < 1:
                        buckets["0.1-1"] += 1
                    elif t_wall < 5:
                        buckets["1-5"] += 1
                    elif t_wall < 30:
                        buckets["5-30"] += 1
                    elif t_wall < 120:
                        buckets["30-120"] += 1
                    else:
                        buckets[">=120"] += 1

                    if t_wall > 5.0:
                        st1 = self._dsv41_probe_status()
                        slow.append(
                            {"i": created, "wall": round(t_wall, 2), "cpu": round(t_cpu, 2), "mbps": round(mbps, 2)}
                        )
                        _log(
                            f"SLOW i={created}/{total} wall={t_wall:.2f}s cpu={t_cpu:.2f}s "
                            f"mb={mb:.1f} mbps={mbps:.2f} pinned={self._dsv41_probe_pinned(p)} "
                            f"VmPin={st1.get('VmPin')}MiB shape={tuple(p.shape)} {p.dtype} {type(p).__name__}"
                        )
                    elif created % 25 == 0:
                        _log(
                            f"i={created}/{total} wall={t_wall:.4f}s cpu={t_cpu:.4f}s "
                            f"mb={mb:.1f} mbps={mbps:.1f} pinned={self._dsv41_probe_pinned(p)}"
                        )

                    # 单参数卡顿超阈值：把当前所有线程的 Python 栈落盘（不改变任何行为）
                    if t_wall > dump_sec:
                        try:
                            if dump_dir:
                                os.makedirs(dump_dir, exist_ok=True)
                                path = os.path.join(
                                    dump_dir, f"rank{_rank}_pid{os.getpid()}_stall_{int(time.time())}.txt"
                                )
                                with open(path, "a") as fh:
                                    fh.write(f"=== wall={t_wall:.1f}s i={created} shape={tuple(p.shape)} ===\n")
                                    faulthandler.dump_traceback(file=fh, all_threads=True)
                                _log(f"STACK_DUMP path={path} wall={t_wall:.2f}s")
                            else:
                                faulthandler.dump_traceback(all_threads=True)
                        except Exception as exc:  # noqa: BLE001
                            _log(f"STACK_DUMP_FAILED {type(exc).__name__}: {exc}")

        t_end_wall = time.perf_counter() - t_start_wall
        t_end_cpu = time.process_time() - t_start_cpu
        st1 = self._dsv41_probe_status()
        _log(
            f"done created={created}/{total} skipped_nograd={skipped_nograd} "
            f"wall={t_end_wall:.2f}s cpu={t_end_cpu:.2f}s "
            f"ratio={(t_end_cpu / t_end_wall if t_end_wall else 0):.3f} "
            f"VmPin={st1.get('VmPin')}MiB VmRSS={st1.get('VmRSS')}MiB buckets={buckets} slow={slow[:10]}"
        )

    def _dsv41_prepare_optimizer_step(self):
        """[dsv41 2026-10-17] E38 诊断/规避钩子：task54 首步 Adam 惰性建状态卡死。

        证据（worklog E38）：卡在 Adam._init_group 的
        ``state["exp_avg"] = torch.zeros_like(p)`` 内 ~3h，期间设备/主机零分配、
        零映射增长、futex 短睡 + ~40-50% CPU = 死等而非慢速推进。

        VERL_DSV41_OPT_STATE_DEBUG=1:
          打印 param group 概况 + 每个参数建状态（step/exp_avg/exp_avg_sq，完全
          复刻 torch Adam 惰性初始化口径）的进度行（每 25 个或单次 >5s 打一条），
          另有 30s 看门狗线程报告"当前卡在哪个参数的哪个形状"。预建状态后
          torch 的 _init_group 会因 len(state)!=0 直接跳过（数值口径不变）。
        VERL_DSV41_OPT_STATE_DEBUG_DIR:
          同时把进度行落到该目录下 rank_<pid>.log（防 Ray 日志聚合延迟）。
        VERL_DSV41_EMPTY_CACHE_BEFORE_OPT=1:
          step 前 empty_cache + gc（对齐 M27/M28 思路，整理 allocator 状态）。
        """

        import gc
        import os
        import threading
        import time

        if os.environ.get("VERL_DSV41_EMPTY_CACHE_BEFORE_OPT", "0") == "1":
            try:
                torch.npu.empty_cache()
            except Exception:
                pass
            gc.collect()

        # [1106 P0-C3] 新逐参数探针：wall/cpu/MB/MBps/VmPin，长卡顿自动落栈。
        # 默认关闭；打开时由它代替 torch 的惰性建状态（口径与 torch 完全一致），
        # 下面那个老 DEBUG 钩子会因为 len(state)!=0 自然跳过。
        if os.environ.get("MY_ENV_DSV41_OPT_STATE_PROBE", "0") == "1":
            self._dsv41_probe_optimizer_state()

        if os.environ.get("VERL_DSV41_OPT_STATE_DEBUG", "0") != "1":
            return

        opt = self.optimizer
        groups = opt.param_groups
        total = sum(len(g["params"]) for g in groups)

        dbg_dir = os.environ.get("VERL_DSV41_OPT_STATE_DEBUG_DIR")
        try:
            import torch.distributed as _dist

            _rank = _dist.get_rank() if _dist.is_initialized() else -1
        except Exception:
            _rank = -1

        def _log(msg):
            line = f"[opt-state-debug][rank{_rank}] {msg}"
            print(line, flush=True)
            if dbg_dir:
                try:
                    os.makedirs(dbg_dir, exist_ok=True)
                    with open(os.path.join(dbg_dir, f"rank{_rank}_pid{os.getpid()}.log"), "a") as fh:
                        fh.write(f"{time.strftime('%H:%M:%S')} {msg}\n")
                except Exception:
                    pass

        _log(f"optimizer={type(opt).__name__} groups={len(groups)} params={total}")
        if groups and groups[0]["params"]:
            p0 = groups[0]["params"][0]
            try:
                pinned = p0.is_pinned()
            except Exception:
                pinned = None
            _log(
                f"param0 type={type(p0).__name__} shape={tuple(p0.shape)} dtype={p0.dtype} "
                f"device={p0.device} grad={'None' if p0.grad is None else tuple(p0.grad.shape)} "
                f"is_pinned={pinned} local_type={type(getattr(p0, '_local_tensor', None)).__name__}"
            )

        prog = {"i": 0, "t0": time.time(), "cur": None, "last": -1}
        stop = threading.Event()

        def _watch():
            while not stop.wait(30):
                stuck = "  <=== 30s 无进展" if prog["i"] == prog["last"] else ""
                _log(f"[watch] created={prog['i']}/{total} cur={prog['cur']} elapsed={time.time() - prog['t0']:.0f}s{stuck}")
                prog["last"] = prog["i"]

        threading.Thread(target=_watch, daemon=True).start()
        try:
            with torch.no_grad():
                for gi, group in enumerate(groups):
                    for p in group["params"]:
                        if p.grad is None:
                            continue
                        st = opt.state[p]
                        if len(st) != 0:
                            continue
                        prog["cur"] = f"g{gi} {type(p).__name__}{tuple(p.shape)} {p.dtype} {p.device}"
                        t0 = time.time()
                        if group.get("capturable", False) or group.get("fused", False):
                            st["step"] = torch.zeros((), dtype=torch.float, device=p.device)
                        else:
                            st["step"] = torch.tensor(0.0)
                        st["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        st["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        if group.get("amsgrad", False):
                            st["max_exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        prog["i"] += 1
                        dt = time.time() - t0
                        if prog["i"] % 25 == 0 or dt > 5.0:
                            _log(f"created={prog['i']}/{total} last_dt={dt:.2f}s cur={prog['cur']}")
        finally:
            stop.set()
        _log(f"done created={prog['i']}/{total} elapsed={time.time() - prog['t0']:.0f}s")
