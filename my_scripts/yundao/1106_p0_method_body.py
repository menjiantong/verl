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

