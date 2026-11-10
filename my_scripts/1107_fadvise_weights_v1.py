#!/usr/bin/env python3
# [启动结果] ⏳ 待验证：2026-11-06 首次创建（80.5.25.130）。
# =============================================================================
# 1107 对权重目录做 POSIX_FADV_DONTNEED，回收 checkpoint 读取留下的干净页缓存
#
# 背景（130 实测，2026-11-06）：
#   - with-ref（USE_KL=1）跑 engram 时，模型加载后 Cached≈1099 GiB（其中 Shmem 456 GiB，
#     文件页缓存 ~643 GiB），MemFree 只剩 60 GiB；
#   - vLLM 在 rollout 的 _gather_mm_embeddings 里做一个小 pinned 分配就失败：
#     aclrtMallocHostWithCfg 207001 / rtsMallocHost out of memory；
#   - 内核的 drop_caches=1 在本机约 20 GiB/min（NFS 页多），太慢；
#     fadvise 只针对权重文件，秒级完成，效果等价于云道 relief 档的 posix_fadvise。
#
# 反向思考：为什么不直接 drop_caches？
#   它是全局的、要遍历整机 LRU，实测会把下一档的启动拖 20+ 分钟；而且会顺手丢掉
#   与本次训练无关的页。fadvise 只标记权重文件自己的页，范围可控、可重复。
#
# 用法：
#   python3 1107_fadvise_weights_v1.py /mnt/share/m00899630/weights/DeepSeek-V4.1-Flash-top32-bf16
# =============================================================================
import os
import sys
import time


def main(root: str) -> int:
    t0 = time.time()
    files = 0
    failed = 0
    total = 0
    for dirpath, _, names in os.walk(root):
        for name in names:
            path = os.path.join(dirpath, name)
            try:
                fd = os.open(path, os.O_RDONLY)
                try:
                    st = os.fstat(fd)
                    # len=0 表示整个文件；只丢弃干净页，脏页/正在映射的页由内核自行保留
                    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                    total += st.st_size
                    files += 1
                finally:
                    os.close(fd)
            except OSError:
                failed += 1
    print(
        f"[1107 fadvise] MY_DEBUG_LOG fadvise_done root={root} files={files} failed={failed} "
        f"bytes={total / 1024**3:.1f}GiB elapsed={time.time() - t0:.1f}s",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <weights_dir>", file=sys.stderr)
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
