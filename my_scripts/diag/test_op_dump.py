"""Tiny CPU test for the sitecustomize op-dump hook."""
import os
import torch
import torch.nn as nn


class Block(nn.Module):
    __module__ = "fsdp_turbo.test"

    def __init__(self):
        super().__init__()
        self.layer_id = 3
        self.inner = Indexer()

    def forward(self, x):
        return self.inner(x) + 1


class Indexer(nn.Module):
    __module__ = "fsdp_turbo.test"

    def forward(self, x):
        return x * 2


class ParallelEmbedding(nn.Module):
    __module__ = "fsdp_turbo.test"

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(16, 4))

    def forward(self, input_ids):
        return self.weight[input_ids]


def main():
    emb = ParallelEmbedding()
    block = Block()
    ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.int64)
    x = emb(ids)
    y = block(x)
    print("out", tuple(y.shape), flush=True)
    print("dump_dir", os.environ.get("MY_ENV_OP_DUMP_DIR"), flush=True)


if __name__ == "__main__":
    main()
