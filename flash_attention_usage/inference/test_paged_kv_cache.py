import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

# Cache bookkeeping does not require model weights, FlashAttention, or a tokenizer.
with patch.dict("sys.modules", {
    "model_llama_kv_cache_paged": Mock(),
    "tokenizer_llama": Mock(),
    "chat_format": Mock(),
}):
    from generation_kv_cache_paged import ModelArgs, PagedKVCache


class PagedKVCacheTest(unittest.TestCase):
    def setUp(self):
        # Exercise the allocator on CPU without changing its CUDA inference API.
        for name in ("zeros", "tensor"):
            factory = getattr(torch, name)

            def on_cpu(*args, _factory=factory, **kwargs):
                kwargs["device"] = "cpu"
                return _factory(*args, **kwargs)

            patcher = patch.object(torch, name, side_effect=on_cpu)
            patcher.start()
            self.addCleanup(patcher.stop)

    def make_cache(self, lengths):
        tokens = torch.zeros((len(lengths), max(lengths)), dtype=torch.long)
        for index, length in enumerate(lengths):
            tokens[index, :length] = 1
        args = ModelArgs(dim=8, n_heads=2, n_kv_heads=1, max_seq_len=1024)
        return PagedKVCache(tokens, args, len(lengths), 1024, SimpleNamespace(pad_id=0))

    def test_prompt_blocks_cover_every_token(self):
        lengths = [1, 255, 256, 257, 512, 600]
        cache = self.make_cache(lengths)
        self.assertEqual((cache.get_last_pos() + 1).tolist(), lengths)
        allocated = []
        for index, length in enumerate(lengths):
            blocks = cache.block_table[index]
            self.assertEqual(len(blocks), (length + 255) // 256)
            self.assertEqual(sum(filled for _, filled in blocks), length)
            self.assertTrue(all(filled == 256 for _, filled in blocks[:-1]))
            self.assertTrue(1 <= blocks[-1][1] <= 256)
            allocated.extend(block for block, _ in blocks)
        self.assertEqual(len(set(allocated)), len(allocated))
        self.assertEqual(len(cache.free_blocks), cache.num_blocks - len(allocated))
        self.assertTrue(set(allocated).isdisjoint(cache.free_blocks))

    def test_decode_reserves_next_block_at_boundary(self):
        cache = self.make_cache([255, 256, 512, 600])
        cache.update([False] * 4, [False] * 4)
        self.assertEqual((cache.get_last_pos() + 1).tolist(), [256, 257, 513, 601])
        self.assertEqual([len(b) for b in cache.block_table.values()], [1, 2, 3, 3])
        cache.update([False] * 4, [False] * 4)
        self.assertEqual((cache.get_last_pos() + 1).tolist(), [257, 258, 514, 602])
        self.assertEqual([len(b) for b in cache.block_table.values()], [2, 2, 3, 3])


if __name__ == "__main__":
    unittest.main()
