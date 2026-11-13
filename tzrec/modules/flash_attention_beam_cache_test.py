# Copyright (c) 2026, Alibaba Group;
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#    http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

import torch
from parameterized import parameterized

from tzrec.modules.flash_attention_beam_cache import (
    FlashAttentionBeamCache,
    create_flash_attention_beam_cache,
)
from tzrec.utils.test_util import (
    create_tiny_causal_lm,
    flash_attn_unavailable,
    mark_ci_scope,
    nv_gpu_unavailable,
    parameterized_name_func,
)


class FlashAttentionBeamPrefixTest(unittest.TestCase):
    def test_prefill_keeps_original_layout_and_attention_compute_dtype(self):
        cache = FlashAttentionBeamCache(1)
        key = torch.randn(3, 2, 257, 8)
        value = torch.randn_like(key).bfloat16()
        returned_key, returned_value = cache.update(key, value, 0)
        self.assertIs(returned_key, key)
        self.assertIs(returned_value, value)
        self.assertEqual(cache.get_seq_length(), 257)
        self.assertEqual(cache._prefix_keys[0].dtype, torch.bfloat16)
        torch.testing.assert_close(cache._prefix_keys[0], key.bfloat16())
        torch.testing.assert_close(cache._prefix_values[0], value)
        prefix = cache._prefix_keys[0]
        cache.reorder_cache(torch.tensor([0, 0, 1, 1, 2, 2]))
        self.assertIs(cache._prefix_keys[0], prefix)
        self.assertEqual(cache._prefix_keys[0].shape[0], 3)

    def test_parent_selection_shares_prefixes_and_isolates_beam_tails(self):
        cache = FlashAttentionBeamCache(1)
        key = torch.randn(2, 2, 513, 8).bfloat16()
        value = torch.randn_like(key)
        cache.update(key, value, 0)
        roots = cache._prefix_keys[0].clone()
        owners = torch.tensor([0, 0, 1, 1])
        cache.reorder_cache(owners)
        tail = (
            torch.arange(4, dtype=torch.bfloat16).reshape(4, 1, 1, 1).expand(4, 2, 1, 8)
        )
        full_key, full_value = cache.update(tail, tail + 10, 0)
        torch.testing.assert_close(full_key[:, :, :513], key[owners])
        torch.testing.assert_close(full_value[:, :, :513], value[owners])
        torch.testing.assert_close(full_key[:, :, 513:], tail)
        parents = torch.tensor([1, 1, 0, 3, 2, 3])
        cache.reorder_cache(parents)
        torch.testing.assert_close(cache._tail_keys[0], tail[parents])
        cache._tail_keys[0][0].fill_(99)
        self.assertEqual(float(cache._tail_keys[0][1, 0, 0, 0]), 1)
        torch.testing.assert_close(cache._prefix_keys[0], roots)
        self.assertEqual(cache.get_seq_length(), 514)


@mark_ci_scope("gpu")
@unittest.skipIf(*nv_gpu_unavailable)
@unittest.skipIf(*flash_attn_unavailable)
class FlashAttentionBeamDecodeTest(unittest.TestCase):
    @parameterized.expand(
        [
            ["qwen2", torch.bfloat16, torch.bfloat16, [5, 8]],
            ["qwen3", torch.bfloat16, torch.bfloat16, [5, 8]],
            ["qwen2", torch.float32, torch.bfloat16, [255, 256, 257]],
            ["qwen3", torch.bfloat16, torch.bfloat16, [255, 256, 257]],
            ["qwen2", torch.float16, torch.float16, [255, 256, 257]],
        ],
        name_func=parameterized_name_func,
    )
    def test_two_branching_steps_match_ordinary_fa2(
        self, model_type, parameter_dtype, compute_dtype, lengths
    ):
        lm = create_tiny_causal_lm(
            100,
            model_type=model_type,
            attn_implementation="flash_attention_2",
            torch_dtype=parameter_dtype,
        ).cuda()
        width, batch = max(lengths), len(lengths)
        ids = torch.arange(batch * width, device="cuda").reshape(batch, width) % 99
        mask = torch.zeros_like(ids)
        for row, length in enumerate(lengths):
            mask[row, -length:] = 1
        positions = (mask.cumsum(-1) - 1).clamp(min=0)
        originals = [layer.self_attn.forward for layer in lm.model.layers]
        with torch.no_grad(), torch.autocast("cuda", dtype=compute_dtype):
            reference = lm(
                input_ids=ids,
                attention_mask=mask,
                position_ids=positions,
                use_cache=True,
                logits_to_keep=1,
            )
            ordinary = reference.past_key_values
            shared = create_flash_attention_beam_cache(lm)
            actual = lm(
                input_ids=ids,
                attention_mask=mask,
                position_ids=positions,
                past_key_values=shared,
                use_cache=True,
                logits_to_keep=1,
            )
            torch.testing.assert_close(actual.logits, reference.logits, atol=0, rtol=0)
            parents = torch.arange(batch, device="cuda").repeat_interleave(2)
            for step in range(2):
                ordinary.reorder_cache(parents)
                shared.reorder_cache(parents)
                mask = mask[parents]
                positions = mask.sum(-1, keepdim=True)
                mask = torch.cat([mask, mask.new_ones(len(parents), 1)], dim=1)
                tokens = (torch.arange(len(parents), device="cuda") + 17 + step)[
                    :, None
                ]
                reference = lm(
                    input_ids=tokens,
                    attention_mask=mask,
                    position_ids=positions,
                    past_key_values=ordinary,
                    use_cache=True,
                    logits_to_keep=1,
                )
                actual = lm(
                    input_ids=tokens,
                    attention_mask=mask,
                    position_ids=positions,
                    past_key_values=shared,
                    use_cache=True,
                    logits_to_keep=1,
                )
                torch.testing.assert_close(
                    actual.logits, reference.logits, atol=0, rtol=0
                )
                parents = (
                    torch.arange(batch, device="cuda")[:, None] * 2
                    + torch.tensor([1, 0, 1], device="cuda")
                ).flatten()
        for layer, original in zip(lm.model.layers, originals):
            self.assertEqual(layer.self_attn.forward, original)


if __name__ == "__main__":
    unittest.main()
