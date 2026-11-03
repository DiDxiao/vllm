# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Committed-token ring regressions for the Domino GRU warm-up."""

import pytest
import torch

from vllm.v1.worker.gpu.spec_decode.domino.speculator import _advance_warmup_ring

MAX_REQS = 4
WARMUP_LEN = 3
NUM_STEPS = 4


@pytest.fixture()
def state():
    torch.manual_seed(0)
    device = torch.device("cpu")
    return {
        "ring": torch.zeros(MAX_REQS, WARMUP_LEN, dtype=torch.int64),
        "seen": torch.zeros(MAX_REQS, dtype=torch.bool),
        "prev_anchor": torch.zeros(MAX_REQS, dtype=torch.int64),
        "last_active_pos": torch.full(
            (MAX_REQS,), -1, dtype=torch.int64
        ),
        "draft_tokens": torch.zeros(MAX_REQS, NUM_STEPS, dtype=torch.int64),
        "last_sampled": torch.zeros(MAX_REQS, dtype=torch.int32),
        "next_prefill_tokens": torch.zeros(MAX_REQS, dtype=torch.int32),
        "device": device,
    }


def advance(st, idx_mapping, num_sampled):
    rows, slots, anchor = _advance_warmup_ring(
        ring=st["ring"],
        seen=st["seen"],
        prev_anchor=st["prev_anchor"],
        last_active_pos=st["last_active_pos"],
        draft_tokens=st["draft_tokens"],
        idx_mapping=idx_mapping,
        num_reqs=idx_mapping.shape[0],
        num_sampled=num_sampled,
        last_sampled=st["last_sampled"],
        next_prefill_tokens=st["next_prefill_tokens"],
        warmup_len=WARMUP_LEN,
        num_speculative_steps=NUM_STEPS,
        device=st["device"],
    )
    st["ring"][slots] = rows
    st["seen"][slots] = True
    st["prev_anchor"][slots] = anchor
    return rows, slots, anchor


def test_first_sighting_broadcasts_prefill_anchor(state):
    state["next_prefill_tokens"][2] = 100
    rows, _, anchor = advance(
        state, torch.tensor([2]), torch.tensor([0], dtype=torch.int32)
    )
    assert torch.equal(rows, torch.tensor([[100, 100, 100]]))
    assert torch.equal(anchor, torch.tensor([100]))


def test_steady_round_appends_prev_anchor_and_accepted_drafts(state):
    state["seen"][2] = True
    state["prev_anchor"][2] = 100
    state["ring"][2] = torch.tensor([300, 299, 100])
    state["last_active_pos"][2] = 0
    state["draft_tokens"][0] = torch.tensor([7, 8, 11, 12])
    state["last_sampled"][2] = 200
    rows, _, _ = advance(
        state, torch.tensor([2]), torch.tensor([3], dtype=torch.int32)
    )
    # appended = [100, 7, 8]; last 3 of [300, 299, 100, 100, 7, 8]
    assert torch.equal(rows, torch.tensor([[100, 7, 8]]))


def test_full_rejection_appends_prev_anchor_only(state):
    state["seen"][2] = True
    state["prev_anchor"][2] = 200
    state["ring"][2] = torch.tensor([100, 7, 8])
    state["last_active_pos"][2] = 0
    state["draft_tokens"][0] = torch.tensor([7, 8, 11, 12])
    state["last_sampled"][2] = 300
    rows, _, _ = advance(
        state, torch.tensor([2]), torch.tensor([1], dtype=torch.int32)
    )
    # appended = [200]; last 3 of [100, 7, 8, 200]
    assert torch.equal(rows, torch.tensor([[7, 8, 200]]))


def test_delta_zero_appends_nothing(state):
    # Chunked-prefill rows report num_sampled == 0.
    state["seen"][2] = True
    state["prev_anchor"][2] = 200
    state["ring"][2] = torch.tensor([7, 8, 200])
    state["last_active_pos"][2] = 0
    state["next_prefill_tokens"][2] = 400
    rows, _, anchor = advance(
        state, torch.tensor([2]), torch.tensor([0], dtype=torch.int32)
    )
    assert torch.equal(rows, torch.tensor([[7, 8, 200]]))
    assert torch.equal(anchor, torch.tensor([400]))


def test_removed_slot_restarts_at_its_new_anchor(state):
    # remove_request clears seen/last_active_pos; a later request that reuses
    # the slot must not inherit the old ring even with num_sampled == 1.
    state["ring"][2] = torch.tensor([7, 8, 200])
    state["seen"][2] = False
    state["last_active_pos"][2] = -1
    state["last_sampled"][2] = 500
    rows, _, _ = advance(
        state, torch.tensor([2]), torch.tensor([1], dtype=torch.int32)
    )
    assert torch.equal(rows, torch.tensor([[500, 500, 500]]))


def test_reordered_batch_maps_draft_rows_by_state_slot(state):
    state["seen"][0] = True
    state["seen"][2] = True
    state["prev_anchor"][0] = 10
    state["prev_anchor"][2] = 20
    state["ring"][0] = torch.tensor([10, 10, 10])
    state["ring"][2] = torch.tensor([20, 20, 20])
    state["last_active_pos"][0] = 1
    state["last_active_pos"][2] = 0
    state["draft_tokens"][0] = torch.tensor([71, 72, 73, 74])
    state["draft_tokens"][1] = torch.tensor([81, 82, 83, 84])
    state["last_sampled"][0] = 60
    state["last_sampled"][2] = 70
    # state 2 moved to active position 0, state 0 to position 1.
    rows, slots, anchor = advance(
        state,
        torch.tensor([2, 0]),
        torch.tensor([3, 2], dtype=torch.int32),
    )
    assert torch.equal(slots, torch.tensor([2, 0]))
    assert torch.equal(rows[0], torch.tensor([20, 71, 72]))
    assert torch.equal(rows[1], torch.tensor([10, 10, 81]))
    assert torch.equal(anchor, torch.tensor([70, 60]))
