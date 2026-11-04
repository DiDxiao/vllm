# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Committed-token ring regressions for the Domino GRU warm-up."""

from types import MethodType
from types import SimpleNamespace

import pytest
import torch

from vllm.v1.worker.gpu.spec_decode.domino.speculator import (
    DominoSpeculator,
    _advance_warmup_ring,
)

MAX_REQS = 4
WARMUP_LEN = 3
NUM_STEPS = 4


def _state():
    torch.manual_seed(0)
    device = torch.device("cpu")
    return {
        "ring": torch.zeros(MAX_REQS, WARMUP_LEN, dtype=torch.int64),
        "seen": torch.zeros(MAX_REQS, dtype=torch.bool),
        "inited": torch.zeros(MAX_REQS, dtype=torch.bool),
        "prev_anchor": torch.zeros(MAX_REQS, dtype=torch.int64),
        "last_drafts": torch.zeros(MAX_REQS, NUM_STEPS, dtype=torch.int64),
        # RequestState.last_sampled_tokens is [max_num_reqs, 1].
        "last_sampled": torch.zeros(MAX_REQS, 1, dtype=torch.int64),
        "next_prefill_tokens": torch.zeros(MAX_REQS, dtype=torch.int32),
        "device": device,
    }


@pytest.fixture()
def state():
    return _state()


def advance(st, idx_mapping, num_sampled):
    rows, slots, anchor, active = _advance_warmup_ring(
        ring=st["ring"],
        seen=st["seen"],
        inited=st["inited"],
        prev_anchor=st["prev_anchor"],
        last_drafts=st["last_drafts"],
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
    st["seen"][slots[active]] = True
    st["prev_anchor"][slots[active]] = anchor[active]
    return rows, slots, anchor, active


def test_first_decode_row_uses_the_admission_seeded_history(state):
    # init_request seeded the ring with the prompt tail [11, 12, 13].
    state["inited"][2] = True
    state["ring"][2] = torch.tensor([11, 12, 13])
    state["last_sampled"][2, 0] = 100  # first response token = anchor
    rows, _, anchor, active = advance(
        state, torch.tensor([2]), torch.tensor([1], dtype=torch.int32)
    )
    assert torch.equal(rows, torch.tensor([[11, 12, 13]]))
    assert torch.equal(anchor, torch.tensor([100]))
    assert active.tolist() == [True]


def test_unseeded_first_row_falls_back_to_anchor_broadcast(state):
    # No admission hook ran: the anchor broadcast stands in.
    state["last_sampled"][2, 0] = 100
    rows, _, _, active = advance(
        state, torch.tensor([2]), torch.tensor([1], dtype=torch.int32)
    )
    assert torch.equal(rows, torch.tensor([[100, 100, 100]]))
    assert active.tolist() == [True]


def test_chunked_prefill_row_is_inert(state):
    # num_sampled == 0 (chunked prefill): next_prefill_tokens holds a prompt
    # token of the NEXT chunk; the ring must not absorb it.
    state["inited"][2] = True
    state["seen"][2] = True
    state["prev_anchor"][2] = 40
    state["ring"][2] = torch.tensor([7, 8, 40])
    state["last_drafts"][2] = torch.tensor([71, 72, 73, 74])
    state["next_prefill_tokens"][2] = 96  # mid-prompt token
    rows, _, _, active = advance(
        state, torch.tensor([2]), torch.tensor([0], dtype=torch.int32)
    )
    assert torch.equal(rows, torch.tensor([[7, 8, 40]]))
    assert active.tolist() == [False]


def test_steady_round_appends_prev_anchor_and_accepted_drafts(state):
    state["inited"][2] = True
    state["seen"][2] = True
    state["prev_anchor"][2] = 100
    state["ring"][2] = torch.tensor([300, 299, 100])
    state["last_drafts"][2] = torch.tensor([7, 8, 11, 12])
    state["last_sampled"][2, 0] = 200
    rows, _, _, _ = advance(
        state, torch.tensor([2]), torch.tensor([3], dtype=torch.int32)
    )
    # appended = [100, 7, 8]; last 3 of [300, 299, 100, 100, 7, 8]
    assert torch.equal(rows, torch.tensor([[100, 7, 8]]))


def test_full_rejection_appends_prev_anchor_only(state):
    state["inited"][2] = True
    state["seen"][2] = True
    state["prev_anchor"][2] = 200
    state["ring"][2] = torch.tensor([100, 7, 8])
    state["last_drafts"][2] = torch.tensor([7, 8, 11, 12])
    state["last_sampled"][2, 0] = 300
    rows, _, _, _ = advance(
        state, torch.tensor([2]), torch.tensor([1], dtype=torch.int32)
    )
    # appended = [200]; last 3 of [100, 7, 8, 200]
    assert torch.equal(rows, torch.tensor([[7, 8, 200]]))


def test_drafts_are_read_from_the_own_state_slot(state):
    # A deferred/dummy round rewrote other requests' draft rows; each
    # request's own last_drafts snapshot is what enters its ring.
    for slot, ring, prev, drafts in (
        (0, [10, 10, 10], 10, [81, 82, 83, 84]),
        (2, [20, 20, 20], 20, [71, 72, 73, 74]),
    ):
        state["inited"][slot] = True
        state["seen"][slot] = True
        state["prev_anchor"][slot] = prev
        state["ring"][slot] = torch.tensor(ring)
        state["last_drafts"][slot] = torch.tensor(drafts)
    state["last_sampled"][0, 0] = 60
    state["last_sampled"][2, 0] = 70
    rows, slots, anchor, _ = advance(
        state,
        torch.tensor([2, 0]),
        torch.tensor([3, 2], dtype=torch.int32),
    )
    assert torch.equal(slots, torch.tensor([2, 0]))
    assert torch.equal(rows[0], torch.tensor([20, 71, 72]))
    assert torch.equal(rows[1], torch.tensor([10, 10, 81]))
    assert torch.equal(anchor, torch.tensor([70, 60]))


def _speculator_stub():
    stub = SimpleNamespace(
        correction_warmup_len=WARMUP_LEN,
        num_speculative_steps=NUM_STEPS,
        device=torch.device("cpu"),
        _warmup_ring=torch.zeros(MAX_REQS, WARMUP_LEN, dtype=torch.int64),
        _warmup_seed_ids=torch.zeros(MAX_REQS, WARMUP_LEN, dtype=torch.int64),
        _warmup_seen=torch.zeros(MAX_REQS, dtype=torch.bool),
        _warmup_inited=torch.zeros(MAX_REQS, dtype=torch.bool),
        _warmup_prev_anchor=torch.zeros(MAX_REQS, dtype=torch.int64),
        _warmup_last_drafts=torch.zeros(MAX_REQS, NUM_STEPS, dtype=torch.int64),
        draft_tokens=torch.zeros(MAX_REQS, NUM_STEPS, dtype=torch.int64),
    )
    stub._update_warmup_ring = MethodType(
        DominoSpeculator._update_warmup_ring, stub
    )
    stub.init_request = MethodType(DominoSpeculator.init_request, stub)
    stub.remove_request = MethodType(DominoSpeculator.remove_request, stub)
    return stub


def test_update_warmup_ring_end_to_end_through_prefill_and_decode():
    spec = _speculator_stub()
    idx_mapping = torch.tensor([3])

    # Admission: prompt tail seeds the ring.
    spec.init_request(3, [5, 11, 12, 13])
    assert torch.equal(
        spec._warmup_ring[3], torch.tensor([11, 12, 13])
    )

    # Chunked-prefill propose rounds are inert; next_prefill_tokens holds a
    # mid-prompt token of the next chunk ([max_num_reqs] contract).
    next_prefill = torch.zeros(MAX_REQS, dtype=torch.int32)
    next_prefill[3] = 96
    for _ in range(3):
        spec._update_warmup_ring(
            1,
            idx_mapping,
            torch.tensor([0], dtype=torch.int32),
            torch.zeros(MAX_REQS, 1, dtype=torch.int64),
            next_prefill,
        )
    # Inert rows carry their own ring (the admission history); the mid-prompt
    # token never enters, and the slot is not yet marked seen.
    assert torch.equal(
        spec._warmup_seed_ids[0], torch.tensor([11, 12, 13])
    )
    assert not spec._warmup_seen[3]
    assert spec._warmup_inited[3]

    # First decode round: keep the seeded history, mark seen.
    spec._warmup_last_drafts[3] = torch.tensor([7, 8, 11, 12])
    last_sampled = torch.zeros(MAX_REQS, 1, dtype=torch.int64)
    last_sampled[3, 0] = 100
    spec._update_warmup_ring(
        1,
        idx_mapping,
        torch.tensor([1], dtype=torch.int32),
        last_sampled,
        next_prefill,
    )
    assert torch.equal(
        spec._warmup_seed_ids[0], torch.tensor([11, 12, 13])
    )
    assert spec._warmup_seen[3]
    assert spec._warmup_prev_anchor[3].item() == 100

    # Steady round with two accepted drafts.
    spec._warmup_last_drafts[3] = torch.tensor([7, 8, 11, 12])
    last_sampled[3, 0] = 200
    spec._update_warmup_ring(
        1,
        idx_mapping,
        torch.tensor([3], dtype=torch.int32),
        last_sampled,
        next_prefill,
    )
    assert torch.equal(
        spec._warmup_seed_ids[0], torch.tensor([100, 7, 8])
    )


def test_removed_then_readmitted_slot_reseeds_from_new_history():
    spec = _speculator_stub()
    spec.init_request(1, [50, 51, 52, 53])
    spec.remove_request(1)
    # New request reuses the slot with a shorter history.
    spec.init_request(1, [70, 71])
    assert torch.equal(
        spec._warmup_ring[1], torch.tensor([70, 70, 71])
    )
    assert spec._warmup_inited[1]
