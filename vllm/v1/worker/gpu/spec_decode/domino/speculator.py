# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Domino speculator: block-parallel draft backbone + GRU-corrected sampling."""

from typing import Any

import torch

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample
from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator


class DominoSpeculator(DSparkSpeculator):
    """Domino is DSpark-shaped, but replaces the Markov head with a GRU
    correction loop.

    Execution shape:
      * ``num_query_per_req == num_speculative_tokens`` (anchor-as-first),
      * one parallel draft forward,
      * left-to-right sampling where each position adds a GRU-conditioned
        residual logit correction.
    """

    _speculator_name = "Domino"

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        super().__init__(vllm_config, device)

        # Domino draft layers are non-causal on every layer (block-
        # bidirectional training), including sliding-window layers; the
        # DFlash default (sliding layers causal) does not apply.
        from vllm.model_executor.models.qwen3_domino import (
            domino_has_any_non_causal,
        )

        self.requires_non_causal = domino_has_any_non_causal(
            self.draft_model_config.hf_config
        )

        self.sample_from_anchor = True
        self.num_query_per_req = self.num_speculative_steps

        hf_config = self.draft_model_config.hf_config
        dflash_config = getattr(hf_config, "dflash_config", None) or {}
        self.target_hidden_size = dflash_config.get(
            "target_hidden_size", hf_config.hidden_size
        )
        self.num_draft_layers = hf_config.num_hidden_layers

        # Context representation in the proposer buffer, matching
        # ``combine_hidden_states``: flare fuses one routed feature per draft
        # layer ([num_tokens, num_draft_layers * target_hidden_size]), fc
        # projects the concatenated taps to draft width once and every layer
        # reads that same feature ([num_tokens, hidden_size]).
        self.fusion_mode = dflash_config.get("fusion_mode", "fc")
        if self.fusion_mode == "flare":
            self.hidden_size = self.num_draft_layers * self.target_hidden_size
        else:
            self.hidden_size = hf_config.hidden_size
        self.hidden_states = torch.zeros(
            self.max_num_tokens,
            self.hidden_size,
            dtype=self.dtype,
            device=device,
        )

        # GRU warm-up over the committed tokens left of the anchor
        # (dflash_config.correction_warmup_len, resolved in load_draft_model).
        # The ring is keyed by request-state slot; the active-order seed ids
        # are refreshed each propose for the captured walk to read.
        self.correction_warmup_len = 0
        self._warmup_ring: torch.Tensor | None = None
        self._warmup_seed_ids: torch.Tensor | None = None
        self._warmup_seen: torch.Tensor | None = None
        self._warmup_prev_anchor: torch.Tensor | None = None
        self._warmup_last_active_pos: torch.Tensor | None = None

    def init_cudagraph_manager(self, cudagraph_mode: CUDAGraphMode) -> None:
        # Let DFlash's graph manager decide: FULL / FULL_DECODE_ONLY is
        # captured as an ACL graph on Ascend, while PIECEWISE / NONE stays
        # eager. The fixed-length GRU correction loop is unrolled during
        # capture, so it is part of the same graph.
        super().init_cudagraph_manager(cudagraph_mode)

    def _skip_draft_dp_sync(self) -> bool:
        # The Domino draft backbone is a dense transformer with no MoE/EP
        # layers, so its proposal forward runs no cross-DP collectives. Skip
        # the second per-step DP dispatch all-reduce inherited from DFlash;
        # that all-reduce can deadlock when DP ranks are not all executing a
        # real batch (e.g. idle ranks running dummy batches).
        return True

    def load_draft_model(
        self,
        target_model: torch.nn.Module,
        target_attn_layer_names: set[str],
    ) -> torch.nn.Module:
        from vllm.model_executor.models.qwen3_domino import load_domino_model

        model = load_domino_model(target_model, self.vllm_config)
        self.correction_warmup_len = int(
            getattr(model, "correction_warmup_len", 0)
        )
        if self.correction_warmup_len > 0:
            n = self.correction_warmup_len
            self._warmup_ring = torch.zeros(
                self.max_num_reqs, n, dtype=torch.int64, device=self.device
            )
            self._warmup_seed_ids = torch.zeros(
                self.max_num_reqs, n, dtype=torch.int64, device=self.device
            )
            self._warmup_seen = torch.zeros(
                self.max_num_reqs, dtype=torch.bool, device=self.device
            )
            self._warmup_prev_anchor = torch.zeros(
                self.max_num_reqs, dtype=torch.int64, device=self.device
            )
            self._warmup_last_active_pos = torch.full(
                (self.max_num_reqs,), -1, dtype=torch.int64, device=self.device
            )
        if self.draft_logits is not None and model.draft_id_to_target_id is not None:
            d2t = model.draft_id_to_target_id
            self._d2t_scatter_index = (
                torch.arange(d2t.shape[0], device=d2t.device) + d2t
            )
            self._draft_scatter_buf = torch.full(
                (self.max_num_reqs, self.vocab_size),
                float("-inf"),
                dtype=self.draft_logits.dtype,
                device=self.device,
            )
        return model

    def remove_request(self, req_idx: int) -> None:
        # A freed state slot must not leak its ring into the next request.
        if self._warmup_seen is not None:
            self._warmup_seen[req_idx] = False
            self._warmup_last_active_pos[req_idx] = -1

    def _update_warmup_ring(
        self,
        num_reqs: int,
        idx_mapping: torch.Tensor,
        num_sampled: torch.Tensor,
        last_sampled: torch.Tensor,
        next_prefill_tokens: torch.Tensor,
    ) -> None:
        new_rows, state, anchor = _advance_warmup_ring(
            ring=self._warmup_ring,
            seen=self._warmup_seen,
            prev_anchor=self._warmup_prev_anchor,
            last_active_pos=self._warmup_last_active_pos,
            draft_tokens=self.draft_tokens,
            idx_mapping=idx_mapping,
            num_reqs=num_reqs,
            num_sampled=num_sampled,
            last_sampled=last_sampled,
            next_prefill_tokens=next_prefill_tokens,
            warmup_len=self.correction_warmup_len,
            num_speculative_steps=self.num_speculative_steps,
            device=self.device,
        )
        self._warmup_ring[state] = new_rows
        self._warmup_seed_ids[:num_reqs] = new_rows
        self._warmup_seen[state] = True
        self._warmup_prev_anchor[state] = anchor

    def propose(
        self,
        input_batch,
        attn_metadata: dict[str, Any],
        slot_mappings: dict[str, torch.Tensor],
        last_hidden_states: torch.Tensor,
        aux_hidden_states: list[torch.Tensor] | None,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        last_sampled: torch.Tensor,
        next_prefill_tokens: torch.Tensor,
        temperature: torch.Tensor,
        seeds: torch.Tensor,
        num_tokens_across_dp: torch.Tensor | None = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        is_profile: bool = False,
    ) -> torch.Tensor:
        if self.correction_warmup_len > 0 and not dummy_run:
            self._update_warmup_ring(
                input_batch.num_reqs,
                input_batch.idx_mapping,
                num_sampled,
                last_sampled,
                next_prefill_tokens,
            )
        draft_tokens = super().propose(
            input_batch,
            attn_metadata,
            slot_mappings,
            last_hidden_states,
            aux_hidden_states,
            num_sampled,
            num_rejected,
            last_sampled,
            next_prefill_tokens,
            temperature,
            seeds,
            num_tokens_across_dp,
            dummy_run,
            skip_attn_for_dummy_run,
            mm_inputs,
            is_profile=is_profile,
        )
        if self.correction_warmup_len > 0 and not dummy_run:
            # The walk just rewrote draft_tokens in this round's active
            # order; remember it so the next update maps state -> drafts.
            state = input_batch.idx_mapping[: input_batch.num_reqs].to(torch.int64)
            self._warmup_last_active_pos[state] = torch.arange(
                input_batch.num_reqs, device=self.device
            )
        return draft_tokens

    def _sample_step(
        self,
        logits_i: torch.Tensor,
        idx_map_i: torch.Tensor,
        sample_pos_i: torch.Tensor,
        col: int,
    ) -> torch.Tensor:
        if self.draft_logits is not None:
            if self._d2t_scatter_index is not None:
                assert self._draft_scatter_buf is not None
                buf = self._draft_scatter_buf[: logits_i.shape[0]]
                buf.index_copy_(1, self._d2t_scatter_index, logits_i.to(buf.dtype))
                logits_i = buf
            return gumbel_sample(
                logits_i,
                idx_map_i,
                self.temperature,
                self.seeds,
                sample_pos_i - 1,
                apply_temperature=True,
                output_processed_logits=self.draft_logits,
                output_processed_logits_col=self._step_cols[col],
                use_fp64=self.use_fp64_gumbel,
            )
        return self.model.map_draft_to_target(logits_i.argmax(dim=-1))

    def _sample_sequential(
        self,
        num_reqs: int,
        head_hidden: torch.Tensor,
    ) -> None:
        n_spec = self.num_speculative_steps
        num_sample = num_reqs * n_spec
        sample_hidden = head_hidden[self.sample_indices[:num_sample]]

        shared = bool(getattr(self.model, "shared_emb_readout", False))
        if shared:
            # Shared emb readout: the whole vocab projection per step is the
            # folded [V, emb] head applied to head_down_proj(z); base logits
            # are never materialized in vocab space.
            base_emb = self.model.head_emb(sample_hidden)
            base_emb = base_emb.view(num_reqs, n_spec, -1)
        else:
            base_logits = self.model.compute_draft_logits(sample_hidden)
            vocab_size = base_logits.shape[-1]
            base_logits = base_logits.view(num_reqs, n_spec, vocab_size)

        idx_map = self.sample_idx_mapping[:num_sample].view(num_reqs, n_spec)
        sample_pos = self.sample_pos[:num_sample].view(num_reqs, n_spec)

        prefix_len = int(getattr(self.model, "pure_draft_prefix_len", 0))
        if prefix_len > n_spec:
            raise ValueError(
                f"Domino pure_draft_prefix_len ({prefix_len}) cannot exceed "
                f"num_speculative_tokens ({n_spec})"
            )

        anchor = self.input_buffers.input_ids[self._anchor_idx[:num_reqs]]
        prefix_ids = torch.empty(
            num_reqs,
            1 + prefix_len,
            dtype=torch.int64,
            device=self.device,
        )
        prefix_ids[:, 0] = anchor

        for i in range(prefix_len):
            logits_i = (
                self.model.emb_logits(base_emb[:, i])
                if shared
                else base_logits[:, i]
            )
            draft_i = self._sample_step(
                logits_i,
                idx_map[:, i],
                sample_pos[:, i],
                i,
            )
            self.draft_tokens[:num_reqs, i] = draft_i
            prefix_ids[:, 1 + i] = draft_i

        seed_ids = prefix_ids
        if self.correction_warmup_len > 0:
            seed_ids = torch.cat(
                [self._warmup_seed_ids[:num_reqs], prefix_ids], dim=1
            )
        prefix_embeds = self.model.embed_tokens(seed_ids)
        _, gru_hidden = self.model.domino_gru_forward(prefix_embeds)

        sample_hidden_3d = sample_hidden.view(num_reqs, n_spec, -1)
        for i in range(prefix_len, n_spec):
            if shared:
                u_i = base_emb[:, i] + self.model.correction_emb(gru_hidden[0])
                logits_i = self.model.emb_logits(u_i)
            else:
                z_i = sample_hidden_3d[:, i]
                s_i = gru_hidden[0]
                features = torch.cat([z_i, s_i], dim=-1)
                correction = self.model.correction_bias(features)
                logits_i = base_logits[:, i] + correction

            draft_i = self._sample_step(
                logits_i,
                idx_map[:, i],
                sample_pos[:, i],
                i,
            )
            self.draft_tokens[:num_reqs, i] = draft_i

            if i + 1 < n_spec:
                next_embeds = self.model.embed_tokens(draft_i)
                _, gru_hidden = self.model.domino_gru_cell(
                    next_embeds,
                    gru_hidden,
                )

    def _generate_draft(
        self,
        num_reqs: int,
        num_tokens_padded: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    ) -> None:
        head_hidden = self._run_model(
            num_tokens_padded,
            attn_metadata,
            slot_mappings,
            num_tokens_across_dp,
            cudagraph_runtime_mode,
        )
        self._sample_sequential(num_reqs, head_hidden)


def _advance_warmup_ring(
    *,
    ring: torch.Tensor,
    seen: torch.Tensor,
    prev_anchor: torch.Tensor,
    last_active_pos: torch.Tensor,
    draft_tokens: torch.Tensor,
    idx_mapping: torch.Tensor,
    num_reqs: int,
    num_sampled: torch.Tensor,
    last_sampled: torch.Tensor,
    next_prefill_tokens: torch.Tensor,
    warmup_len: int,
    num_speculative_steps: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Advance the committed-token ring for the active requests.

    ``num_sampled`` is THIS round's commit count (accepted drafts plus the
    bonus), zeroed for chunked-prefill rows; delta == 0 appends nothing so
    uncommitted anchors never enter the ring. The first sighting of a request
    broadcasts the anchor, matching training's sequence-start clamp (the
    prompt tail never reaches the ring). Returns the new ring rows, the state
    indices they belong to, and each row's anchor.
    """
    state = idx_mapping[:num_reqs].to(torch.int64)
    num_sampled = num_sampled[:num_reqs].to(torch.int64)
    first_sighting = ~seen[state]
    anchor = torch.where(
        num_sampled > 0,
        last_sampled[state].to(torch.int64),
        next_prefill_tokens[state].to(torch.int64),
    )

    ring_rows = ring[state]
    ring_rows = torch.where(
        first_sighting.unsqueeze(1),
        anchor.unsqueeze(1).expand(num_reqs, warmup_len),
        ring_rows,
    )

    # Newly committed stream since the last propose, in order:
    # [previous anchor, accepted drafts]. A request absent from the previous
    # round has no valid draft row to read, and one that skipped a verify can
    # commit at most the bonus, so its accepted-draft mask stays empty.
    delta = torch.where(first_sighting, torch.zeros_like(num_sampled), num_sampled)
    last_pos = last_active_pos[state]
    has_last = last_pos >= 0
    last_drafts = draft_tokens[last_pos.clamp(min=0)].to(torch.int64)
    appended = torch.empty(
        num_reqs, 1 + num_speculative_steps, dtype=torch.int64, device=device
    )
    appended[:, 0] = torch.where(has_last, prev_anchor[state], anchor)
    slot = torch.arange(1, 1 + num_speculative_steps, device=device)
    accept_mask = (delta.view(-1, 1) > slot.view(1, -1)) & has_last.view(-1, 1)
    appended[:, 1:] = torch.where(accept_mask, last_drafts, anchor.unsqueeze(1))

    # Last warmup_len entries of [old ring, appended]: the valid window
    # starts at delta.
    combined = torch.cat([ring_rows, appended], dim=1)
    gather_idx = delta.view(-1, 1) + torch.arange(warmup_len, device=device).view(
        1, -1
    )
    new_rows = torch.gather(combined, 1, gather_idx)
    return new_rows, state, anchor
