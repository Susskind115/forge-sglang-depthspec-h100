"""Exact dense-merged execution for unequal-width DepthSpec segments."""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import TYPE_CHECKING, Any

import torch

from sglang.srt.model_executor.forward_context import (
    ForwardContext,
    forward_context,
)

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


@dataclass(frozen=True)
class RaggedSegmentInput:
    """A segment plus optional continuation state for mixed schedules."""

    segment: Any
    forward_batch: ForwardBatch
    initial_hidden_states: torch.Tensor | None = None
    initial_time: float = 0.0


@dataclass(frozen=True)
class RaggedSegmentGroup:
    """One attention group retaining its original ForwardBatch contract."""

    segment: Any
    forward_batch: ForwardBatch
    row_start: int
    row_end: int
    initial_hidden_states: torch.Tensor | None = None
    initial_time: float = 0.0

    @property
    def row_count(self) -> int:
        return self.row_end - self.row_start

    @property
    def schedule_length(self) -> int:
        return len(self.segment.loop_indices)


@dataclass(frozen=True)
class RaggedActiveSegment:
    """One group's execution state at a recurrence micro-step."""

    group: RaggedSegmentGroup
    schedule_index: int
    loop_index: int
    step_weight: float
    current_time: float


@dataclass(frozen=True)
class RaggedRecurrenceStep:
    """Dense-mergeable active rows at one recurrence position."""

    step_index: int
    active_segments: tuple[RaggedActiveSegment, ...]

    @property
    def row_count(self) -> int:
        return sum(item.group.row_count for item in self.active_segments)


@dataclass(frozen=True)
class RaggedSegmentExecutionState:
    """Per-segment raw recurrence state after a ragged execution."""

    hidden_states: torch.Tensor
    current_time: float
    next_loop_index: int


@dataclass(frozen=True)
class RaggedExecutionResult:
    """Concatenated outputs plus per-segment continuation states."""

    normalized_hidden_states: torch.Tensor
    group_states: tuple[RaggedSegmentExecutionState, ...]


@dataclass(frozen=True)
class RaggedForwardPlan:
    """Row-level layout for a mixed-phase physical forward."""

    groups: tuple[RaggedSegmentGroup, ...]
    input_ids: torch.Tensor
    positions: torch.Tensor
    out_cache_loc: torch.Tensor
    qo_indptr: torch.Tensor
    row_segment_id: torch.Tensor
    row_request_count: torch.Tensor
    row_request_id: torch.Tensor
    row_loop_id: torch.Tensor
    row_step_weight: torch.Tensor
    row_initial_time: torch.Tensor
    row_attention_contract_id: torch.Tensor
    row_kv_namespace_id: torch.Tensor
    loop_indices: tuple[int, ...]
    step_schedule: tuple[float, ...]
    attention_contracts: tuple[str, ...]
    kv_namespaces: tuple[str, ...]

    @property
    def row_count(self) -> int:
        return int(self.input_ids.shape[0])

    @property
    def padding_rows(self) -> int:
        return 0


def build_ragged_forward_plan(
    segment_batches: list[tuple[Any, ForwardBatch]],
    *,
    loop_indices: tuple[int, ...],
    step_schedule: tuple[float, ...],
) -> RaggedForwardPlan:
    """Build a no-padding layout from heterogeneous LoopSegments."""
    if not segment_batches:
        raise ValueError("Cannot build an empty RaggedForwardPlan")
    if len(loop_indices) != len(step_schedule):
        raise ValueError("Ragged forward loop schedule mismatch")

    input_ids = []
    positions = []
    out_cache_locs = []
    row_request_ids = []
    row_segment_id = []
    row_request_count = []
    row_loop_id = []
    row_step_weight = []
    row_initial_time = []
    row_attention_contract_id = []
    row_kv_namespace_id = []
    groups = []
    attention_contracts = sorted(
        {segment.attention_contract for segment, _ in segment_batches}
    )
    kv_namespaces = sorted(
        {
            "draft" if segment.loop_indices[0] >= 9 else "target"
            for segment, _ in segment_batches
        }
    )
    contract_to_id = {
        name: index for index, name in enumerate(attention_contracts)
    }
    namespace_to_id = {
        name: index for index, name in enumerate(kv_namespaces)
    }
    row_start = 0
    qo_values = [0]
    first_segment = segment_batches[0][0]
    for segment_id, (segment, forward_batch) in enumerate(segment_batches):
        row_count = int(segment.input_ids.shape[0])
        row_end = row_start + row_count
        input_ids.append(segment.input_ids)
        positions.append(segment.positions)
        out_cache_locs.append(segment.out_cache_loc)
        request_count = int(segment.seq_lens.shape[0])
        rows_per_request = row_count // request_count
        if row_count % request_count != 0:
            raise ValueError(f"Ragged segment {segment.phase} has invalid rows")
        if segment.req_pool_indices is None:
            raise ValueError("Ragged segments require req_pool_indices")
        row_request_ids.append(
            segment.req_pool_indices.repeat_interleave(rows_per_request)
        )
        row_segment_id.extend([segment_id] * row_count)
        row_request_count.extend(
            [int(segment.seq_lens.shape[0])] * row_count
        )
        row_loop_id.extend(
            [list(segment.loop_indices)] * row_count
        )
        row_step_weight.extend(
            [list(segment.step_schedule)] * row_count
        )
        row_initial_time.extend([0.0] * row_count)
        row_attention_contract_id.extend(
            [contract_to_id[segment.attention_contract]] * row_count
        )
        namespace = (
            "draft" if segment.loop_indices[0] >= 9 else "target"
        )
        row_kv_namespace_id.extend(
            [namespace_to_id[namespace]] * row_count
        )
        groups.append(
            RaggedSegmentGroup(
                segment=segment,
                forward_batch=forward_batch,
                row_start=row_start,
                row_end=row_end,
            )
        )
        qo_values.append(row_end)
        row_start = row_end

    device = first_segment.input_ids.device
    return RaggedForwardPlan(
        groups=tuple(groups),
        input_ids=torch.cat(input_ids),
        positions=torch.cat(positions),
        out_cache_loc=torch.cat(out_cache_locs),
        qo_indptr=torch.tensor(
            qo_values, dtype=torch.int32, device=device
        ),
        row_segment_id=torch.tensor(
            row_segment_id, dtype=torch.int32, device=device
        ),
        row_request_count=torch.tensor(
            row_request_count, dtype=torch.int32, device=device
        ),
        row_request_id=torch.cat(row_request_ids).to(
            dtype=torch.int32, device=device
        ),
        row_loop_id=torch.tensor(
            row_loop_id, dtype=torch.int32, device=device
        ),
        row_step_weight=torch.tensor(
            row_step_weight, dtype=torch.float32, device=device
        ),
        row_initial_time=torch.tensor(
            row_initial_time, dtype=torch.float32, device=device
        ),
        row_attention_contract_id=torch.tensor(
            row_attention_contract_id,
            dtype=torch.int32,
            device=device,
        ),
        row_kv_namespace_id=torch.tensor(
            row_kv_namespace_id,
            dtype=torch.int32,
            device=device,
        ),
        loop_indices=loop_indices,
        step_schedule=step_schedule,
        attention_contracts=tuple(attention_contracts),
        kv_namespaces=tuple(kv_namespaces),
    )


def build_heterogeneous_ragged_forward_plan(
    segment_inputs: list[RaggedSegmentInput],
) -> RaggedForwardPlan:
    """Build a no-padding plan where each segment owns its loop schedule."""
    if not segment_inputs:
        raise ValueError("Cannot build an empty heterogeneous plan")
    input_ids = []
    positions = []
    out_cache_locs = []
    row_request_ids = []
    row_segment_id = []
    row_request_count = []
    row_loop_id = []
    row_step_weight = []
    row_initial_time = []
    row_attention_contract_id = []
    row_kv_namespace_id = []
    groups = []
    attention_contracts = sorted(
        {item.segment.attention_contract for item in segment_inputs}
    )
    kv_namespaces = sorted(
        {
            "draft" if item.segment.loop_indices[0] >= 9 else "target"
            for item in segment_inputs
        }
    )
    contract_to_id = {
        name: index for index, name in enumerate(attention_contracts)
    }
    namespace_to_id = {
        name: index for index, name in enumerate(kv_namespaces)
    }
    max_schedule_length = max(
        len(item.segment.loop_indices) for item in segment_inputs
    )
    row_start = 0
    qo_values = [0]
    first_segment = segment_inputs[0].segment

    for segment_id, item in enumerate(segment_inputs):
        segment = item.segment
        row_count = int(segment.input_ids.shape[0])
        if len(segment.loop_indices) != len(segment.step_schedule):
            raise ValueError(f"Segment {segment.phase} schedule mismatch")
        request_count = int(segment.seq_lens.shape[0])
        if row_count % request_count:
            raise ValueError(f"Segment {segment.phase} has invalid rows")
        if segment.req_pool_indices is None:
            raise ValueError("Heterogeneous segments require req_pool_indices")
        if item.initial_hidden_states is not None and int(
            item.initial_hidden_states.shape[0]
        ) != row_count:
            raise ValueError(f"Segment {segment.phase} hidden row mismatch")

        row_end = row_start + row_count
        rows_per_request = row_count // request_count
        input_ids.append(segment.input_ids)
        positions.append(segment.positions)
        out_cache_locs.append(segment.out_cache_loc)
        row_request_ids.append(
            segment.req_pool_indices.repeat_interleave(rows_per_request)
        )
        row_segment_id.extend([segment_id] * row_count)
        row_request_count.extend([request_count] * row_count)
        padded_loops = list(segment.loop_indices) + [-1] * (
            max_schedule_length - len(segment.loop_indices)
        )
        padded_steps = list(segment.step_schedule) + [0.0] * (
            max_schedule_length - len(segment.step_schedule)
        )
        row_loop_id.extend([padded_loops] * row_count)
        row_step_weight.extend([padded_steps] * row_count)
        row_initial_time.extend([item.initial_time] * row_count)
        row_attention_contract_id.extend(
            [contract_to_id[segment.attention_contract]] * row_count
        )
        namespace = "draft" if segment.loop_indices[0] >= 9 else "target"
        row_kv_namespace_id.extend(
            [namespace_to_id[namespace]] * row_count
        )
        groups.append(
            RaggedSegmentGroup(
                segment=segment,
                forward_batch=item.forward_batch,
                row_start=row_start,
                row_end=row_end,
                initial_hidden_states=item.initial_hidden_states,
                initial_time=item.initial_time,
            )
        )
        qo_values.append(row_end)
        row_start = row_end

    device = first_segment.input_ids.device
    return RaggedForwardPlan(
        groups=tuple(groups),
        input_ids=torch.cat(input_ids),
        positions=torch.cat(positions),
        out_cache_loc=torch.cat(out_cache_locs),
        qo_indptr=torch.tensor(qo_values, dtype=torch.int32, device=device),
        row_segment_id=torch.tensor(
            row_segment_id, dtype=torch.int32, device=device
        ),
        row_request_count=torch.tensor(
            row_request_count, dtype=torch.int32, device=device
        ),
        row_request_id=torch.cat(row_request_ids).to(
            dtype=torch.int32, device=device
        ),
        row_loop_id=torch.tensor(row_loop_id, dtype=torch.int32, device=device),
        row_step_weight=torch.tensor(
            row_step_weight, dtype=torch.float32, device=device
        ),
        row_initial_time=torch.tensor(
            row_initial_time, dtype=torch.float32, device=device
        ),
        row_attention_contract_id=torch.tensor(
            row_attention_contract_id, dtype=torch.int32, device=device
        ),
        row_kv_namespace_id=torch.tensor(
            row_kv_namespace_id, dtype=torch.int32, device=device
        ),
        loop_indices=(),
        step_schedule=(),
        attention_contracts=tuple(attention_contracts),
        kv_namespaces=tuple(kv_namespaces),
    )


def compile_ragged_schedule(
    plan: RaggedForwardPlan,
) -> tuple[RaggedRecurrenceStep, ...]:
    """Compile per-segment schedules into shared dense micro-steps."""
    schedule_positions = [0 for _ in plan.groups]
    current_times = [group.initial_time for group in plan.groups]
    steps = []
    step_index = 0
    while any(
        position < group.schedule_length
        for position, group in zip(schedule_positions, plan.groups)
    ):
        active_segments = []
        for group_id, (position, group) in enumerate(
            zip(schedule_positions, plan.groups)
        ):
            if position >= group.schedule_length:
                continue
            active_segments.append(
                RaggedActiveSegment(
                    group=group,
                    schedule_index=position,
                    loop_index=group.segment.loop_indices[position],
                    step_weight=group.segment.step_schedule[position],
                    current_time=current_times[group_id],
                )
            )
        steps.append(
            RaggedRecurrenceStep(
                step_index=step_index,
                active_segments=tuple(active_segments),
            )
        )
        for group_id, group in enumerate(plan.groups):
            active = next(
                (
                    item
                    for item in active_segments
                    if item.group is group
                ),
                None,
            )
            if active is None:
                continue
            schedule_positions[group_id] += 1
            current_times[group_id] += active.step_weight
        step_index += 1
    return tuple(steps)


def _grouped_attention(
    block,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    plan: RaggedForwardPlan,
    loop_index: int,
    attention_backend,
) -> torch.Tensor:
    """Run attention per segment while preserving grouped current K/V."""
    output = torch.empty_like(query)
    for group in plan.groups:
        attention_backend.init_forward_metadata(group.forward_batch)
        group.forward_batch.mark_forward_metadata_ready()
        output[group.row_start : group.row_end] = block.attn.radix_attention[
            loop_index
        ](
            query[group.row_start : group.row_end],
            key[group.row_start : group.row_end],
            value[group.row_start : group.row_end],
            group.forward_batch,
            save_kv_cache=False,
        )
    return output


def _active_grouped_attention(
    block,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    active_segments: tuple[RaggedActiveSegment, ...],
    attention_backend,
    initialize_metadata: bool,
    save_kv_cache: bool,
    metadata_snapshots: dict[int, Any] | None,
) -> torch.Tensor:
    """Run grouped attention for active rows with their own loop routes."""
    output = torch.empty_like(query)
    row_offset = 0
    previous_forward_batch = None
    for active in active_segments:
        group = active.group
        row_count = group.row_count
        forward_batch_id = id(group.forward_batch)
        if metadata_snapshots is not None:
            if forward_batch_id not in metadata_snapshots:
                attention_backend.init_forward_metadata(group.forward_batch)
                group.forward_batch.mark_forward_metadata_ready()
                metadata = attention_backend.forward_metadata
                cloned_fields = {
                    field.name: (
                        getattr(metadata, field.name).detach().clone()
                        if torch.is_tensor(getattr(metadata, field.name))
                        else getattr(metadata, field.name)
                    )
                    for field in fields(metadata)
                }
                metadata_snapshots[forward_batch_id] = replace(
                    metadata,
                    **cloned_fields,
                )
            else:
                attention_backend.forward_metadata = metadata_snapshots[
                    forward_batch_id
                ]
        elif initialize_metadata and group.forward_batch is not previous_forward_batch:
            attention_backend.init_forward_metadata(group.forward_batch)
            group.forward_batch.mark_forward_metadata_ready()
        previous_forward_batch = group.forward_batch
        output[row_offset : row_offset + row_count] = block.attn.radix_attention[
            active.loop_index
        ](
            query[row_offset : row_offset + row_count],
            key[row_offset : row_offset + row_count],
            value[row_offset : row_offset + row_count],
            group.forward_batch,
            save_kv_cache=save_kv_cache,
        )
        row_offset += row_count
    return output


def execute_grouped_ragged_schedule(
    model,
    attention_backend,
    plan: RaggedForwardPlan,
    *,
    shared_metadata: bool = True,
    save_kv_cache: bool = False,
    per_group_metadata_snapshots: bool = False,
    block_graphs_by_row: dict[int, Any] | None = None,
) -> RaggedExecutionResult:
    """Execute different segment schedules with shared dense micro-steps."""
    gpt = model.gpt
    hidden_states = torch.empty(
        (plan.row_count, gpt.config.hidden_size),
        dtype=gpt.transformer.wte.weight.dtype,
        device=gpt.transformer.wte.weight.device,
    )
    for group in plan.groups:
        if group.initial_hidden_states is None:
            initial_hidden = gpt.transformer.wte(
                group.segment.input_ids
            ) + gpt.transformer.wpe(group.segment.positions)
            initial_hidden = gpt.transformer.drop(initial_hidden)
        else:
            initial_hidden = group.initial_hidden_states
        hidden_states[group.row_start : group.row_end].copy_(initial_hidden)

    with forward_context(ForwardContext(attn_backend=attention_backend)):
        forward_batches = {id(group.forward_batch) for group in plan.groups}
        shared_forward_batch = (
            plan.groups[0].forward_batch
            if shared_metadata and len(forward_batches) == 1
            else None
        )
        if shared_forward_batch is not None:
            attention_backend.init_forward_metadata(shared_forward_batch)
            shared_forward_batch.mark_forward_metadata_ready()
        metadata_snapshots = {} if per_group_metadata_snapshots else None
        for recurrence_step in compile_ragged_schedule(plan):
            active = recurrence_step.active_segments
            active_rows = torch.cat(
                [
                    torch.arange(
                        item.group.row_start,
                        item.group.row_end,
                        device=hidden_states.device,
                    )
                    for item in active
                ]
            )
            active_hidden = hidden_states.index_select(0, active_rows)
            time_values = torch.cat(
                [
                    active_hidden.new_full(
                        (item.group.row_count,),
                        item.current_time,
                        dtype=torch.float32,
                    )
                    for item in active
                ]
            )
            step_values = torch.cat(
                [
                    active_hidden.new_full(
                        (item.group.row_count,),
                        item.step_weight,
                        dtype=torch.float32,
                    )
                    for item in active
                ]
            )
            conditioning = gpt.time_embedder(time_values) + gpt.dt_embedder(
                step_values
            )
            for block_index, block in enumerate(gpt.transformer.h.blocks):
                block_graph = (
                    block_graphs_by_row.get(active_rows, {}).get(block_index)
                    if block_graphs_by_row is not None
                    else None
                )
                if block_graph is not None:
                    block_graph.hidden_input.copy_(active_hidden)
                    block_graph.conditioning_input.copy_(conditioning)
                    block_graph.pre_attention_graph.replay()
                    attention_output = _active_grouped_attention(
                        block,
                        block_graph.query,
                        block_graph.key,
                        block_graph.value,
                        active,
                        attention_backend,
                        True,
                        save_kv_cache,
                        metadata_snapshots,
                    )
                    block_graph.attention_input.copy_(attention_output)
                    block_graph.post_attention_graph.replay()
                    active_hidden = block_graph.hidden_output
                else:
                    gate_msa, gate_mlp, scale_msa, scale_mlp = (
                        block.adaLN_modulation(conditioning).chunk(4, dim=-1)
                    )
                    normed_hidden = block.norm_1(active_hidden) * (1 + scale_msa)
                    qkv = block.attn.project_qkv(normed_hidden)
                    query, key, value = qkv.split(
                        [
                            block.attn.q_size,
                            block.attn.kv_size,
                            block.attn.kv_size,
                        ],
                        dim=-1,
                    )
                    attention_output = _active_grouped_attention(
                        block,
                        query,
                        key,
                        value,
                        active,
                        attention_backend,
                        shared_forward_batch is None,
                        save_kv_cache,
                        metadata_snapshots,
                    )
                    projected = block.attn.project_output(attention_output)
                    active_hidden = active_hidden + gate_msa * projected
                    active_hidden = active_hidden + gate_mlp * block.mlp(
                        block.norm_2(active_hidden) * (1 + scale_mlp)
                    )
            hidden_states.index_copy_(0, active_rows, active_hidden)
    group_states = tuple(
        RaggedSegmentExecutionState(
            hidden_states=hidden_states[group.row_start : group.row_end],
            current_time=group.initial_time + sum(group.segment.step_schedule),
            next_loop_index=group.segment.loop_indices[-1] + 1,
        )
        for group in plan.groups
    )
    return RaggedExecutionResult(
        normalized_hidden_states=gpt.transformer.norm_f(hidden_states),
        group_states=group_states,
    )


def execute_grouped_ragged_stem(
    model,
    attention_backend,
    plan: RaggedForwardPlan,
    reference_state,
):
    """Execute dense parts jointly and attention with original contracts."""
    gpt = model.gpt
    hidden_states = gpt.transformer.wte(plan.input_ids) + gpt.transformer.wpe(
        plan.positions
    )
    hidden_states = gpt.transformer.drop(hidden_states)
    current_time = 0.0

    with forward_context(ForwardContext(attn_backend=attention_backend)):
        for loop_index, step_size in zip(
            plan.loop_indices,
            plan.step_schedule,
            strict=True,
        ):
            row_count = plan.row_count
            time_values = hidden_states.new_full(
                (row_count,), current_time, dtype=torch.float32
            )
            step_values = hidden_states.new_full(
                (row_count,), step_size, dtype=torch.float32
            )
            conditioning = gpt.time_embedder(time_values) + gpt.dt_embedder(
                step_values
            )
            for block in gpt.transformer.h.blocks:
                gate_msa, gate_mlp, scale_msa, scale_mlp = (
                    block.adaLN_modulation(conditioning).chunk(4, dim=-1)
                )
                normed_hidden = block.norm_1(hidden_states) * (1 + scale_msa)
                qkv = block.attn.project_qkv(normed_hidden)
                query, key, value = qkv.split(
                    [
                        block.attn.q_size,
                        block.attn.kv_size,
                        block.attn.kv_size,
                    ],
                    dim=-1,
                )
                attention_output = _grouped_attention(
                    block,
                    query,
                    key,
                    value,
                    plan,
                    loop_index,
                    attention_backend,
                )
                projected = block.attn.project_output(attention_output)
                hidden_states = hidden_states + gate_msa * projected
                hidden_states = hidden_states + gate_mlp * block.mlp(
                    block.norm_2(hidden_states) * (1 + scale_mlp)
                )
            current_time += step_size

    normalized_hidden = gpt.transformer.norm_f(hidden_states)
    return normalized_hidden, type(reference_state)(
        hidden_states=hidden_states,
        current_time=current_time,
        next_loop_index=plan.loop_indices[-1] + 1,
    )
