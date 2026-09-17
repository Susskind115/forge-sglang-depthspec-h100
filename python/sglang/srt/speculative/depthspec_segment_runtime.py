"""Request-level runtime state for DepthSpec forward segments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from sglang.srt.speculative.depthspec_segment_planner import (
    SegmentDescriptor,
    SegmentPhase,
)


RuntimePhase = Literal[
    "proposal_partial_stem",
    "proposal_coarse",
    "authoritative_full_stem",
    "target_continuation",
    "target_compatible_draft_extend",
    "normal_decode",
]


_NEXT_PHASE: dict[RuntimePhase, RuntimePhase] = {
    "proposal_partial_stem": "proposal_coarse",
    "proposal_coarse": "authoritative_full_stem",
    "authoritative_full_stem": "target_continuation",
    "target_continuation": "target_compatible_draft_extend",
    "target_compatible_draft_extend": "proposal_partial_stem",
    "normal_decode": "normal_decode",
}

_PREVIOUS_PHASE: dict[RuntimePhase, RuntimePhase | None] = {
    "proposal_partial_stem": "target_compatible_draft_extend",
    "proposal_coarse": "proposal_partial_stem",
    "authoritative_full_stem": "proposal_coarse",
    "target_continuation": "authoritative_full_stem",
    "target_compatible_draft_extend": "target_continuation",
    "normal_decode": "normal_decode",
}


@dataclass(frozen=True)
class SegmentPhaseSpec:
    """CPU-only execution metadata for a runtime phase."""

    loop_indices: tuple[int, ...]
    step_schedule: tuple[float, ...]
    rows_per_request: int
    attention_contract: str
    kv_namespace: str


@dataclass
class RequestRuntimeState:
    """Mutable phase state owned by one request."""

    request_id: int
    request_count: int
    phase: RuntimePhase
    round_id: int
    ready_tick: int
    lane: int | None = None
    active_segment_id: str | None = None
    completed_segment_count: int = 0


@dataclass(frozen=True)
class RuntimeSegmentBatch:
    """A no-padding heterogeneous batch ready for the ragged executor."""

    segments: tuple[SegmentDescriptor, ...]
    payload_rows: int


@dataclass
class RuntimeStatistics:
    registered_requests: int = 0
    materialized_segments: int = 0
    materialized_batches: int = 0
    completed_segments: int = 0
    rejected_dependency_checks: int = 0


@dataclass
class DepthSpecSegmentRuntime:
    """Track per-request phases and materialize dependency-ready segments.

    This class deliberately owns no GPU tensors. The worker binds each
    descriptor to its LoopSegment and ForwardBatch only after this CPU-only
    scheduler has selected the request/phase for execution.
    """

    topk: int = 3
    speculative_num_steps: int = 2
    num_draft_tokens: int = 6
    max_segments_per_batch: int = 16
    max_rows_per_batch: int = 256
    statistics: RuntimeStatistics = field(default_factory=RuntimeStatistics)
    _states: dict[int, RequestRuntimeState] = field(default_factory=dict)
    _active: dict[str, SegmentDescriptor] = field(default_factory=dict)
    _completed: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        if self.topk <= 0:
            raise ValueError("DepthSpec segment runtime requires topk > 0")
        if self.speculative_num_steps <= 0:
            raise ValueError("DepthSpec segment runtime requires steps > 0")
        if self.num_draft_tokens <= 0:
            raise ValueError("DepthSpec segment runtime requires draft tokens > 0")

    @property
    def phase_specs(self) -> dict[RuntimePhase, SegmentPhaseSpec]:
        target_loops = tuple(range(8))
        target_steps = (0.125,) * 8
        return {
            "proposal_partial_stem": SegmentPhaseSpec(
                loop_indices=(0, 1),
                step_schedule=(0.125, 0.125),
                rows_per_request=self.topk + 1,
                attention_contract="tree-custom",
                kv_namespace="target",
            ),
            "proposal_coarse": SegmentPhaseSpec(
                loop_indices=(11,),
                step_schedule=(0.75,),
                rows_per_request=self.topk,
                attention_contract="draft-chain",
                kv_namespace="draft",
            ),
            "authoritative_full_stem": SegmentPhaseSpec(
                loop_indices=(0, 1),
                step_schedule=(0.125, 0.125),
                rows_per_request=self.num_draft_tokens,
                attention_contract="tree-custom",
                kv_namespace="target",
            ),
            "target_continuation": SegmentPhaseSpec(
                loop_indices=target_loops[2:],
                step_schedule=target_steps[2:],
                rows_per_request=self.num_draft_tokens,
                attention_contract="tree-custom",
                kv_namespace="target",
            ),
            "target_compatible_draft_extend": SegmentPhaseSpec(
                loop_indices=(9, 10, 11),
                step_schedule=(0.125, 0.125, 0.75),
                rows_per_request=self.speculative_num_steps + 2,
                attention_contract="tree-custom-draft",
                kv_namespace="draft",
            ),
            "normal_decode": SegmentPhaseSpec(
                loop_indices=target_loops,
                step_schedule=target_steps,
                rows_per_request=1,
                attention_contract="causal-decode",
                kv_namespace="target",
            ),
        }

    def register_request(
        self,
        request_id: int,
        *,
        phase: RuntimePhase,
        round_id: int,
        ready_tick: int = 0,
        lane: int | None = None,
        request_count: int = 1,
    ) -> RequestRuntimeState:
        """Register a request at an already-reached phase."""
        if request_id in self._states:
            raise ValueError(f"Request {request_id} is already registered")
        if round_id < 0:
            raise ValueError(f"Request {request_id} has invalid round_id")
        if request_count <= 0:
            raise ValueError(f"Request {request_id} has invalid request_count")
        state = RequestRuntimeState(
            request_id=request_id,
            request_count=request_count,
            phase=phase,
            round_id=round_id,
            ready_tick=ready_tick,
            lane=lane,
        )
        self._states[request_id] = state
        self.statistics.registered_requests += 1
        dependency = self._dependency_for(state)
        if dependency is not None and dependency not in self._completed:
            del self._states[request_id]
            self.statistics.registered_requests -= 1
            self.statistics.rejected_dependency_checks += 1
            raise ValueError(
                f"Request {request_id} cannot start at {phase}: "
                f"missing dependency {dependency}"
            )
        return state

    def mark_seed_completed(self, segment_id: str) -> None:
        """Record a phase completed before this runtime was created."""
        if segment_id in self._completed or segment_id in self._active:
            raise ValueError(f"Seed segment {segment_id} already exists")
        self._completed.add(segment_id)

    def materialize_batch(
        self,
        *,
        tick: int | None = None,
        allowed_segment_ids: set[str] | None = None,
    ) -> RuntimeSegmentBatch | None:
        """Materialize ready requests into one dependency-valid ragged batch."""
        selected: list[SegmentDescriptor] = []
        payload_rows = 0
        for state in self._ready_states(tick=tick):
            if len(selected) >= self.max_segments_per_batch:
                break
            segment_id = self._segment_id(
                state.request_id,
                state.round_id,
                state.phase,
            )
            if (
                allowed_segment_ids is not None
                and segment_id not in allowed_segment_ids
            ):
                continue
            descriptor = self._build_descriptor(state)
            if payload_rows + descriptor.row_count > self.max_rows_per_batch:
                continue
            selected.append(descriptor)
            payload_rows += descriptor.row_count
            self._active[descriptor.segment_id] = descriptor
            state.active_segment_id = descriptor.segment_id
        if not selected:
            return None
        self.statistics.materialized_segments += len(selected)
        self.statistics.materialized_batches += 1
        return RuntimeSegmentBatch(
            segments=tuple(selected),
            payload_rows=payload_rows,
        )

    def complete_segment(self, segment_id: str) -> RuntimePhase:
        """Complete an active segment and advance its request phase."""
        descriptor = self._active.pop(segment_id, None)
        if descriptor is None:
            raise ValueError(f"Segment {segment_id} is not active")
        state = self._states.get(descriptor.request_id)
        if state is None or state.active_segment_id != segment_id:
            raise ValueError(f"Segment {segment_id} has inconsistent owner state")
        if state.phase != descriptor.phase:
            raise ValueError(f"Segment {segment_id} phase mismatch")

        self._completed.add(segment_id)
        state.active_segment_id = None
        state.completed_segment_count += 1
        state.phase = _NEXT_PHASE[state.phase]
        if state.phase in {
            "proposal_partial_stem",
            "normal_decode",
        }:
            state.round_id += 1
        self.statistics.completed_segments += 1
        return state.phase

    def ready_count(self, *, tick: int | None = None) -> int:
        return sum(1 for _ in self._ready_states(tick=tick))

    def active_count(self) -> int:
        return len(self._active)

    def state(self, request_id: int) -> RequestRuntimeState:
        try:
            return self._states[request_id]
        except KeyError as error:
            raise ValueError(f"Unknown request {request_id}") from error

    def retire_request(
        self,
        request_id: int,
        *,
        forget_history: bool = False,
    ) -> None:
        """Remove a finished request from the live scheduling state.

        ``forget_history`` is reserved for isolated diagnostic lanes whose
        dependencies will never be queried again. Real request retirement can
        keep completed segment IDs for downstream dependency checks.
        """
        state = self._states.get(request_id)
        if state is None:
            raise ValueError(f"Unknown request {request_id}")
        if state.active_segment_id is not None:
            raise ValueError(f"Request {request_id} still has an active segment")
        del self._states[request_id]
        if forget_history:
            prefix = f"r{request_id}-k"
            self._completed = {
                segment_id
                for segment_id in self._completed
                if not segment_id.startswith(prefix)
            }

    def descriptor_for_request(self, request_id: int) -> SegmentDescriptor:
        """Build the current deterministic descriptor for a registered request."""
        return self._build_descriptor(self.state(request_id))

    def _ready_states(
        self, *, tick: int | None
    ) -> list[RequestRuntimeState]:
        if tick is not None and tick < 0:
            raise ValueError("Runtime tick cannot be negative")
        states = [
            state
            for state in self._states.values()
            if state.active_segment_id is None
            and (tick is None or state.ready_tick <= tick)
        ]
        return sorted(
            states,
            key=lambda item: (
                item.ready_tick,
                item.lane if item.lane is not None else 2**31 - 1,
                item.request_id,
            ),
        )

    def _dependency_for(self, state: RequestRuntimeState) -> str | None:
        previous = _PREVIOUS_PHASE[state.phase]
        if previous is None:
            return None
        if state.phase == "proposal_partial_stem":
            if state.round_id == 0:
                return None
            return self._segment_id(
                state.request_id,
                state.round_id - 1,
                previous,
            )
        if state.phase == "normal_decode":
            if state.round_id == 0:
                return None
            return self._segment_id(
                state.request_id,
                state.round_id - 1,
                previous,
            )
        return self._segment_id(state.request_id, state.round_id, previous)

    def _build_descriptor(
        self, state: RequestRuntimeState
    ) -> SegmentDescriptor:
        dependency = self._dependency_for(state)
        if dependency is not None and dependency not in self._completed:
            self.statistics.rejected_dependency_checks += 1
            raise ValueError(
                f"Request {state.request_id} is not ready at {state.phase}: "
                f"missing {dependency}"
            )
        spec = self.phase_specs[state.phase]
        return SegmentDescriptor(
            segment_id=self._segment_id(
                state.request_id,
                state.round_id,
                state.phase,
            ),
            request_id=state.request_id,
            phase=state.phase,
            loop_indices=spec.loop_indices,
            step_schedule=spec.step_schedule,
            attention_contract=spec.attention_contract,
            rows_per_request=spec.rows_per_request,
            kv_namespace=spec.kv_namespace,
            device="cpu",
            dtype="float32",
            request_count=state.request_count,
            priority=0,
            ready_tick=state.ready_tick,
            depends_on=() if dependency is None else (dependency,),
        )

    @staticmethod
    def _segment_id(
        request_id: int,
        round_id: int,
        phase: SegmentPhase | RuntimePhase,
    ) -> str:
        return f"r{request_id}-k{round_id}-{phase}"
