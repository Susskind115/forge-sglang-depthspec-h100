"""Bind CPU scheduling descriptors to executable ragged segments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sglang.srt.speculative.depthspec_ragged_execution import (
    RaggedForwardPlan,
    RaggedSegmentInput,
    build_heterogeneous_ragged_forward_plan,
)
from sglang.srt.speculative.depthspec_segment_planner import (
    SegmentDescriptor,
)
from sglang.srt.speculative.depthspec_segment_runtime import (
    DepthSpecSegmentRuntime,
)


@dataclass(frozen=True)
class BoundRuntimeSegment:
    """One scheduler descriptor and its executable GPU-side input."""

    descriptor: SegmentDescriptor
    segment_input: RaggedSegmentInput


@dataclass(frozen=True)
class BoundRuntimeBatch:
    """A dependency-valid, tensor-backed ragged execution batch."""

    segments: tuple[BoundRuntimeSegment, ...]
    forward_plan: RaggedForwardPlan
    payload_rows: int


@dataclass
class BindingStatistics:
    bound_segments: int = 0
    rejected_bindings: int = 0
    released_segments: int = 0
    materialized_batches: int = 0


@dataclass
class DepthSpecSegmentBindingStore:
    """Own the transition from CPU phase scheduling to GPU segment inputs."""

    statistics: BindingStatistics = field(default_factory=BindingStatistics)
    _bindings: dict[str, BoundRuntimeSegment] = field(default_factory=dict)

    @property
    def segment_ids(self) -> set[str]:
        return set(self._bindings)

    def bind(
        self,
        descriptor: SegmentDescriptor,
        segment_input: RaggedSegmentInput,
    ) -> BoundRuntimeSegment:
        """Validate and retain the executable input for a descriptor."""
        segment = segment_input.segment
        if descriptor.segment_id in self._bindings:
            self.statistics.rejected_bindings += 1
            raise ValueError(
                f"Segment {descriptor.segment_id} is already bound"
            )
        if descriptor.phase != segment.phase:
            self.statistics.rejected_bindings += 1
            raise ValueError(
                f"Segment {descriptor.segment_id} phase mismatch: "
                f"{descriptor.phase} != {segment.phase}"
            )
        if descriptor.loop_indices != tuple(segment.loop_indices):
            self.statistics.rejected_bindings += 1
            raise ValueError(
                f"Segment {descriptor.segment_id} loop schedule mismatch"
            )
        if descriptor.step_schedule != tuple(segment.step_schedule):
            self.statistics.rejected_bindings += 1
            raise ValueError(
                f"Segment {descriptor.segment_id} step schedule mismatch"
            )
        if descriptor.row_count != int(segment.input_ids.shape[0]):
            self.statistics.rejected_bindings += 1
            raise ValueError(
                f"Segment {descriptor.segment_id} row count mismatch"
            )
        if segment.req_pool_indices is None:
            self.statistics.rejected_bindings += 1
            raise ValueError(
                f"Segment {descriptor.segment_id} requires req_pool_indices"
            )
        binding = BoundRuntimeSegment(
            descriptor=descriptor,
            segment_input=segment_input,
        )
        self._bindings[descriptor.segment_id] = binding
        self.statistics.bound_segments += 1
        return binding

    def get(
        self, descriptors: tuple[SegmentDescriptor, ...]
    ) -> list[BoundRuntimeSegment]:
        result = []
        for descriptor in descriptors:
            binding = self._bindings.get(descriptor.segment_id)
            if binding is None:
                raise ValueError(
                    f"Scheduled segment {descriptor.segment_id} has no binding"
                )
            result.append(binding)
        return result

    def release(self, segment_id: str) -> BoundRuntimeSegment:
        """Remove a binding after its output has been consumed."""
        binding = self._bindings.pop(segment_id, None)
        if binding is None:
            raise ValueError(f"Cannot release unbound segment {segment_id}")
        self.statistics.released_segments += 1
        return binding

    def ready_count(self) -> int:
        return len(self._bindings)


@dataclass(frozen=True)
class DepthSpecBoundSegmentScheduler:
    """Combine phase dependencies, bindings, and ragged plan construction."""

    runtime: DepthSpecSegmentRuntime
    bindings: DepthSpecSegmentBindingStore

    def materialize_batch(
        self, *, tick: int | None = None
    ) -> BoundRuntimeBatch | None:
        batch = self.runtime.materialize_batch(
            tick=tick,
            allowed_segment_ids=self.bindings.segment_ids,
        )
        if batch is None:
            return None
        bound_segments = tuple(self.bindings.get(batch.segments))
        forward_plan = build_heterogeneous_ragged_forward_plan(
            [item.segment_input for item in bound_segments]
        )
        self.bindings.statistics.materialized_batches += 1
        return BoundRuntimeBatch(
            segments=bound_segments,
            forward_plan=forward_plan,
            payload_rows=batch.payload_rows,
        )

    def complete_batch(
        self,
        batch: BoundRuntimeBatch,
    ) -> dict[str, Any]:
        """Advance runtime states and release consumed bindings."""
        next_phases = {}
        for item in batch.segments:
            next_phases[item.descriptor.segment_id] = (
                self.runtime.complete_segment(item.descriptor.segment_id)
            )
            self.bindings.release(item.descriptor.segment_id)
        return next_phases
