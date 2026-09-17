"""Dependency-aware planner for DepthSpec forward segments."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Literal


SegmentPhase = Literal[
    "proposal_partial_stem",
    "proposal_coarse",
    "authoritative_full_stem",
    "target_continuation",
    "target_compatible_draft_extend",
    "normal_decode",
]


@dataclass(frozen=True)
class SegmentDescriptor:
    """A schedulable forward unit independent of the global request phase."""

    segment_id: str
    request_id: int
    phase: SegmentPhase | str
    loop_indices: tuple[int, ...]
    step_schedule: tuple[float, ...]
    attention_contract: str
    rows_per_request: int
    kv_namespace: str
    device: str
    dtype: str
    request_count: int = 1
    priority: int = 0
    ready_tick: int = 0
    depends_on: tuple[str, ...] = ()
    execution_rows_per_request: int | None = None

    def __post_init__(self) -> None:
        if len(self.loop_indices) != len(self.step_schedule):
            raise ValueError(
                f"Segment {self.segment_id} has mismatched loop schedule"
            )
        if self.request_count <= 0 or self.rows_per_request <= 0:
            raise ValueError(f"Segment {self.segment_id} has invalid row shape")
        if (
            self.execution_rows_per_request is not None
            and self.execution_rows_per_request < self.rows_per_request
        ):
            raise ValueError(f"Segment {self.segment_id} cannot shrink rows")

    @property
    def row_count(self) -> int:
        return self.request_count * self.rows_per_request

    @property
    def execution_width(self) -> int:
        return self.execution_rows_per_request or self.rows_per_request

    @property
    def execution_row_count(self) -> int:
        return self.request_count * self.execution_width

    @property
    def requires_padding(self) -> bool:
        return self.execution_width > self.rows_per_request

    @property
    def compatibility_key(self) -> tuple:
        return (
            self.loop_indices,
            self.step_schedule,
            self.attention_contract,
            self.execution_width,
            self.kv_namespace,
            self.device,
            self.dtype,
        )


@dataclass(frozen=True)
class PlannedSegmentBatch:
    """One physically executable batch of compatible segments."""

    compatibility_key: tuple
    segments: tuple[SegmentDescriptor, ...]
    total_rows: int
    payload_rows: int


@dataclass
class PlannerStatistics:
    submitted: int = 0
    blocked: int = 0
    unblocked: int = 0
    dispatched: int = 0
    dispatched_batches: int = 0
    compatibility_groups: int = 0


@dataclass
class _PlannerEntry:
    descriptor: SegmentDescriptor
    sequence: int


@dataclass
class ForwardSegmentPlanner:
    """Group ready segments without coupling requests to a global phase.

    The planner is intentionally deterministic and CPU-only. Runtime workers
    own GPU tensors and state transitions; this class only decides which ready
    descriptors can share a physical forward.
    """

    max_segments: int = 16
    max_rows: int = 256
    statistics: PlannerStatistics = field(default_factory=PlannerStatistics)
    _next_sequence: int = 0
    _completed: set[str] = field(default_factory=set)
    _pending: dict[str, SegmentDescriptor] = field(default_factory=dict)
    _ready_by_key: dict[tuple, deque[_PlannerEntry]] = field(
        default_factory=lambda: defaultdict(deque)
    )

    def submit(self, descriptor: SegmentDescriptor) -> None:
        if descriptor.request_count <= 0:
            raise ValueError("Cannot submit an empty segment")
        self.statistics.submitted += 1
        self._next_sequence += 1
        if not self._dependencies_ready(descriptor):
            self._pending[descriptor.segment_id] = descriptor
            self.statistics.blocked += 1
            return
        self._ready_by_key[descriptor.compatibility_key].append(
            _PlannerEntry(descriptor, self._next_sequence)
        )

    def complete(self, segment_id: str) -> None:
        self._completed.add(segment_id)
        newly_ready = []
        for pending_id, descriptor in self._pending.items():
            if self._dependencies_ready(descriptor):
                newly_ready.append(pending_id)
        for pending_id in newly_ready:
            descriptor = self._pending.pop(pending_id)
            self._next_sequence += 1
            self._ready_by_key[descriptor.compatibility_key].append(
                _PlannerEntry(descriptor, self._next_sequence)
            )
            self.statistics.unblocked += 1

    def ready_count(self) -> int:
        return sum(len(queue) for queue in self._ready_by_key.values())

    def pending_count(self) -> int:
        return len(self._pending)

    def peek_batch(self) -> PlannedSegmentBatch | None:
        self.statistics.compatibility_groups = len(self._ready_by_key)
        best: list[_PlannerEntry] | None = None
        best_score: tuple | None = None
        for queue in self._ready_by_key.values():
            if not queue:
                continue
            selected = self._select_from_queue(queue)
            if not selected:
                continue
            score = self._score(selected)
            if best_score is None or score < best_score:
                best = selected
                best_score = score
        if best is None:
            return None
        descriptors = tuple(entry.descriptor for entry in best)
        return PlannedSegmentBatch(
            compatibility_key=descriptors[0].compatibility_key,
            segments=descriptors,
            total_rows=sum(
                descriptor.execution_row_count for descriptor in descriptors
            ),
            payload_rows=sum(
                descriptor.row_count for descriptor in descriptors
            ),
        )

    def pop_batch(self) -> PlannedSegmentBatch | None:
        planned = self.peek_batch()
        if planned is None:
            return None
        selected_ids = {descriptor.segment_id for descriptor in planned.segments}
        queue = self._ready_by_key[planned.compatibility_key]
        remaining = deque(
            entry for entry in queue if entry.descriptor.segment_id not in selected_ids
        )
        if remaining:
            self._ready_by_key[planned.compatibility_key] = remaining
        else:
            del self._ready_by_key[planned.compatibility_key]
        self.statistics.dispatched += len(planned.segments)
        self.statistics.dispatched_batches += 1
        return planned

    def _dependencies_ready(self, descriptor: SegmentDescriptor) -> bool:
        return all(
            dependency in self._completed for dependency in descriptor.depends_on
        )

    def _select_from_queue(
        self, queue: deque[_PlannerEntry]
    ) -> list[_PlannerEntry]:
        selected: list[_PlannerEntry] = []
        total_rows = 0
        for entry in queue:
            if len(selected) >= self.max_segments:
                break
            row_count = entry.descriptor.execution_row_count
            if selected and total_rows + row_count > self.max_rows:
                continue
            if not selected and row_count > self.max_rows:
                continue
            selected.append(entry)
            total_rows += row_count
        return selected

    @staticmethod
    def _score(selected: list[_PlannerEntry]) -> tuple:
        first = selected[0].descriptor
        return (
            -first.priority,
            first.ready_tick,
            selected[0].sequence,
            -sum(entry.descriptor.execution_row_count for entry in selected),
            len(selected),
        )
