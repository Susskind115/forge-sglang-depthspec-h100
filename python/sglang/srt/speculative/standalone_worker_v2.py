import gc
import json
import logging
import os
import time
from contextlib import contextmanager
import copy as copy_module
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from sglang.srt.distributed.parallel_state_wrapper import ParallelState
from sglang.srt.layers.moe.utils import (
    draft_model_build_scope,
    speculative_moe_backend_context,
)
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.runtime_context import get_parallel, get_schedule
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.adaptive_runtime_state import (
    AdaptiveController,
)
from sglang.srt.speculative.base_spec_worker import (
    BaseSpecWorker,
    EagleDraftWorkerBase,
)
from sglang.srt.speculative.depthspec_runtime_config import (
    DepthSpecRuntimeConfig,
    activate_depthspec_runtime_config,
    attach_depthspec_runtime_config,
    get_active_depthspec_runtime_config,
)
from sglang.srt.speculative.depthspec_ragged_execution import (
    RaggedSegmentInput,
    build_ragged_forward_plan,
    build_heterogeneous_ragged_forward_plan,
    compile_ragged_schedule,
    execute_grouped_ragged_stem,
    execute_grouped_ragged_schedule,
)
from sglang.srt.speculative.depthspec_segment_binding import (
    DepthSpecBoundSegmentScheduler,
    DepthSpecSegmentBindingStore,
)
from sglang.srt.speculative.depthspec_segment_runtime import (
    DepthSpecSegmentRuntime,
)
from sglang.srt.speculative.eagle_utils import default_tree_mask_mode
from sglang.srt.speculative.eagle_worker_v2 import EagleDraftWorker, EAGLEWorkerV2
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_utils import draft_tp_context, get_plan_stream
from sglang.srt.utils import empty_context, get_bool_env_var, is_cuda

if is_cuda():
    from sgl_kernel import segment_packbits  # noqa: F401

logger = logging.getLogger(__name__)
SGLANG_RETURN_ORIGINAL_LOGPROB = get_bool_env_var("SGLANG_RETURN_ORIGINAL_LOGPROB")


def _localize_tree_accept_index(
    accept_index: torch.Tensor,
    tree_width: int,
) -> torch.Tensor:
    """Convert flattened batch-tree rows to per-request tree row indices."""
    request_offsets = (
        torch.arange(
            accept_index.shape[0],
            device=accept_index.device,
            dtype=accept_index.dtype,
        )
        * tree_width
    )
    localized = accept_index - request_offsets[:, None]
    return torch.where(accept_index >= 0, localized, accept_index)


def _append_stage_timing(stage: str, batch_size: int, elapsed: float) -> None:
    """Optionally record per-stage wall time for DepthSpec experiments."""
    timing_path = get_active_depthspec_runtime_config().diagnostics.get(
        "stage_timing_path"
    )
    if not timing_path:
        return
    row = {"stage": stage, "batch_size": batch_size, "elapsed_s": elapsed}
    Path(timing_path).parent.mkdir(parents=True, exist_ok=True)
    with Path(timing_path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row) + "\n")


def _append_segment_trace(
    *,
    phase: str,
    request_count: int,
    row_count: int,
    loop_indices: tuple[int, ...],
    step_schedule: tuple[float, ...],
    attention_contract: str,
    graph_hit: bool,
    elapsed_s: float,
    seq_lens_sum: int | None = None,
    merged_segment_count: int = 1,
    merged_phases: tuple[str, ...] | None = None,
) -> None:
    """Record an execution-plan row without synchronizing tensor values."""
    trace_path = get_active_depthspec_runtime_config().diagnostics.get(
        "segment_trace_path"
    )
    if not trace_path:
        return
    row = {
        "captured_unix_s": time.time(),
        "phase": phase,
        "request_count": request_count,
        "row_count": row_count,
        "loop_indices": list(loop_indices),
        "step_schedule": list(step_schedule),
        "attention_contract": attention_contract,
        "graph_hit": graph_hit,
        "elapsed_s": elapsed_s,
        "seq_lens_sum": seq_lens_sum,
        "merged_segment_count": merged_segment_count,
        "merged_phases": list(merged_phases) if merged_phases is not None else None,
    }
    output = Path(trace_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row) + "\n")


@dataclass
class LoopSegment:
    """A row-level LoopFormer forward request for one execution phase."""

    input_ids: torch.Tensor
    positions: torch.Tensor
    out_cache_loc: torch.Tensor
    seq_lens: torch.Tensor
    seq_lens_cpu: torch.Tensor | None
    seq_lens_sum: int
    spec_info: Any
    req_pool_indices: torch.Tensor | None = None
    phase: str = "missing_stem"
    loop_indices: tuple[int, ...] = (0, 1)
    step_schedule: tuple[float, ...] = (0.125, 0.125)
    attention_contract: str = "tree-custom"
    ragged_qo_indptr: tuple[int, ...] | None = None

    @property
    def request_count(self) -> int:
        return int(self.seq_lens.shape[0])

    @property
    def row_count(self) -> int:
        return int(self.input_ids.shape[0])


class LoopSegmentExecutor:
    """Execute a loop segment while preserving the caller's schedule state."""

    def __init__(self, worker, batch):
        self.worker = worker
        self.batch = batch

    @staticmethod
    def _compatibility_key(segment: LoopSegment):
        spec_info = segment.spec_info
        return (
            segment.loop_indices,
            segment.step_schedule,
            segment.attention_contract,
            int(getattr(spec_info, "num_tokens_per_req", spec_info.draft_token_num)),
            int(
                getattr(
                    spec_info,
                    "num_kv_tokens_per_req",
                    spec_info.draft_token_num,
                )
            ),
            int(spec_info.spec_steps),
            int(spec_info.topk),
            segment.input_ids.device,
            segment.input_ids.dtype,
            segment.positions.dtype,
            getattr(spec_info, "capture_hidden_mode", None),
        )

    @staticmethod
    def _merge_segments(segments: list[LoopSegment]) -> LoopSegment:
        """Concatenate compatible segments into one fixed-width verify batch."""
        if not segments:
            raise ValueError("Cannot merge an empty LoopSegment list")
        keys = {LoopSegmentExecutor._compatibility_key(segment) for segment in segments}
        if len(keys) != 1:
            raise ValueError(
                "Cannot merge LoopSegments with incompatible execution contracts"
            )
        first = segments[0]
        if any(
            getattr(segment.spec_info, "retrieve_cum_len", None) is not None
            for segment in segments
        ):
            raise ValueError("Merged LoopSegments do not support retrieve_cum_len")
        if any(segment.req_pool_indices is None for segment in segments):
            raise ValueError("Merged LoopSegments require req_pool_indices")

        spec_infos = [segment.spec_info for segment in segments]
        first_spec = spec_infos[0]
        merged_spec = replace(
            first_spec,
            draft_token=torch.cat([item.draft_token for item in spec_infos]),
            custom_mask=torch.cat([item.custom_mask for item in spec_infos]),
            positions=torch.cat([item.positions for item in spec_infos]),
            retrieve_index=torch.cat(
                [item.retrieve_index for item in spec_infos], dim=0
            ),
            retrieve_next_token=torch.cat(
                [item.retrieve_next_token for item in spec_infos], dim=0
            ),
            retrieve_next_sibling=torch.cat(
                [item.retrieve_next_sibling for item in spec_infos], dim=0
            ),
            seq_lens_sum=sum(item.seq_lens_sum for item in segments),
            seq_lens_cpu=torch.cat([item.seq_lens_cpu for item in segments]),
        )
        return LoopSegment(
            input_ids=torch.cat([item.input_ids for item in segments]),
            positions=torch.cat([item.positions for item in segments]),
            out_cache_loc=torch.cat([item.out_cache_loc for item in segments]),
            seq_lens=torch.cat([item.seq_lens for item in segments]),
            seq_lens_cpu=torch.cat([item.seq_lens_cpu for item in segments]),
            seq_lens_sum=merged_spec.seq_lens_sum,
            spec_info=merged_spec,
            req_pool_indices=torch.cat(
                [item.req_pool_indices for item in segments]
            ),
            phase="+".join(item.phase for item in segments),
            loop_indices=first.loop_indices,
            step_schedule=first.step_schedule,
            attention_contract=first.attention_contract,
        )

    def execute_many(self, segments: list[LoopSegment]):
        """Execute compatible segments in one physical LoopFormer forward."""
        segment = self._merge_segments(segments)
        return self._execute_merged_segment(
            segment,
            source_segments=segments,
            allow_piecewise_graph=True,
        )

    @staticmethod
    def _ragged_compatibility_key(segment: LoopSegment):
        spec_info = segment.spec_info
        return (
            segment.loop_indices,
            segment.step_schedule,
            segment.attention_contract,
            segment.input_ids.device,
            segment.input_ids.dtype,
            segment.positions.dtype,
            getattr(spec_info, "capture_hidden_mode", None),
        )

    @staticmethod
    def _merge_ragged_segments(segments: list[LoopSegment]) -> LoopSegment:
        """Flatten unequal-width tree segments without dummy rows."""
        from sglang.srt.speculative.eagle_info import EagleVerifyInput

        if not segments:
            raise ValueError("Cannot merge an empty ragged LoopSegment list")
        keys = {
            LoopSegmentExecutor._ragged_compatibility_key(segment)
            for segment in segments
        }
        if len(keys) != 1:
            raise ValueError("Ragged LoopSegments have incompatible schedules")
        if any(segment.req_pool_indices is None for segment in segments):
            raise ValueError("Ragged LoopSegments require req_pool_indices")
        if any(
            getattr(segment.spec_info, "retrieve_cum_len", None) is not None
            for segment in segments
        ):
            raise ValueError("Ragged LoopSegments do not support retrieve_cum_len")

        first = segments[0]
        input_ids = []
        positions = []
        out_cache_locs = []
        seq_lens = []
        seq_lens_cpu = []
        req_pool_indices = []
        mask_rows = []
        row_offset = 0
        qo_indptr = [0]
        max_kv_width = max(
            int(
                getattr(
                    segment.spec_info,
                    "num_kv_tokens_per_req",
                    segment.spec_info.draft_token_num,
                )
            )
            for segment in segments
        )
        for segment in segments:
            spec_info = segment.spec_info
            query_width = int(
                getattr(
                    spec_info,
                    "num_tokens_per_req",
                    spec_info.draft_token_num,
                )
            )
            kv_width = int(
                getattr(
                    spec_info,
                    "num_kv_tokens_per_req",
                    spec_info.draft_token_num,
                )
            )
            request_count = int(segment.seq_lens.shape[0])
            if int(segment.input_ids.numel()) != request_count * query_width:
                raise ValueError(
                    f"Ragged segment {segment.phase} has invalid query row count"
                )

            input_ids.append(segment.input_ids)
            positions.append(segment.positions)
            out_cache_locs.append(segment.out_cache_loc)
            seq_lens.append(
                (
                    segment.seq_lens + max_kv_width
                ).repeat_interleave(query_width)
            )
            if segment.seq_lens_cpu is None:
                seq_lens_cpu.append(
                    (segment.seq_lens.cpu() + max_kv_width).repeat_interleave(
                        query_width
                    )
                )
            else:
                seq_lens_cpu.append(
                    (
                        segment.seq_lens_cpu + max_kv_width
                    ).repeat_interleave(query_width)
                )
            req_pool_indices.append(
                segment.req_pool_indices.repeat_interleave(query_width)
            )

            custom_mask = spec_info.custom_mask
            mask_offset = 0
            for sequence_length_tensor in segment.seq_lens:
                sequence_length = int(sequence_length_tensor.item())
                request_mask_size = query_width * (sequence_length + kv_width)
                request_mask = custom_mask[
                    mask_offset : mask_offset + request_mask_size
                ]
                for row_index in range(query_width):
                    row_start = row_index * (sequence_length + kv_width)
                    prefix = request_mask[
                        row_start : row_start + sequence_length
                    ]
                    tree = request_mask[
                        row_start + sequence_length : row_start
                        + sequence_length
                        + kv_width
                    ]
                    padding = custom_mask.new_zeros(max_kv_width - kv_width)
                    mask_rows.append(torch.cat((prefix, tree, padding)))
                mask_offset += request_mask_size
            if mask_offset != int(custom_mask.numel()):
                raise ValueError(
                    f"Ragged segment {segment.phase} mask layout mismatch"
                )
            row_offset += int(segment.input_ids.numel())
            qo_indptr.append(row_offset)

        flat_input_ids = torch.cat(input_ids)
        total_rows = int(flat_input_ids.numel())
        flat_seq_lens = torch.cat(seq_lens)
        flat_seq_lens_cpu = torch.cat(seq_lens_cpu)
        flat_req_pool_indices = torch.cat(req_pool_indices)
        flat_positions = torch.cat(positions)
        flat_out_cache_locs = torch.cat(out_cache_locs)
        flat_mask = torch.cat(mask_rows)
        retrieve_index = torch.arange(
            total_rows, device=flat_input_ids.device, dtype=torch.int64
        ).unsqueeze(1)
        retrieve_empty = torch.full(
            (total_rows, 1), -1, device=flat_input_ids.device, dtype=torch.int64
        )
        merged_spec = EagleVerifyInput(
            draft_token=flat_input_ids,
            custom_mask=flat_mask,
            positions=flat_positions,
            retrieve_index=retrieve_index,
            retrieve_next_token=retrieve_empty,
            retrieve_next_sibling=retrieve_empty,
            retrieve_cum_len=None,
            spec_steps=1,
            topk=1,
            draft_token_num=1,
            num_tokens_per_req=1,
            num_kv_tokens_per_req=max_kv_width,
            depthspec_preserve_full_tree_kv=True,
            capture_hidden_mode=first.spec_info.capture_hidden_mode,
            seq_lens_sum=int(flat_seq_lens.sum().item()),
            seq_lens_cpu=flat_seq_lens_cpu,
        )
        return LoopSegment(
            input_ids=flat_input_ids,
            positions=flat_positions,
            out_cache_loc=flat_out_cache_locs,
            seq_lens=flat_seq_lens,
            seq_lens_cpu=flat_seq_lens_cpu,
            seq_lens_sum=merged_spec.seq_lens_sum,
            spec_info=merged_spec,
            req_pool_indices=flat_req_pool_indices,
            phase="+".join(segment.phase for segment in segments),
            loop_indices=first.loop_indices,
            step_schedule=first.step_schedule,
            attention_contract="ragged-tree-custom",
            ragged_qo_indptr=tuple(qo_indptr),
        )

    def execute_ragged_many(self, segments: list[LoopSegment]):
        """Execute unequal-width segments with one query row per request."""
        segment = self._merge_ragged_segments(segments)
        return self._execute_merged_segment(
            segment,
            source_segments=segments,
            allow_piecewise_graph=False,
        )

    def _execute_merged_segment(
        self,
        segment: LoopSegment,
        *,
        source_segments: list[LoopSegment],
        allow_piecewise_graph: bool,
    ):
        from sglang.srt.model_executor.forward_batch_info import (
            CaptureHiddenMode,
            ForwardBatch,
        )
        from sglang.srt.model_executor.forward_context import (
            ForwardContext,
            forward_context,
        )

        target_runner = self.worker.target_worker.model_runner
        target_model = target_runner.model
        saved_state = {
            name: getattr(self.batch, name, None)
            for name in (
                "input_ids",
                "out_cache_loc",
                "out_cache_loc_dsv4",
                "req_pool_indices",
                "seq_lens",
                "seq_lens_cpu",
                "seq_lens_sum",
                "forward_mode",
                "spec_info",
            )
        }
        self.batch.input_ids = segment.input_ids
        self.batch.out_cache_loc = segment.out_cache_loc
        if segment.req_pool_indices is not None:
            self.batch.req_pool_indices = segment.req_pool_indices
        self.batch.seq_lens = segment.seq_lens
        self.batch.seq_lens_cpu = segment.seq_lens_cpu
        self.batch.seq_lens_sum = segment.seq_lens_sum
        self.batch.spec_info = segment.spec_info
        from sglang.srt.model_executor.forward_batch_info import ForwardMode

        self.batch.forward_mode = ForwardMode.TARGET_VERIFY
        started = time.perf_counter()
        old_kv_indptr = target_runner.attn_backend.kv_indptr
        old_mask_indptr = target_runner.attn_backend.mask_indptr
        if segment.ragged_qo_indptr is not None:
            physical_request_count = int(segment.seq_lens.shape[0])
            if old_kv_indptr.numel() < physical_request_count + 1:
                target_runner.attn_backend.kv_indptr = torch.zeros(
                    physical_request_count + 1,
                    dtype=old_kv_indptr.dtype,
                    device=old_kv_indptr.device,
                )
            if old_mask_indptr.numel() < physical_request_count + 1:
                target_runner.attn_backend.mask_indptr = torch.zeros(
                    physical_request_count + 1,
                    dtype=old_mask_indptr.dtype,
                    device=old_mask_indptr.device,
                )
        try:
            forward_batch = ForwardBatch.init_new(
                self.batch,
                target_runner,
                capture_hidden_mode=CaptureHiddenMode.NULL,
                return_hidden_states_before_norm=False,
            )
            target_runner.attn_backend.init_forward_metadata(forward_batch)
            forward_batch.mark_forward_metadata_ready()
            with forward_context(
                ForwardContext(attn_backend=target_runner.attn_backend)
            ):
                draft_worker = getattr(self.worker, "_draft_worker", self.worker)
                piecewise_runners = getattr(
                    draft_worker,
                    "canonical_stem_piecewise_graph_runners",
                    {},
                )
                use_missing_graph = get_active_depthspec_runtime_config().diagnostics.get(
                    "canonical_missing_stem_piecewise_graph",
                    get_active_depthspec_runtime_config().canonical_stem_piecewise_graph,
                )
                piecewise_runner = (
                    piecewise_runners.get(int(forward_batch.input_ids.shape[0]))
                    if use_missing_graph and allow_piecewise_graph
                    else None
                )
                if piecewise_runner is None:
                    _, state = target_model.gpt.forward_branch(
                        forward_batch.input_ids,
                        forward_batch.positions,
                        forward_batch,
                        loop_indices=segment.loop_indices,
                        step_schedule=segment.step_schedule,
                        return_state=True,
                    )
                else:
                    _, state = piecewise_runner.execute(
                        forward_batch.input_ids,
                        forward_batch.positions,
                        forward_batch,
                        return_state=True,
                    )
            _append_segment_trace(
                phase=segment.phase,
                request_count=segment.request_count,
                row_count=segment.row_count,
                loop_indices=segment.loop_indices,
                step_schedule=segment.step_schedule,
                attention_contract=segment.attention_contract,
                graph_hit=piecewise_runner is not None,
                elapsed_s=time.perf_counter() - started,
                seq_lens_sum=segment.seq_lens_sum,
                merged_segment_count=len(source_segments),
                merged_phases=tuple(item.phase for item in source_segments),
            )
            return state, forward_batch
        finally:
            target_runner.attn_backend.kv_indptr = old_kv_indptr
            target_runner.attn_backend.mask_indptr = old_mask_indptr
            for name, value in saved_state.items():
                setattr(self.batch, name, value)

    def execute(self, segment: LoopSegment):
        state, forward_batch = self.execute_many([segment])
        return state, forward_batch


class StandaloneDraftWorker(EagleDraftWorker):
    """Custom EagleDraftWorker that doesn't share embeddings/lm_head with target model."""

    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        ps: ParallelState,
        nccl_port: int,
        target_worker: TpModelWorker,
    ):
        EagleDraftWorkerBase.__init__(self)

        # copy args
        self.server_args = server_args
        self.gpu_id = gpu_id
        self.ps = ps
        self.nccl_port = nccl_port
        self.target_worker = target_worker
        target_model = target_worker.model_runner.model
        architectures = getattr(target_model.config, "architectures", None) or []
        architecture = architectures[0] if architectures else None
        self.runtime_config = DepthSpecRuntimeConfig.resolve(
            server_args,
            architecture,
        )
        activate_depthspec_runtime_config(self.runtime_config)
        for owner in (
            target_worker,
            target_worker.model_runner,
            target_model,
            target_model.gpt,
        ):
            attach_depthspec_runtime_config(owner, self.runtime_config)
        logger.info(
            "DepthSpec runtime config: %s",
            json.dumps(self.runtime_config.to_log_dict(), sort_keys=True),
        )

        # Args for easy access
        self.device = server_args.device
        self.topk = server_args.speculative_eagle_topk
        self.speculative_num_steps = server_args.speculative_num_steps
        self.speculative_num_draft_tokens = server_args.speculative_num_draft_tokens
        self.speculative_algorithm = SpeculativeAlgorithm.from_string(
            server_args.speculative_algorithm
        )

        self._rebuild_topk1_chain_buffers()

        # Set constant
        from sglang.srt.speculative.eagle_info import EagleDraftInput

        EagleDraftInput.ALLOC_LEN_PER_DECODE = max(
            self.speculative_num_steps * self.topk, self.speculative_num_draft_tokens
        )

        self.unified_runner = self.runtime_config.unified_runner
        if self.unified_runner:
            self.draft_worker = target_worker
        else:
            # Load draft model weights only. The standalone draft is a real model
            # whose MoE gates run during construction; the scope routes their
            # fusion decision to the speculative leaf (it does not swap
            # runner_backend — the draft's forwards run outside that context).
            with empty_context(), draft_model_build_scope():
                self.draft_worker = TpModelWorker(
                    server_args=server_args,
                    gpu_id=gpu_id,
                    # spec workers don't support pipeline parallelism
                    ps=replace(ps, pp_rank=0),
                    nccl_port=nccl_port,
                    is_draft_worker=True,
                    # The draft runs at absolute target positions.
                    context_length=target_worker.model_runner.model_config.context_len,
                )

        # Alias for better readability
        self.draft_runner = self.draft_worker.model_runner
        attach_depthspec_runtime_config(self.draft_worker, self.runtime_config)
        attach_depthspec_runtime_config(self.draft_runner, self.runtime_config)
        attach_depthspec_runtime_config(self.draft_runner.model, self.runtime_config)
        attach_depthspec_runtime_config(
            self.draft_runner.model.gpt,
            self.runtime_config,
        )
        if not self.unified_runner:
            self._share_loopformer_weights_if_supported()
        self.draft_tp_context = (
            draft_tp_context if get_parallel().enable_dp_attention else empty_context
        )
        self.tree_mask_mode = default_tree_mask_mode()
        self.plan_stream, self.plan_stream_ctx = get_plan_stream(self.device)
        # draft_forward reads this (set in EagleDraftWorker.__init__, skipped here).
        self.index_share_for_mtp_iteration = (
            getattr(
                self.draft_runner.model_config.hf_config,
                "index_share_for_mtp_iteration",
                False,
            )
            and self.topk == 1
        )
        self.dsa_index_topk = None
        self.seed_dsa_topk_from_draft_extend = False
        self.dsa_extend_topk_buf = None
        self.hot_token_id = None
        self.cuda_graph_runner = None
        self.cuda_graph_runner_for_draft_extend = None
        self.active_path_graph_runner = None
        self.canonical_proposal_graph_runner = None
        self.target_compatible_piecewise_graph_runner = None
        self.target_compatible_piecewise_graph_runners = {}
        self.target_continuation_piecewise_graph_runner = None
        self.target_continuation_piecewise_graph_runners = {}
        self.canonical_stem_piecewise_graph_runners = {}
        self.partial_stem_position_offsets: dict[int, torch.Tensor] = {}
        self.partial_stem_retrieve_templates: dict[
            tuple[int, int], tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ] = {}
        self.partial_stem_tree_masks: dict[int, torch.Tensor] = {}
        self.partial_stem_masks_by_length: dict[int, torch.Tensor] = {}
        self.partial_stem_mask_buffers: dict[tuple[int, int], torch.Tensor] = {}
        self.target_attn_backend = None
        self.loop_segment_executor_enabled = self.runtime_config.loop_segment_executor

    def _partial_stem_position_offsets(
        self,
        topk: int,
        device: torch.device,
    ) -> torch.Tensor:
        key = topk
        cached = self.partial_stem_position_offsets.get(key)
        if cached is None or cached.device != device:
            cached = torch.tensor(
                [0] + [1] * topk, device=device, dtype=torch.int64
            )
            self.partial_stem_position_offsets[key] = cached
        return cached

    def _partial_stem_mask(
        self,
        sequence_length: int,
        width: int,
        device: torch.device,
    ) -> torch.Tensor:
        cached = self.partial_stem_masks_by_length.get(sequence_length)
        if cached is not None and cached.device == device:
            return cached
        tree_mask = self.partial_stem_tree_masks.get(width)
        if tree_mask is None or tree_mask.device != device:
            tree_mask = torch.eye(width, dtype=torch.bool, device=device)
            tree_mask[1:, 0] = True
            self.partial_stem_tree_masks[width] = tree_mask
        mask = torch.ones(
            width * (sequence_length + width),
            dtype=torch.bool,
            device=device,
        )
        mask[width * sequence_length :].view(width, width).copy_(tree_mask)
        self.partial_stem_masks_by_length[sequence_length] = mask
        return mask

    def _partial_stem_retrieve_template(
        self,
        batch_size: int,
        topk: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        key = (batch_size, topk)
        cached = self.partial_stem_retrieve_templates.get(key)
        if cached is not None and cached[0].device == device:
            return cached
        width = topk + 1
        retrieve_index = (
            torch.arange(width, device=device, dtype=torch.int64)
            .unsqueeze(0)
            .expand(batch_size, -1)
            .contiguous()
        )
        retrieve_next_token = torch.full(
            (batch_size, width), -1, device=device, dtype=torch.int64
        )
        retrieve_next_sibling = torch.full(
            (batch_size, width), -1, device=device, dtype=torch.int64
        )
        if topk > 0:
            retrieve_next_token[:, 0] = 1
            first_level_siblings = torch.full(
                (topk,), -1, device=device, dtype=torch.int64
            )
            first_level_siblings[:-1] = torch.arange(
                2, width, device=device, dtype=torch.int64
            )
            retrieve_next_sibling[:, 1:width] = first_level_siblings
        template = (retrieve_index, retrieve_next_token, retrieve_next_sibling)
        self.partial_stem_retrieve_templates[key] = template
        return template

    def _partial_stem_persistent_mask(
        self,
        sequence_lengths: list[int],
        width: int,
        device: torch.device,
    ) -> torch.Tensor:
        batch_size = len(sequence_lengths)
        key = (batch_size, width)
        buffer = self.partial_stem_mask_buffers.get(key)
        context_length = self.target_worker.model_runner.model_config.context_len
        max_cells = width * batch_size * (int(context_length) + width)
        if buffer is None or buffer.device != device or buffer.numel() < max_cells:
            buffer = torch.empty(max_cells, dtype=torch.bool, device=device)
            self.partial_stem_mask_buffers[key] = buffer

        tree_mask = self.partial_stem_tree_masks.get(width)
        if tree_mask is None or tree_mask.device != device:
            tree_mask = torch.eye(width, dtype=torch.bool, device=device)
            tree_mask[1:, 0] = True
            self.partial_stem_tree_masks[width] = tree_mask

        buffer.fill_(True)
        offset = 0
        for sequence_length in sequence_lengths:
            tree_start = offset + width * sequence_length
            buffer[tree_start : tree_start + width * width].view(
                width, width
            ).copy_(tree_mask)
            offset += width * (sequence_length + width)
        return buffer[:offset]

    @contextmanager
    def _use_unified_draft_forward(
        self,
        prefix_hidden_states=None,
        use_prefix_handoff=False,
        forward_purpose="other",
    ):
        """Temporarily route the shared runner through draft attention state."""
        target_runner = self.target_worker.model_runner
        old_attention_backend = target_runner.attn_backend
        if self.target_attn_backend is None:
            self.target_attn_backend = old_attention_backend
        target_runner.attn_backend = self.draft_extend_attn_backend
        target_model = target_runner.model
        old_forward_kind = getattr(target_model, "depthspec_forward_kind", "target")
        if prefix_hidden_states is None or not use_prefix_handoff:
            target_model.depthspec_forward_kind = "draft"
            if prefix_hidden_states is not None:
                target_model.depthspec_prefix_handoff_hidden_states = (
                    prefix_hidden_states
                )
        else:
            target_model.depthspec_forward_kind = "draft_coarse_handoff"
            target_model.depthspec_prefix_handoff_hidden_states = prefix_hidden_states
        last_loop = target_model.gpt.loop_num - 1
        use_target_prefix = (
            forward_purpose == "proposal"
            and self.runtime_config.proposal_stem_artifacts
        )
        target_model.depthspec_draft_loop_indices = (
            (0, 1, last_loop)
            if use_target_prefix
            else (last_loop - 2, last_loop - 1, last_loop)
        )
        target_model.depthspec_forward_purpose = forward_purpose
        target_model.gpt.depthspec_forward_purpose = forward_purpose
        try:
            yield
        finally:
            target_model.depthspec_forward_kind = old_forward_kind
            if hasattr(target_model, "depthspec_prefix_handoff_hidden_states"):
                del target_model.depthspec_prefix_handoff_hidden_states
            if hasattr(target_model, "depthspec_prefix_accept_lens"):
                del target_model.depthspec_prefix_accept_lens
            for attribute_name in (
                "depthspec_prefix_accept_index",
                "depthspec_prefix_candidate_tokens",
                "depthspec_prefix_candidate_positions",
            ):
                if hasattr(target_model, attribute_name):
                    delattr(target_model, attribute_name)
            if hasattr(target_model, "depthspec_draft_loop_indices"):
                del target_model.depthspec_draft_loop_indices
            if hasattr(target_model, "depthspec_forward_purpose"):
                del target_model.depthspec_forward_purpose
            if hasattr(target_model.gpt, "depthspec_forward_purpose"):
                del target_model.gpt.depthspec_forward_purpose
            target_runner.attn_backend = old_attention_backend

    def _share_loopformer_weights_if_supported(self) -> None:
        """Share one LoopFormer checkpoint between target and virtual draft.

        DepthSpec uses the same physical weights at different recurrence
        depths. Sharing parameter storage avoids retaining the duplicate
        checkpoint after both model runners have been initialized.
        """
        target_model = self.target_worker.model_runner.model
        draft_model = self.draft_runner.model
        target_arch = target_model.config.architectures[0]
        draft_arch = draft_model.config.architectures[0]
        if target_arch != "LoopFormerSGLangForCausalLM" or draft_arch != target_arch:
            return

        target_parameters = dict(target_model.named_parameters())
        draft_parameters = dict(draft_model.named_parameters())
        if set(target_parameters) != set(draft_parameters):
            raise RuntimeError(
                "DepthSpec weight sharing requires identical parameter names, "
                f"got target={sorted(target_parameters)}, "
                f"draft={sorted(draft_parameters)}"
            )

        shared_count = 0
        with torch.no_grad():
            for name, draft_parameter in draft_parameters.items():
                target_parameter = target_parameters[name]
                if draft_parameter.shape != target_parameter.shape:
                    raise RuntimeError(
                        f"DepthSpec weight shape mismatch for {name}: "
                        f"{tuple(draft_parameter.shape)} vs "
                        f"{tuple(target_parameter.shape)}"
                    )
                draft_parameter.data = target_parameter.data
                shared_count += 1

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info(
            "DepthSpec shared %d LoopFormer parameter tensors with the target model",
            shared_count,
        )

    def _capture_cuda_graphs(self):
        if self.runtime_config.canonical_proposal_graph:
            from sglang.srt.speculative.eagle_draft_cuda_graph_runner import (
                EAGLEDraftCudaGraphRunner,
            )

            target_model = self.target_worker.model_runner.model
            graph_max_batch_size = self.runtime_config.canonical_graph_max_batch_size
            self.canonical_proposal_graph_runner = EAGLEDraftCudaGraphRunner(
                self,
                draft_attn_backend=self.draft_attn_backend,
                speculative_num_steps=self.speculative_num_steps,
                capture_forward_kind="draft_coarse_handoff",
                prefix_handoff_hidden_size=int(target_model.config.hidden_size),
            )
            if self.runtime_config.target_compatible_piecewise_graph:
                rows_per_request = self.speculative_num_steps + 2
                for batch_size in range(1, graph_max_batch_size + 1):
                    row_count = batch_size * rows_per_request
                    self.target_compatible_piecewise_graph_runners[row_count] = (
                        target_model.gpt.build_piecewise_graph_runner(
                            row_count=row_count,
                            loop_indices=(
                                target_model.gpt.loop_num - 3,
                                target_model.gpt.loop_num - 2,
                                target_model.gpt.loop_num - 1,
                            ),
                            step_schedule=(0.125, 0.125, 0.75),
                        )
                    )
                self.target_compatible_piecewise_graph_runner = (
                    self.target_compatible_piecewise_graph_runners.get(
                        rows_per_request
                    )
                )
            if self.runtime_config.target_continuation_piecewise_graph:
                continuation_loop_indices = tuple(
                    range(2, target_model.gpt.target_loop_count)
                )
                continuation_schedule = tuple(
                    target_model.gpt.step_schedule[index]
                    for index in continuation_loop_indices
                )
                rows_per_request = self.speculative_num_draft_tokens
                for batch_size in range(1, graph_max_batch_size + 1):
                    row_count = batch_size * rows_per_request
                    self.target_continuation_piecewise_graph_runners[row_count] = (
                        target_model.gpt.build_piecewise_graph_runner(
                            row_count=row_count,
                            loop_indices=continuation_loop_indices,
                            step_schedule=continuation_schedule,
                            initial_time=sum(target_model.gpt.step_schedule[:2]),
                        )
                    )
                self.target_continuation_piecewise_graph_runner = (
                    self.target_continuation_piecewise_graph_runners.get(
                        rows_per_request
                    )
                )
                target_model.depthspec_target_continuation_piecewise_runner = (
                    self.target_continuation_piecewise_graph_runner
                )
                target_model.depthspec_target_continuation_piecewise_runners = (
                    self.target_continuation_piecewise_graph_runners
                )
            if self.runtime_config.diagnostics.get(
                "mixed_piecewise_graph", False
            ):
                from main_lib.DepthSpec.sglang_models.loopformer import (
                    LoopFormerBlockPiecewiseGraph,
                )

                configured_row_counts = self.runtime_config.diagnostics.get(
                    "mixed_piecewise_graph_row_counts",
                    [7, 8, 9, 14],
                )
                graph_cache = {}
                hidden_size = int(target_model.config.hidden_size)
                for row_count in configured_row_counts:
                    graph_cache[int(row_count)] = {
                        block_index: LoopFormerBlockPiecewiseGraph(
                            block,
                            int(row_count),
                            hidden_size,
                        )
                        for block_index, block in enumerate(
                            target_model.gpt.transformer.h.blocks
                        )
                    }
                target_model.depthspec_mixed_piecewise_graphs = graph_cache
            if self.runtime_config.canonical_stem_piecewise_graph:
                stem_row_counts = {self.topk + 1, 2}
                enable_multi_request_partial_graph = bool(
                    self.runtime_config.diagnostics.get(
                        "canonical_partial_stem_piecewise_graph", False
                    )
                )
                for batch_size in range(2, graph_max_batch_size + 1):
                    stem_row_counts.update(
                        {
                            batch_size * 2,
                            batch_size * 3,
                        }
                    )
                    if enable_multi_request_partial_graph:
                        stem_row_counts.add(batch_size * (self.topk + 1))
                if self.runtime_config.canonical_full_stem_piecewise_graph:
                    stem_row_counts.update(
                        batch_size * self.speculative_num_draft_tokens
                        for batch_size in range(1, graph_max_batch_size + 1)
                    )
                for row_count in sorted(stem_row_counts):
                    self.canonical_stem_piecewise_graph_runners[row_count] = (
                        target_model.gpt.build_piecewise_graph_runner(
                            row_count=row_count,
                            loop_indices=(0, 1),
                            step_schedule=(0.125, 0.125),
                        )
                    )
            return
        if not self.runtime_config.draft_active_path_graph:
            super()._capture_cuda_graphs()
            return
        if self.server_args.max_running_requests != 1:
            # Capturing the width-3 graph for bs>1 corrupts shared backend
            # state even when replay is later disabled. Only the validated
            # single-request deployment may capture this experimental graph.
            logger.warning(
                "DepthSpec active-path graph is limited to max_running_requests=1; "
                "capturing is skipped for multi-request servers"
            )
            super()._capture_cuda_graphs()
            return
        if self.draft_extend_attn_backend is None:
            return

        from sglang.srt.speculative.draft_utils import DraftBackendFactory
        from sglang.srt.speculative.eagle_draft_extend_cuda_graph_runner import (
            EAGLEDraftExtendCudaGraphRunner,
        )

        # Use a separate backend state so width-3 capture does not overwrite the
        # buffers owned by the default width=num_draft_tokens graph runner.
        backend_factory = DraftBackendFactory(
            self.draft_runner,
            self.topk,
            self.speculative_num_steps,
            seed_dsa_topk_from_draft_extend=self.seed_dsa_topk_from_draft_extend,
        )
        active_backend = backend_factory.create_draft_extend_backend()
        active_backend.depthspec_draft_extend_num_tokens_per_req = (
            self.speculative_num_steps + 1
        )
        target_model = self.target_worker.model_runner.model
        old_forward_kind = getattr(target_model, "depthspec_forward_kind", "target")
        old_draft_loop_indices = getattr(
            target_model,
            "depthspec_draft_loop_indices",
            None,
        )
        target_model.depthspec_forward_kind = "draft"
        target_model.depthspec_draft_loop_indices = (
            target_model.gpt.loop_num - 3,
            target_model.gpt.loop_num - 2,
            target_model.gpt.loop_num - 1,
        )
        try:
            # Keep the useful draft-decode graph but skip the width-6
            # draft-extend graph; this mode installs the width-3 runner below.
            disable_draft_extend = "SGLANG_DISABLE_DRAFT_EXTEND_CUDA_GRAPH"
            had_disable_draft_extend = disable_draft_extend in os.environ
            previous_disable_draft_extend = os.environ.get(disable_draft_extend)
            os.environ[disable_draft_extend] = "1"
            try:
                super()._capture_cuda_graphs()
            finally:
                if had_disable_draft_extend:
                    os.environ[disable_draft_extend] = previous_disable_draft_extend
                else:
                    del os.environ[disable_draft_extend]

            self.active_path_graph_runner = EAGLEDraftExtendCudaGraphRunner(
                self,
                draft_extend_attn_backend=active_backend,
                captured_req_width=self.speculative_num_steps + 1,
                use_dynamic_select_index=True,
            )
        finally:
            target_model.depthspec_forward_kind = old_forward_kind
            if old_draft_loop_indices is None:
                if hasattr(target_model, "depthspec_draft_loop_indices"):
                    del target_model.depthspec_draft_loop_indices
            else:
                target_model.depthspec_draft_loop_indices = old_draft_loop_indices

    def draft(self, batch):
        started = time.perf_counter()
        target_model = self.target_worker.model_runner.model
        draft_input = batch.spec_info
        if self.runtime_config.canonical_proposal_replacement:
            stem_started = time.perf_counter()
            self._prepare_partial_canonical_stem_audit(batch)
            _append_stage_timing(
                "canonical_proposal_stem",
                int(batch.seq_lens.numel()),
                time.perf_counter() - stem_started,
            )
            coarse_started = time.perf_counter()
            result = self._draft_from_canonical_stem(batch, draft_input)
            _append_stage_timing(
                "canonical_proposal_coarse",
                int(batch.seq_lens.numel()),
                time.perf_counter() - coarse_started,
            )
            _append_stage_timing(
                "draft", int(batch.seq_lens.numel()), time.perf_counter() - started
            )
            return result
        if self.runtime_config.canonical_partial_stem and self.runtime_config.canonical_stem:
            partial_started = time.perf_counter()
            self._prepare_partial_canonical_stem_audit(batch)
            _append_stage_timing(
                "partial_canonical_stem_precompute",
                int(batch.seq_lens.numel()),
                time.perf_counter() - partial_started,
            )
        artifact_audit = bool(
            self.runtime_config.diagnostics.get("prefix_artifact_audit", False)
        )
        artifact_relay = bool(
            self.runtime_config.diagnostics.get("draft_artifact_relay", False)
        )
        prefix_model = target_model.gpt
        prefix_model.depthspec_capture_prefix_artifact = artifact_audit
        prefix_model.depthspec_proposal_prefix_artifacts = []
        prefix_model.depthspec_relay_artifacts = [] if artifact_relay else None
        target_model.depthspec_proposal_prefix_artifacts = (
            prefix_model.depthspec_proposal_prefix_artifacts
        )
        context = (
            self._use_unified_draft_forward(forward_purpose="proposal")
            if self.unified_runner
            else empty_context()
        )
        with context:
            result = super().draft(batch)
        audit_path = self.runtime_config.diagnostics.get(
            "standalone_layout_audit_path"
        )
        if audit_path and not batch.forward_mode.is_idle():
            row = {
                "phase": "proposal",
                "seq_lens": batch.seq_lens.detach().cpu().tolist(),
                "root_tokens": draft_input.bonus_tokens.detach().cpu().tolist(),
                "draft_positions": draft_input.positions.detach().cpu().tolist(),
                "draft_initial_positions": getattr(
                    draft_input,
                    "depthspec_initial_positions",
                    draft_input.positions,
                )
                .detach()
                .cpu()
                .tolist(),
                "verify_tokens": result.draft_token.detach().cpu().tolist(),
                "verify_positions": result.positions.detach().cpu().tolist(),
            }
            output = Path(audit_path)
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row) + "\n")
        _append_stage_timing(
            "draft", int(batch.seq_lens.numel()), time.perf_counter() - started
        )
        return result

    @staticmethod
    def _proposal_row_count(batch, forward_batch) -> int:
        if forward_batch.input_ids is not None:
            return int(forward_batch.input_ids.shape[0])
        spec_info = getattr(forward_batch, "spec_info", None)
        width = getattr(spec_info, "num_tokens_per_req", None)
        if width is not None:
            return int(batch.seq_lens.numel()) * int(width)
        return 0

    def _append_partial_stem_phase_timing(
        self,
        *,
        batch_size: int,
        row_count: int,
        input_build_s: float,
        verify_prepare_s: float,
        attention_metadata_s: float,
        forward_execute_s: float,
        total_s: float,
        graph_hit: bool,
    ) -> None:
        timing_path = self.runtime_config.diagnostics.get(
            "partial_stem_phase_timing_path"
        )
        if not timing_path:
            return
        row = {
            "captured_unix_s": time.time(),
            "batch_size": batch_size,
            "row_count": row_count,
            "input_build_s": input_build_s,
            "verify_prepare_s": verify_prepare_s,
            "attention_metadata_s": attention_metadata_s,
            "forward_execute_s": forward_execute_s,
            "total_s": total_s,
            "graph_hit": graph_hit,
        }
        output = Path(timing_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row) + "\n")

    def _draft_from_canonical_stem(self, batch, draft_input):
        """Generate proposal children from canonical loops0/1 plus loop11."""
        from sglang.srt.speculative.eagle_worker_common import (
            build_eagle_verify_input,
            prepare_for_draft,
        )

        target_model = self.target_worker.model_runner.model
        partial = getattr(target_model, "depthspec_partial_stem_state", None)
        if partial is None:
            raise ValueError("Canonical proposal replacement requires partial stem")
        bs = int(batch.seq_lens.numel())
        topk = int(draft_input.topk_index.shape[1])
        expected_rows = bs * (topk + 1)
        partial_hidden = partial["state"].hidden_states
        if partial_hidden.shape[0] != expected_rows:
            raise ValueError(
                "Canonical proposal stem row mismatch: "
                f"{partial_hidden.shape[0]} != {expected_rows}"
            )
        candidate_hidden = partial_hidden.reshape(bs, topk + 1, -1)[
            :, 1:, :
        ].reshape(bs * topk, -1)

        graph_runner = self.canonical_proposal_graph_runner
        forward_batch, can_run_graph = prepare_for_draft(
            draft_input,
            self.req_to_token_pool,
            batch,
            graph_runner,
            self.draft_runner,
            self.topk,
            self.speculative_num_steps,
        )
        if (
            not can_run_graph
            and not forward_batch.forward_mode.is_idle()
            and self.speculative_num_steps > 1
        ):
            self.draft_attn_backend.init_forward_metadata(forward_batch)
            forward_batch.mark_forward_metadata_ready()
        if can_run_graph:
            forward_batch.spec_info.depthspec_prefix_handoff_hidden_states = (
                candidate_hidden
            )
            proposal_started = time.perf_counter()
            parent_list, top_scores_index, draft_tokens, draft_probs = (
                graph_runner.execute(forward_batch)
            )
            _append_segment_trace(
                phase="proposal_coarse",
                request_count=int(batch.seq_lens.numel()),
                row_count=self._proposal_row_count(batch, forward_batch),
                loop_indices=(11,),
                step_schedule=(0.75,),
                attention_contract="draft-chain",
                graph_hit=True,
                elapsed_s=time.perf_counter() - proposal_started,
            )
        else:
            context = self._use_unified_draft_forward(
                candidate_hidden,
                use_prefix_handoff=True,
                forward_purpose="proposal",
            )
            with context:
                proposal_started = time.perf_counter()
                parent_list, top_scores_index, draft_tokens, draft_probs = (
                    self.draft_forward(forward_batch)
                )
                _append_segment_trace(
                    phase="proposal_coarse",
                    request_count=int(batch.seq_lens.numel()),
                    row_count=self._proposal_row_count(batch, forward_batch),
                    loop_indices=(11,),
                    step_schedule=(0.75,),
                    attention_contract="draft-chain",
                    graph_hit=False,
                    elapsed_s=time.perf_counter() - proposal_started,
                )
        return build_eagle_verify_input(
            batch,
            draft_input,
            parent_list,
            top_scores_index,
            draft_tokens,
            draft_probs,
            target_worker=self.target_worker,
            topk=self.topk,
            num_steps=self.speculative_num_steps,
            num_draft_tokens=self.speculative_num_draft_tokens,
            tree_mask_mode=self.tree_mask_mode,
            device=self.device,
        )

    def _prepare_partial_canonical_stem_audit(self, batch):
        """Precompute a root-plus-first-level canonical stem before drafting."""
        from sglang.srt.model_executor.forward_context import (
            ForwardContext,
            forward_context,
        )
        from sglang.srt.speculative.eagle_info import EagleVerifyInput
        from sglang.srt.speculative.eagle_utils import eagle_prepare_for_verify

        draft_input = batch.spec_info
        if draft_input.bonus_tokens is None or draft_input.topk_index is None:
            raise ValueError("Partial stem audit requires draft seed tokens")

        partial_total_started = time.perf_counter()
        target_runner = self.target_worker.model_runner
        target_model = target_runner.model
        device = batch.device
        batch_size = int(batch.seq_lens.numel())
        topk = int(draft_input.topk_index.shape[1])
        width = topk + 1
        root = draft_input.bonus_tokens.reshape(batch_size, 1)
        first_level = draft_input.topk_index.reshape(batch_size, topk)
        partial_tokens = torch.cat((root, first_level), dim=1).reshape(-1)
        cache_templates = bool(
            self.runtime_config.diagnostics.get("cache_partial_stem_templates", False)
        )
        persistent_mask = bool(
            self.runtime_config.diagnostics.get(
                "persistent_partial_stem_mask", False
            )
        )
        if cache_templates:
            position_offsets = self._partial_stem_position_offsets(topk, device)
        else:
            position_offsets = torch.tensor(
                [0] + [1] * topk, device=device, dtype=torch.int64
            )
        partial_positions = (
            batch.seq_lens[:, None] + position_offsets[None, :]
        ).reshape(-1)

        if self.runtime_config.diagnostics.get("vectorized_partial_stem_mask", False):
            sequence_lens = batch.seq_lens.long()
            sequence_prefix = torch.cumsum(sequence_lens, dim=0) - sequence_lens
            sequence_sum = (
                int(batch.seq_lens_sum)
                if batch.seq_lens_sum is not None
                else int(sequence_lens.sum().item())
            )
            custom_mask = torch.ones(
                width * (sequence_sum + batch_size * width),
                dtype=torch.bool,
                device=device,
            )
            tree_starts = (
                sequence_prefix + torch.arange(batch_size, device=device) * width
            ) * width
            tree_rows = torch.arange(width, device=device).repeat_interleave(width)
            tree_columns = torch.arange(width, device=device).repeat(width)
            tree_cells = tree_starts[:, None] + tree_rows[None] * width + (
                tree_columns[None]
            )
            allowed_rows = torch.arange(width, device=device)
            parent_rows = torch.arange(1, width, device=device)
            allowed_tree_cells = torch.cat(
                (
                    tree_starts[:, None] + allowed_rows[None] * width + allowed_rows[None],
                    tree_starts[:, None] + parent_rows[None] * width,
                ),
                dim=1,
            ).reshape(-1)
            custom_mask.index_fill_(0, tree_cells.reshape(-1), False)
            custom_mask.index_fill_(0, allowed_tree_cells, True)
        elif persistent_mask:
            if batch.seq_lens_cpu is not None:
                sequence_lengths = batch.seq_lens_cpu.tolist()
            else:
                sequence_lengths = batch.seq_lens.tolist()
            custom_mask = self._partial_stem_persistent_mask(
                sequence_lengths,
                width,
                device,
            )
        elif cache_templates:
            if batch.seq_lens_cpu is not None:
                sequence_lengths = batch.seq_lens_cpu.tolist()
            else:
                sequence_lengths = batch.seq_lens.tolist()
            mask_segments = [
                self._partial_stem_mask(int(sequence_length), width, device)
                for sequence_length in sequence_lengths
            ]
            custom_mask = (
                mask_segments[0]
                if len(mask_segments) == 1
                else torch.cat(mask_segments, dim=0)
            )
        else:
            mask_segments = []
            tree_mask = torch.eye(width, dtype=torch.bool, device=device)
            tree_mask[1:, 0] = True
            for sequence_length in batch.seq_lens.tolist():
                prefix_mask = torch.ones(
                    (width, sequence_length), dtype=torch.bool, device=device
                )
                mask_segments.append(
                    torch.cat((prefix_mask, tree_mask), dim=1).reshape(-1)
                )
            custom_mask = torch.cat(mask_segments, dim=0)
        if cache_templates:
            (
                retrieve_index,
                retrieve_next_token,
                retrieve_next_sibling,
            ) = self._partial_stem_retrieve_template(batch_size, topk, device)
        else:
            retrieve_index = (
                torch.arange(width, device=device, dtype=torch.int64)
                .unsqueeze(0)
                .expand(batch_size, -1)
            )
            retrieve_next_token = torch.full(
                (batch_size, width), -1, device=device, dtype=torch.int64
            )
            retrieve_next_sibling = torch.full(
                (batch_size, width), -1, device=device, dtype=torch.int64
            )
            if topk > 0:
                retrieve_next_token[:, 0] = 1
                first_level_siblings = torch.full(
                    (topk,), -1, device=device, dtype=torch.int64
                )
                first_level_siblings[:-1] = torch.arange(
                    2, width, device=device, dtype=torch.int64
                )
                retrieve_next_sibling[:, 1:width] = first_level_siblings
        partial_verify_input = EagleVerifyInput(
            draft_token=partial_tokens,
            custom_mask=custom_mask,
            positions=partial_positions,
            retrieve_index=retrieve_index,
            retrieve_next_token=retrieve_next_token,
            retrieve_next_sibling=retrieve_next_sibling,
            retrieve_cum_len=None,
            spec_steps=1,
            topk=topk,
            draft_token_num=width,
            capture_hidden_mode=None,
            seq_lens_sum=(
                int(batch.seq_lens_sum)
                if batch.seq_lens_sum is not None
                else int(batch.seq_lens.sum().item())
            ),
            seq_lens_cpu=(
                batch.seq_lens_cpu
                if batch.seq_lens_cpu is not None
                else batch.seq_lens.cpu()
            ),
        )
        input_build_elapsed = time.perf_counter() - partial_total_started

        saved_state = {
            name: getattr(batch, name, None)
            for name in (
                "input_ids",
                "out_cache_loc",
                "out_cache_loc_dsv4",
                "prefix_lens",
                "extend_lens",
                "extend_num_tokens",
                "seq_lens_sum",
                "forward_mode",
                "spec_info",
            )
        }
        batch.spec_info = partial_verify_input
        try:
            verify_prepare_started = time.perf_counter()
            partial_forward_batch, can_run_graph = eagle_prepare_for_verify(
                partial_verify_input,
                self.req_to_token_pool,
                batch,
                self.target_worker,
            )
            if can_run_graph:
                raise ValueError("Partial stem audit unexpectedly selected a graph")
            verify_prepare_elapsed = time.perf_counter() - verify_prepare_started
            attention_metadata_started = time.perf_counter()
            target_runner.attn_backend.init_forward_metadata(partial_forward_batch)
            partial_forward_batch.mark_forward_metadata_ready()
            attention_metadata_elapsed = (
                time.perf_counter() - attention_metadata_started
            )
            with forward_context(
                ForwardContext(attn_backend=target_runner.attn_backend)
            ):
                partial_started = time.perf_counter()
                piecewise_runner = self.canonical_stem_piecewise_graph_runners.get(
                    int(partial_forward_batch.input_ids.shape[0])
                )
                if not self.runtime_config.diagnostics.get(
                    "canonical_partial_stem_piecewise_graph",
                    self.runtime_config.canonical_stem_piecewise_graph
                    and self.server_args.max_running_requests == 1,
                ):
                    piecewise_runner = None
                forward_execute_started = time.perf_counter()
                if piecewise_runner is None:
                    _, partial_state = target_model.gpt.forward_branch(
                        partial_forward_batch.input_ids,
                        partial_forward_batch.positions,
                        partial_forward_batch,
                        loop_indices=(0, 1),
                        step_schedule=(0.125, 0.125),
                        return_state=True,
                    )
                else:
                    _, partial_state = piecewise_runner.execute(
                        partial_forward_batch.input_ids,
                        partial_forward_batch.positions,
                        partial_forward_batch,
                        return_state=True,
                    )
                _append_segment_trace(
                    phase="proposal_partial_stem",
                    request_count=int(batch.seq_lens.numel()),
                    row_count=int(partial_forward_batch.input_ids.shape[0]),
                    loop_indices=(0, 1),
                    step_schedule=(0.125, 0.125),
                    attention_contract="tree-custom",
                    graph_hit=piecewise_runner is not None,
                    elapsed_s=time.perf_counter() - partial_started,
                )
                forward_execute_elapsed = (
                    time.perf_counter() - forward_execute_started
                )
        finally:
            for name, value in saved_state.items():
                setattr(batch, name, value)

        self._append_partial_stem_phase_timing(
            batch_size=batch_size,
            row_count=int(partial_forward_batch.input_ids.shape[0]),
            input_build_s=input_build_elapsed,
            verify_prepare_s=verify_prepare_elapsed,
            attention_metadata_s=attention_metadata_elapsed,
            forward_execute_s=forward_execute_elapsed,
            total_s=time.perf_counter() - partial_total_started,
            graph_hit=piecewise_runner is not None,
        )

        capture_audit_snapshot = bool(
            self.runtime_config.diagnostics.get("partial_stem_audit_path")
        )
        slots = None
        saved_kvs = None
        if capture_audit_snapshot:
            pool = target_runner.token_to_kv_pool
            slots = partial_forward_batch.out_cache_loc.detach().cpu().tolist()
            blocks_per_loop = int(target_model.config.num_hidden_layers)
            saved_kvs = {}
            for loop_offset in range(2):
                for block_index in range(blocks_per_loop):
                    layer_id = loop_offset * blocks_per_loop + block_index
                    saved_kvs[(loop_offset, block_index)] = (
                        pool.get_key_buffer(layer_id)[slots].clone(),
                        pool.get_value_buffer(layer_id)[slots].clone(),
                    )
        target_model.depthspec_partial_stem_state = {
            "state": partial_state,
            "slots": slots,
            "kvs": saved_kvs,
            "tokens": partial_tokens.detach().cpu().tolist(),
            "positions": partial_positions.detach().cpu().tolist(),
            "forward_batch": partial_forward_batch,
            "segment": LoopSegment(
                input_ids=partial_forward_batch.input_ids,
                positions=partial_forward_batch.positions,
                out_cache_loc=partial_forward_batch.out_cache_loc,
                seq_lens=batch.seq_lens,
                seq_lens_cpu=(
                    batch.seq_lens_cpu
                    if batch.seq_lens_cpu is not None
                    else batch.seq_lens.cpu()
                ),
                seq_lens_sum=int(batch.seq_lens_sum)
                if batch.seq_lens_sum is not None
                else int(batch.seq_lens.sum().item()),
                spec_info=partial_verify_input,
                req_pool_indices=batch.req_pool_indices,
                phase="proposal_partial_stem",
                loop_indices=(0, 1),
                step_schedule=(0.125, 0.125),
                attention_contract="tree-custom",
            ),
        }

    def _draft_extend_for_decode(self, batch, batch_result):
        started = time.perf_counter()
        if self.unified_runner and self.runtime_config.draft_active_path_extend:
            with self._use_unified_draft_forward(forward_purpose="draft_extend_decode"):
                self._run_active_path_draft_extend(batch, batch_result)
            _append_stage_timing(
                "draft_extend_decode",
                int(batch.seq_lens.numel()),
                time.perf_counter() - started,
            )
            return None

        prefix_hidden_states = None
        artifact_relay = bool(
            self.runtime_config.diagnostics.get("draft_artifact_relay", False)
        )
        if artifact_relay and self.unified_runner:
            target_model = self.target_worker.model_runner.model
            candidate = getattr(target_model, "depthspec_relay_hidden_states", None)
            expected_rows = len(batch.seq_lens) * self.speculative_num_draft_tokens
            if candidate is not None and candidate.shape[0] == expected_rows:
                prefix_hidden_states = candidate
                target_model.depthspec_prefix_accept_lens = batch_result.accept_lens
                target_model.depthspec_prefix_accept_index = getattr(
                    batch_result.next_draft_input,
                    "accept_index",
                    None,
                )
                target_model.depthspec_prefix_candidate_tokens = getattr(
                    batch_result.next_draft_input,
                    "candidate_tokens",
                    None,
                )
                target_model.depthspec_prefix_candidate_positions = getattr(
                    batch_result.next_draft_input,
                    "candidate_positions",
                    None,
                )
            else:
                logger.warning(
                    "DepthSpec draft artifact relay fell back to normal draft extend"
                )
            if (
                artifact_relay
                and self.runtime_config.diagnostics.get(
                    "draft_relay_mini_bonus", False
                )
                and prefix_hidden_states is not None
            ):
                self._copy_accepted_transition_prefix_kv(batch, batch_result)
                self._compute_mini_bonus_transition(batch, batch_result)
            if (
                artifact_relay
                and self.runtime_config.diagnostics.get(
                    "draft_relay_mini_path", False
                )
                and prefix_hidden_states is not None
            ):
                self._compute_mini_transition_path(batch, batch_result)
        elif self.unified_runner and (
            self.runtime_config.diagnostics.get("prefix_state_handoff", False)
            or self.runtime_config.prefix_shift_audit
        ):
            candidate = getattr(batch_result.logits_output, "hidden_states", None)
            expected_rows = len(batch.seq_lens) * self.speculative_num_draft_tokens
            if candidate is not None and candidate.shape[0] == expected_rows:
                prefix_hidden_states = candidate
                target_model = self.target_worker.model_runner.model
                target_model.depthspec_prefix_handoff_hidden_states = candidate
                target_model.depthspec_prefix_accept_lens = batch_result.accept_lens
                target_model.depthspec_prefix_accept_index = getattr(
                    batch_result.next_draft_input,
                    "accept_index",
                    None,
                )
                target_model.depthspec_prefix_candidate_tokens = getattr(
                    batch_result.next_draft_input,
                    "candidate_tokens",
                    None,
                )
                target_model.depthspec_prefix_candidate_positions = getattr(
                    batch_result.next_draft_input,
                    "candidate_positions",
                    None,
                )
        context = (
            self._use_unified_draft_forward(
                prefix_hidden_states,
                use_prefix_handoff=(
                    prefix_hidden_states is not None
                    and (
                        artifact_relay
                        or self.runtime_config.diagnostics.get(
                            "prefix_state_handoff", False
                        )
                    )
                ),
                forward_purpose="draft_extend_decode",
            )
            if self.unified_runner
            else empty_context()
        )
        with context:
            result = super()._draft_extend_for_decode(batch, batch_result)
        if (
            artifact_relay
            and self.runtime_config.diagnostics.get(
                "target_first_level_relay", False
            )
        ):
            # Diagnostic compensation: target verify already knows the next
            # distribution at the last accepted node. Supply its first-level
            # top-k while retaining the relayed hidden state for deeper steps.
            logits = batch_result.logits_output.next_token_logits
            bs = len(batch.seq_lens)
            request_indices = torch.arange(bs, device=self.device)
            local_accept_index = _localize_tree_accept_index(
                batch_result.next_draft_input.accept_index,
                self.speculative_num_draft_tokens,
            )
            last_indices = local_accept_index[
                request_indices,
                (batch_result.accept_lens - 1).long(),
            ]
            selected_logits = logits.view(bs, self.speculative_num_draft_tokens, -1)[
                request_indices, last_indices
            ]
            target_probs = torch.softmax(selected_logits, dim=-1)
            target_topk_p, target_topk_index = torch.topk(
                target_probs,
                self.topk,
                dim=-1,
            )
            batch_result.next_draft_input.topk_p = target_topk_p
            batch_result.next_draft_input.topk_index = target_topk_index
        _append_stage_timing(
            "draft_extend_decode",
            int(batch.seq_lens.numel()),
            time.perf_counter() - started,
        )
        return result

    def _copy_accepted_transition_prefix_kv(self, batch, batch_result) -> None:
        """Shift accepted node loop9/10 KV onto transition-row slots."""
        target_runner = self.target_worker.model_runner
        target_model = target_runner.model
        pool = target_runner.token_to_kv_pool
        rows = self.req_to_token_pool.req_to_token
        blocks_per_loop = int(target_model.config.num_hidden_layers)
        draft_first_loop = target_model.gpt.loop_num - 3

        source_locs = []
        target_locs = []
        for request_index in range(len(batch.seq_lens)):
            accept_length = min(
                int(batch_result.accept_lens[request_index].item()),
                self.speculative_num_draft_tokens,
            )
            seq_len = int(batch.seq_lens[request_index].item())
            req_pool_index = int(batch.req_pool_indices[request_index].item())
            # Draft-extend transition j consumes accepted node j+1. Verify's
            # accepted-node compaction places node j+1 at seq_len+j+1, while
            # draft extend writes transition j at seq_len+j.
            for transition_index in range(max(accept_length - 1, 0)):
                target_locs.append(
                    int(rows[req_pool_index, seq_len + transition_index].item())
                )
                source_locs.append(
                    int(rows[req_pool_index, seq_len + transition_index + 1].item())
                )

        if not source_locs:
            return
        source_locs = torch.tensor(source_locs, dtype=torch.int64, device=self.device)
        target_locs = torch.tensor(target_locs, dtype=torch.int64, device=self.device)
        for loop_offset in range(2):
            loop_index = draft_first_loop + loop_offset
            for block_index in range(blocks_per_loop):
                layer_id = loop_index * blocks_per_loop + block_index
                key_buffer = pool.get_key_buffer(layer_id)
                value_buffer = pool.get_value_buffer(layer_id)
                # Clone sources because source and destination ranges overlap.
                key_buffer[target_locs] = key_buffer[source_locs].clone()
                value_buffer[target_locs] = value_buffer[source_locs].clone()

    def _compute_mini_bonus_transition(self, batch, batch_result) -> None:
        """Compute only each request's selected bonus transition prefix state.

        The accepted proposal tokens already carry draft-loop KV through verify's
        accepted-slot move. The selected target-predicted bonus token is new, so
        this one-row-per-request mini extend materializes its loops9/10 state.
        """
        from sglang.srt.model_executor.forward_batch_info import (
            ForwardBatch,
            ForwardMode,
        )
        from sglang.srt.model_executor.forward_context import (
            ForwardContext,
            forward_context,
        )
        from sglang.srt.speculative.eagle_info import EagleDraftExtendInput

        target_runner = self.target_worker.model_runner
        target_model = target_runner.model
        relay_hidden = getattr(target_model, "depthspec_relay_hidden_states", None)
        if relay_hidden is None:
            raise ValueError("Mini bonus relay requires relay hidden states")

        bs = len(batch.seq_lens)
        num_draft_tokens = self.speculative_num_draft_tokens
        predicted_tokens = batch_result.next_token_ids.reshape(
            bs,
            num_draft_tokens,
        )
        input_ids = []
        positions = []
        out_cache_locs = []
        prefix_lens = []
        selected_rows = []
        rows = self.req_to_token_pool.req_to_token

        for request_index in range(bs):
            selected_row = (
                min(
                    int(batch_result.accept_lens[request_index].item()),
                    num_draft_tokens,
                )
                - 1
            )
            flat_row = request_index * num_draft_tokens + selected_row
            selected_rows.append(selected_row)
            input_ids.append(int(predicted_tokens[request_index, selected_row].item()))
            position = int(batch.seq_lens[request_index].item()) + selected_row
            positions.append(position)
            prefix_lens.append(position)
            req_pool_index = int(batch.req_pool_indices[request_index].item())
            out_cache_locs.append(int(rows[req_pool_index, position].item()))

        device = self.device
        saved_state = {
            "input_ids": batch.input_ids,
            "out_cache_loc": batch.out_cache_loc,
            "prefix_lens": getattr(batch, "prefix_lens", None),
            "extend_lens": getattr(batch, "extend_lens", None),
            "extend_num_tokens": getattr(batch, "extend_num_tokens", None),
            "seq_lens_sum": getattr(batch, "seq_lens_sum", None),
            "forward_mode": batch.forward_mode,
            "spec_info": batch.spec_info,
        }
        old_attention_backend = target_runner.attn_backend
        target_runner.attn_backend = self.draft_extend_attn_backend
        batch.input_ids = torch.tensor(input_ids, dtype=torch.int64, device=device)
        batch.out_cache_loc = torch.tensor(
            out_cache_locs, dtype=torch.int64, device=device
        )
        batch.prefix_lens = prefix_lens
        batch.extend_lens = [1] * bs
        batch.extend_num_tokens = bs
        batch.forward_mode = ForwardMode.DRAFT_EXTEND_V2
        batch.spec_info = EagleDraftExtendInput(
            positions=torch.tensor(positions, dtype=torch.int64, device=device),
            num_tokens_per_req=1,
            num_tokens_for_logprob_per_req=1,
        )

        try:
            forward_batch = ForwardBatch.init_new(
                batch,
                self.draft_runner,
                capture_hidden_mode=None,
                return_hidden_states_before_norm=False,
            )
            self.draft_extend_attn_backend.init_forward_metadata(forward_batch)
            forward_batch.mark_forward_metadata_ready()
            gpt = target_model.gpt
            with forward_context(
                ForwardContext(attn_backend=self.draft_extend_attn_backend)
            ):
                _, bonus_prefix_state = gpt.forward_branch(
                    forward_batch.input_ids,
                    forward_batch.positions,
                    forward_batch,
                    loop_indices=(gpt.loop_num - 3, gpt.loop_num - 2),
                    step_schedule=(0.125, 0.125),
                    return_state=True,
                )
            for request_index, selected_row in enumerate(selected_rows):
                flat_row = request_index * num_draft_tokens + selected_row
                relay_hidden[flat_row].copy_(
                    bonus_prefix_state.hidden_states[request_index]
                )
        finally:
            target_runner.attn_backend = old_attention_backend
            batch.input_ids = saved_state["input_ids"]
            batch.out_cache_loc = saved_state["out_cache_loc"]
            batch.prefix_lens = saved_state["prefix_lens"]
            batch.extend_lens = saved_state["extend_lens"]
            batch.extend_num_tokens = saved_state["extend_num_tokens"]
            batch.seq_lens_sum = saved_state["seq_lens_sum"]
            batch.forward_mode = saved_state["forward_mode"]
            batch.spec_info = saved_state["spec_info"]

    def _run_active_path_draft_extend(self, batch, batch_result) -> None:
        """Replace tree-width draft extend with an accepted-path-only forward."""
        from sglang.srt.model_executor.forward_batch_info import (
            CaptureHiddenMode,
            ForwardBatch,
            ForwardMode,
        )
        from sglang.srt.runtime_context import get_spec
        from sglang.srt.speculative.eagle_info import EagleDraftExtendInput
        from sglang.srt.speculative.spec_utils import renorm_draft_probs
        from sglang.srt.utils import fast_topk

        bs = len(batch.seq_lens)
        num_draft_tokens = self.speculative_num_draft_tokens
        predicted_tokens = batch_result.next_token_ids.reshape(
            bs,
            num_draft_tokens,
        )
        fixed_active_path = self.runtime_config.draft_active_path_fixed
        position_offset = 1 if self.runtime_config.target_aligned_layout else 0
        tokens_per_req = (
            self.speculative_num_steps + 1 if fixed_active_path else num_draft_tokens
        )
        if fixed_active_path:
            active_width = self.speculative_num_steps + 1
            transition_offsets = torch.arange(
                active_width, device=self.device, dtype=torch.int64
            )
            transition_positions = (
                batch.seq_lens[:, None]
                + position_offset
                + transition_offsets[None, :]
            )
            request_rows = self.req_to_token_pool.req_to_token[
                batch.req_pool_indices.long()
            ]
            transition_locs = torch.gather(
                request_rows,
                1,
                transition_positions.long(),
            )
            input_ids = predicted_tokens[:, :active_width].reshape(-1)
            positions = transition_positions.reshape(-1)
            out_cache_locs = transition_locs.reshape(-1)
            request_indices = torch.arange(bs, device=self.device, dtype=torch.int64)
            select_indices = (
                request_indices * active_width + batch_result.accept_lens - 1
            )
            prefix_lens = batch.seq_lens + position_offset
            extend_lens = torch.full(
                (bs,), active_width, dtype=torch.int64, device=self.device
            )
        else:
            transition_offsets = torch.arange(
                num_draft_tokens, device=self.device, dtype=torch.int64
            )
            active_mask = (
                transition_offsets[None, :] < batch_result.accept_lens[:, None]
            )
            transition_positions = (
                batch.seq_lens[:, None]
                + position_offset
                + transition_offsets[None, :]
            )
            request_rows = self.req_to_token_pool.req_to_token[
                batch.req_pool_indices.long()
            ]
            transition_locs = torch.gather(
                request_rows,
                1,
                transition_positions.long(),
            )
            input_ids = predicted_tokens[active_mask]
            positions = transition_positions[active_mask]
            out_cache_locs = transition_locs[active_mask]
            select_indices = torch.cumsum(batch_result.accept_lens, dim=0) - 1
            prefix_lens = batch.seq_lens + position_offset
            extend_lens = batch_result.accept_lens

        saved_state = {
            "input_ids": batch.input_ids,
            "out_cache_loc": batch.out_cache_loc,
            "prefix_lens": getattr(batch, "prefix_lens", None),
            "extend_lens": getattr(batch, "extend_lens", None),
            "extend_num_tokens": getattr(batch, "extend_num_tokens", None),
            "seq_lens_sum": getattr(batch, "seq_lens_sum", None),
            "forward_mode": batch.forward_mode,
            "spec_info": batch.spec_info,
        }
        batch.input_ids = input_ids
        batch.out_cache_loc = out_cache_locs
        batch.prefix_lens = prefix_lens
        batch.extend_lens = extend_lens
        batch.extend_num_tokens = int(input_ids.shape[0])
        batch.forward_mode = ForwardMode.DRAFT_EXTEND_V2
        batch.spec_info = EagleDraftExtendInput(
            positions=positions,
            num_tokens_per_req=tokens_per_req,
            num_tokens_for_logprob_per_req=1,
            select_index=select_indices,
            num_correct_drafts=batch_result.accept_lens - 1,
            num_accept_tokens=batch_result.accept_lens,
        )

        try:
            forward_batch = ForwardBatch.init_new(
                batch,
                self.draft_runner,
                capture_hidden_mode=CaptureHiddenMode.NULL,
                return_hidden_states_before_norm=False,
            )
            # Match prepare_for_draft_extend(): graph metadata sees post-write
            # sequence lengths and derives prefix length as seq_len - extend_len.
            forward_batch.seq_lens = (
                forward_batch.seq_lens
                + position_offset
                + forward_batch.extend_seq_lens
            )
            if forward_batch.seq_lens_cpu is not None:
                if forward_batch.extend_seq_lens_cpu is not None:
                    cpu_extend_lens = torch.tensor(
                        forward_batch.extend_seq_lens_cpu,
                        dtype=forward_batch.seq_lens_cpu.dtype,
                        device=forward_batch.seq_lens_cpu.device,
                    )
                elif fixed_active_path:
                    cpu_extend_lens = torch.full(
                        (bs,),
                        tokens_per_req,
                        dtype=forward_batch.seq_lens_cpu.dtype,
                        device=forward_batch.seq_lens_cpu.device,
                    )
                    forward_batch.spec_info.extend_seq_lens_cpu = [tokens_per_req] * bs
                else:
                    cpu_extend_lens = forward_batch.extend_seq_lens.to(
                        forward_batch.seq_lens_cpu.device,
                        non_blocking=False,
                    )
                    forward_batch.spec_info.extend_seq_lens_cpu = (
                        forward_batch.extend_seq_lens.cpu().tolist()
                    )
                forward_batch.seq_lens_cpu = (
                    forward_batch.seq_lens_cpu
                    + position_offset
                    + cpu_extend_lens
                )
            forward_batch.seq_lens_sum += int(input_ids.shape[0]) + (
                bs * position_offset
            )
            forward_batch.spec_info.extend_seq_lens_tensor = (
                forward_batch.extend_seq_lens
            )
            active_graph_runner = getattr(self, "active_path_graph_runner", None)
            used_active_graph = False
            # Multi-request replay currently corrupts the target KV lifecycle.
            # Keep the fast graph for the validated single-request mode and fall
            # back to eager execution until the shared-cache invariant is fixed.
            if self.runtime_config.target_verify_draft_extend:
                logits_output = self._run_target_compatible_draft_extend(
                    batch,
                    batch_result,
                    input_ids,
                    positions,
                    out_cache_locs,
                    prefix_lens,
                    tokens_per_req,
                    select_indices,
                )
            elif (
                active_graph_runner is not None
                and bs == 1
                and active_graph_runner.can_run_graph(forward_batch)
            ):
                used_active_graph = True
                logits_output = active_graph_runner.execute(forward_batch)
            else:
                logits_output = self.draft_runner.forward(forward_batch).logits_output
            if (
                used_active_graph
                and self.runtime_config.diagnostics.get("active_graph_audit", False)
            ):
                graph_logits = logits_output.next_token_logits.clone()
                graph_metadata = (
                    active_graph_runner.draft_extend_attn_backend.forward_metadata
                )
                logger.info(
                    "DepthSpec active graph metadata before eager: "
                    "qo=%s kv_indptr=%s max_extend=%s kv_indices=%s",
                    graph_metadata.qo_indptr.tolist(),
                    graph_metadata.kv_indptr.tolist(),
                    graph_metadata.max_extend_len,
                    graph_metadata.kv_indices[
                        : min(16, graph_metadata.kv_indices.shape[0])
                    ].tolist(),
                )
                old_backend = self.draft_runner.attn_backend
                self.draft_runner.attn_backend = (
                    active_graph_runner.draft_extend_attn_backend
                )
                try:
                    active_graph_runner.draft_extend_attn_backend.init_forward_metadata(
                        forward_batch
                    )
                    forward_batch.mark_forward_metadata_ready()
                    reference_output = self.draft_runner.forward(
                        forward_batch
                    ).logits_output
                finally:
                    self.draft_runner.attn_backend = old_backend
                reference_logits = reference_output.next_token_logits
                logger.info(
                    "DepthSpec active graph metadata after eager: "
                    "qo=%s kv_indptr=%s max_extend=%s kv_indices=%s",
                    graph_metadata.qo_indptr.tolist(),
                    graph_metadata.kv_indptr.tolist(),
                    graph_metadata.max_extend_len,
                    graph_metadata.kv_indices[
                        : min(16, graph_metadata.kv_indices.shape[0])
                    ].tolist(),
                )
                logit_difference = (graph_logits - reference_logits).abs()
                graph_topk = torch.topk(graph_logits, self.topk, dim=-1).indices
                reference_topk = torch.topk(reference_logits, self.topk, dim=-1).indices
                logger.info(
                    "DepthSpec active graph audit: max_logit_error=%.6f "
                    "mean_logit_error=%.6f topk_match=%s accept_lens=%s "
                    "select_index=%s graph_topk=%s reference_topk=%s",
                    float(logit_difference.max()),
                    float(logit_difference.mean()),
                    torch.equal(graph_topk, reference_topk),
                    batch_result.accept_lens.tolist(),
                    select_indices.tolist(),
                    graph_topk.tolist(),
                    reference_topk.tolist(),
                )
                static_buffers = active_graph_runner.buffers
                logger.info(
                    "DepthSpec active graph inputs: input_ids=%s/%s "
                    "positions=%s/%s seq_lens=%s/%s out_cache=%s/%s",
                    forward_batch.input_ids.tolist(),
                    static_buffers.input_ids[
                        : forward_batch.input_ids.shape[0]
                    ].tolist(),
                    forward_batch.positions.tolist(),
                    static_buffers.positions[
                        : forward_batch.positions.shape[0]
                    ].tolist(),
                    forward_batch.seq_lens.tolist(),
                    static_buffers.seq_lens[: forward_batch.seq_lens.shape[0]].tolist(),
                    forward_batch.out_cache_loc.tolist(),
                    static_buffers.out_cache_loc[
                        : forward_batch.out_cache_loc.shape[0]
                    ].tolist(),
                )
                forward_batch.spec_info.select_index = None
                try:
                    active_graph_runner.draft_extend_attn_backend.init_forward_metadata(
                        forward_batch
                    )
                    forward_batch.mark_forward_metadata_ready()
                    full_reference_output = self.draft_runner.forward(
                        forward_batch
                    ).logits_output
                finally:
                    forward_batch.spec_info.select_index = select_indices
                row_errors = [
                    float(
                        (
                            graph_logits[request_index]
                            - full_reference_output.next_token_logits[row_index]
                        )
                        .abs()
                        .max()
                    )
                    for request_index in range(bs)
                    for row_index in range(tokens_per_req)
                ]
                logger.info(
                    "DepthSpec active graph row errors: %s",
                    row_errors,
                )
            next_logits = logits_output.next_token_logits
            if next_logits is None or next_logits.shape[0] != bs:
                raise RuntimeError(
                    "Active-path draft extend failed to select one row per request: "
                    f"{None if next_logits is None else tuple(next_logits.shape)}"
                )
            if get_spec().speculative_use_rejection_sampling:
                from sglang.srt.speculative.spec_utils import (
                    sample_draft_proposal,
                )

                draft_probs, topk_p, topk_index = sample_draft_proposal(
                    next_logits,
                    batch.sampling_info.temperatures,
                )
                batch_result.next_draft_input.draft_probs = draft_probs
            else:
                probs = renorm_draft_probs(
                    next_logits,
                    batch.sampling_info,
                    get_spec().speculative_use_rejection_sampling,
                )
                topk_p, topk_index = fast_topk(probs, self.topk, dim=-1)
                batch_result.next_draft_input.draft_probs = None
            batch_result.next_draft_input.topk_p = topk_p
            batch_result.next_draft_input.topk_index = topk_index
            batch_result.next_draft_input.hidden_states = logits_output.hidden_states
            if self.runtime_config.diagnostics.get("target_draft_kv_audit_path"):
                self._audit_target_draft_shared_prefix_kv(
                    batch=batch_result,
                    schedule_batch=batch,
                    input_ids=input_ids,
                    positions=positions,
                    out_cache_locs=out_cache_locs,
                    active_width=tokens_per_req,
                )
            if self.runtime_config.diagnostics.get(
                "kv_source_factorial_audit_path"
            ):
                self._audit_kv_source_factorial(
                    batch_result,
                    batch,
                    input_ids,
                    positions,
                    out_cache_locs,
                    tokens_per_req,
                )
        finally:
            batch.input_ids = saved_state["input_ids"]
            batch.out_cache_loc = saved_state["out_cache_loc"]
            batch.prefix_lens = saved_state["prefix_lens"]
            batch.extend_lens = saved_state["extend_lens"]
            batch.extend_num_tokens = saved_state["extend_num_tokens"]
            batch.seq_lens_sum = saved_state["seq_lens_sum"]
            batch.forward_mode = saved_state["forward_mode"]
            batch.spec_info = saved_state["spec_info"]

    def _run_target_compatible_draft_extend(
        self,
        batch,
        batch_result,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        out_cache_locs: torch.Tensor,
        prefix_lens: torch.Tensor,
        tokens_per_req: int,
        select_indices: torch.Tensor,
    ):
        """Run draft rows through the target-verify custom-mask attention path."""
        from sglang.srt.model_executor.forward_batch_info import (
            CaptureHiddenMode,
            ForwardBatch,
            ForwardMode,
        )
        from sglang.srt.model_executor.forward_context import (
            ForwardContext,
            forward_context,
        )
        from sglang.srt.speculative.eagle_info import EagleVerifyInput

        profile_path = self.runtime_config.diagnostics.get(
            "draft_extend_profile_path"
        )
        profile_skip = int(
            self.runtime_config.diagnostics.get("draft_extend_profile_skip", 20)
        )
        draft_extend_profiler = None
        if profile_path:
            profile_index = getattr(self, "_depthspec_draft_extend_profile_count", 0)
            self._depthspec_draft_extend_profile_count = profile_index + 1
            if profile_index == profile_skip:
                draft_extend_profiler = torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ],
                    with_stack=False,
                    record_shapes=False,
                    profile_memory=False,
                )
                draft_extend_profiler.start()

        bs = int(prefix_lens.numel())
        base_prefix_lens = prefix_lens - 1
        base_prefix_lens_cpu = base_prefix_lens.detach().cpu()
        active_input_ids = input_ids.reshape(bs, tokens_per_req)
        active_positions = positions.reshape(bs, tokens_per_req)
        active_cache_locs = out_cache_locs.reshape(bs, tokens_per_req)
        candidate_tokens = batch_result.next_draft_input.candidate_tokens
        accept_index = batch_result.next_draft_input.accept_index
        local_accept_index = _localize_tree_accept_index(
            accept_index,
            candidate_tokens.shape[1],
        )
        request_indices = torch.arange(bs, device=self.device, dtype=torch.long)
        root_rows = local_accept_index[:, 0].long()
        if self.runtime_config.diagnostics.get("cross_request_shape_audit", False):
            logger.info(
                "DepthSpec cross-request shape audit: bs=%d prefix_lens=%s "
                "batch_seq_lens_shape=%s batch_req_pool_shape=%s "
                "accept_lens_shape=%s accept_lens=%s candidate_tokens_shape=%s "
                "accept_index_shape=%s root_rows=%s",
                bs,
                prefix_lens.tolist(),
                tuple(batch.seq_lens.shape),
                tuple(batch.req_pool_indices.shape),
                tuple(batch_result.accept_lens.shape),
                batch_result.accept_lens.tolist(),
                tuple(candidate_tokens.shape),
                tuple(accept_index.shape),
                root_rows.tolist(),
            )
        root_tokens = candidate_tokens[request_indices, root_rows]
        request_rows = self.req_to_token_pool.req_to_token[
            batch.req_pool_indices.long()
        ]
        root_positions = base_prefix_lens.long()
        root_cache_locs = torch.gather(
            request_rows, 1, root_positions[:, None]
        ).squeeze(1)
        query_width = tokens_per_req + 1
        query_input_ids = torch.cat(
            (root_tokens[:, None], active_input_ids), dim=1
        ).reshape(-1)
        query_positions = torch.cat(
            (root_positions[:, None], active_positions), dim=1
        ).reshape(-1)
        query_cache_locs = torch.cat(
            (root_cache_locs[:, None], active_cache_locs), dim=1
        ).reshape(-1)
        tree_mask = torch.ones(
            (query_width, query_width),
            dtype=torch.bool,
            device=self.device,
        ).tril()
        mask_segments = []
        for prefix_len in base_prefix_lens_cpu.tolist():
            prefix_mask = torch.ones(
                (query_width, int(prefix_len)),
                dtype=torch.bool,
                device=self.device,
            )
            mask_segments.append(
                torch.cat((prefix_mask, tree_mask), dim=1).reshape(-1)
            )
        custom_mask = torch.cat(mask_segments)
        retrieve_shape = (bs, query_width)
        verify_input = EagleVerifyInput(
            draft_token=query_input_ids,
            custom_mask=custom_mask,
            positions=query_positions,
            retrieve_index=torch.full(
                retrieve_shape, -1, dtype=torch.long, device=self.device
            ),
            retrieve_next_token=torch.full(
                retrieve_shape, -1, dtype=torch.long, device=self.device
            ),
            retrieve_next_sibling=torch.full(
                retrieve_shape, -1, dtype=torch.long, device=self.device
            ),
            retrieve_cum_len=None,
            spec_steps=query_width - 1,
            topk=1,
            draft_token_num=query_width,
            capture_hidden_mode=CaptureHiddenMode.NULL,
            seq_lens_sum=int(base_prefix_lens_cpu.sum().item()),
            seq_lens_cpu=base_prefix_lens_cpu,
        )

        batch.input_ids = query_input_ids
        batch.out_cache_loc = query_cache_locs
        batch.prefix_lens = base_prefix_lens
        batch.seq_lens = base_prefix_lens
        batch.seq_lens_cpu = verify_input.seq_lens_cpu
        batch.seq_lens_sum = verify_input.seq_lens_sum
        batch.forward_mode = ForwardMode.TARGET_VERIFY
        batch.spec_info = verify_input
        forward_batch = ForwardBatch.init_new(
            batch,
            self.draft_runner,
            capture_hidden_mode=CaptureHiddenMode.NULL,
            return_hidden_states_before_norm=False,
        )

        runner = self.draft_runner
        target_model = runner.model
        old_backend = runner.attn_backend
        runner.attn_backend = self.target_attn_backend
        draft_extend_started = time.perf_counter()
        try:
            self.target_attn_backend.init_forward_metadata(forward_batch)
            forward_batch.mark_forward_metadata_ready()
            with forward_context(
                ForwardContext(attn_backend=self.target_attn_backend)
            ):
                piecewise_runner = self.target_compatible_piecewise_graph_runners.get(
                    int(forward_batch.input_ids.shape[0])
                )
                if piecewise_runner is None:
                    piecewise_runner = self.target_compatible_piecewise_graph_runner
                if (
                    piecewise_runner is None
                    or piecewise_runner.row_count
                    != int(forward_batch.input_ids.shape[0])
                ):
                    logits_output = runner.forward(forward_batch).logits_output
                else:
                    hidden_states = (
                        piecewise_runner.execute(
                            forward_batch.input_ids,
                            forward_batch.positions,
                            forward_batch,
                        )
                    )
                    logits_output = target_model.logits_processor(
                        forward_batch.input_ids,
                        hidden_states,
                        target_model.lm_head,
                        forward_batch,
                    )
        finally:
            runner.attn_backend = old_backend
        _append_segment_trace(
            phase="target_compatible_draft_extend",
            request_count=bs,
            row_count=int(forward_batch.input_ids.shape[0]),
            loop_indices=tuple(
                range(
                    target_model.gpt.loop_num - 3,
                    target_model.gpt.loop_num,
                )
            ),
            step_schedule=(0.125, 0.125, 0.75),
            attention_contract="tree-custom",
            graph_hit=(
                piecewise_runner is not None
                and piecewise_runner.row_count
                == int(forward_batch.input_ids.shape[0])
            ),
            elapsed_s=time.perf_counter() - draft_extend_started,
            seq_lens_sum=verify_input.seq_lens_sum,
        )

        reference_draft_hidden = None
        reference_draft_logits = None
        if self.runtime_config.diagnostics.get("ragged_draft_extend_audit", False):
            reference_draft_hidden = (
                logits_output.hidden_states.clone()
                if logits_output.hidden_states is not None
                else None
            )
            reference_draft_logits = logits_output.next_token_logits.clone()
            draft_loops = tuple(
                range(
                    target_model.gpt.loop_num - 3,
                    target_model.gpt.loop_num,
                )
            )
            draft_segment = LoopSegment(
                input_ids=query_input_ids,
                positions=query_positions,
                out_cache_loc=query_cache_locs,
                seq_lens=base_prefix_lens,
                seq_lens_cpu=base_prefix_lens_cpu,
                seq_lens_sum=verify_input.seq_lens_sum,
                spec_info=verify_input,
                req_pool_indices=batch.req_pool_indices,
                phase="target_compatible_draft_extend",
                loop_indices=draft_loops,
                step_schedule=(0.125, 0.125, 0.75),
                attention_contract="tree-custom-draft",
            )
            self._audit_mixed_draft_extend_schedule(
                batch=batch,
                batch_result=batch_result,
                forward_batch=forward_batch,
                draft_segment=draft_segment,
                normal_input_ids=root_tokens,
                normal_positions=root_positions,
                normal_out_cache_locs=root_cache_locs,
                reference_hidden=reference_draft_hidden,
                reference_logits=reference_draft_logits,
            )

        query_select_indices = (
            torch.arange(bs, device=self.device, dtype=torch.long) * query_width
            + batch_result.accept_lens.long()
        )
        logits_output.next_token_logits = logits_output.next_token_logits[
            query_select_indices
        ]
        if logits_output.hidden_states is not None:
            logits_output.hidden_states = logits_output.hidden_states[
                query_select_indices
            ]
        if draft_extend_profiler is not None:
            draft_extend_profiler.step()
            draft_extend_profiler.stop()
            profile_output = Path(profile_path)
            profile_output.parent.mkdir(parents=True, exist_ok=True)
            draft_extend_profiler.export_chrome_trace(str(profile_output))
        return logits_output

    def _audit_mixed_draft_extend_schedule(
        self,
        *,
        batch,
        batch_result,
        forward_batch,
        draft_segment,
        normal_input_ids,
        normal_positions,
        normal_out_cache_locs,
        reference_hidden,
        reference_logits,
    ) -> None:
        """Validate full, continuation, draft-extend, and optional decode."""
        target_runner = self.target_worker.model_runner
        target_model = target_runner.model
        artifact = getattr(
            target_model,
            "depthspec_ragged_draft_extend_audit",
            None,
        )
        if artifact is None:
            raise ValueError("Mixed draft-extend audit missing target artifacts")
        if artifact.get("reference_continuation_state") is None:
            raise ValueError("Mixed draft-extend audit missing continuation state")

        def tensor_error(left: torch.Tensor, right: torch.Tensor) -> dict:
            difference = (left.float() - right.float()).abs()
            return {
                "max_abs": float(difference.max()),
                "mean_abs": float(difference.mean()),
                "relative_l2": float(
                    (left.float() - right.float()).norm()
                    / right.float().norm().clamp_min(1e-12)
                ),
            }

        full_segment = artifact["full_segment"]
        continuation_segment = artifact["continuation_segment"]
        verify_forward_batch = artifact["verify_forward_batch"]
        full_stem_state = artifact["full_stem_state"]
        reference_continuation_state = artifact["reference_continuation_state"]
        normal_decode_enabled = bool(
            self.runtime_config.diagnostics.get(
                "ragged_normal_decode_audit", False
            )
        )
        decode_segment = None
        decode_forward_batch = None
        reference_decode_state = None
        reference_decode_logits = None
        if normal_decode_enabled:
            from sglang.srt.model_executor.forward_batch_info import (
                CaptureHiddenMode,
                ForwardBatch,
                ForwardMode,
            )

            gpt = target_model.gpt
            decode_loops = tuple(range(int(gpt.target_loop_count)))
            decode_schedule = tuple(
                float(value)
                for value in gpt.step_schedule[: gpt.target_loop_count]
            )
            saved_decode_batch_state = {
                "input_ids": batch.input_ids,
                "out_cache_loc": batch.out_cache_loc,
                "out_cache_loc_dsv4": batch.out_cache_loc_dsv4,
                "seq_lens_sum": batch.seq_lens_sum,
                "forward_mode": batch.forward_mode,
                "spec_info": batch.spec_info,
            }
            batch.input_ids = normal_input_ids
            batch.out_cache_loc = normal_out_cache_locs
            batch.out_cache_loc_dsv4 = None
            batch.seq_lens_sum = int(batch.seq_lens.sum().item())
            batch.forward_mode = ForwardMode.DECODE
            batch.spec_info = None
            try:
                decode_forward_batch = ForwardBatch.init_new(
                    batch,
                    target_runner,
                    capture_hidden_mode=CaptureHiddenMode.NULL,
                    return_hidden_states_before_norm=False,
                )
            finally:
                for name, value in saved_decode_batch_state.items():
                    setattr(batch, name, value)

            decode_segment = LoopSegment(
                input_ids=normal_input_ids,
                positions=normal_positions,
                out_cache_loc=normal_out_cache_locs,
                seq_lens=batch.seq_lens,
                seq_lens_cpu=(
                    batch.seq_lens_cpu
                    if batch.seq_lens_cpu is not None
                    else batch.seq_lens.cpu()
                ),
                seq_lens_sum=int(batch.seq_lens.sum().item()),
                spec_info=None,
                req_pool_indices=batch.req_pool_indices,
                phase="normal_decode",
                loop_indices=decode_loops,
                step_schedule=decode_schedule,
                attention_contract="causal-decode",
            )
            target_runner.attn_backend.init_forward_metadata(
                decode_forward_batch
            )
            decode_forward_batch.mark_forward_metadata_ready()
            decode_reference_plan = build_heterogeneous_ragged_forward_plan(
                [
                    RaggedSegmentInput(
                        decode_segment,
                        decode_forward_batch,
                    )
                ]
            )
            decode_reference_result = execute_grouped_ragged_schedule(
                target_model,
                target_runner.attn_backend,
                decode_reference_plan,
                shared_metadata=True,
            )
            reference_decode_state = decode_reference_result.group_states[0]
            reference_decode_normalized = target_model.gpt.transformer.norm_f(
                reference_decode_state.hidden_states
            )
            reference_decode_logits = target_model.logits_processor(
                normal_input_ids,
                reference_decode_normalized,
                target_model.lm_head,
                decode_forward_batch,
            ).next_token_logits

        segment_inputs = [
            RaggedSegmentInput(full_segment, verify_forward_batch),
            RaggedSegmentInput(
                continuation_segment,
                verify_forward_batch,
                initial_hidden_states=full_stem_state.hidden_states,
                initial_time=full_stem_state.current_time,
            ),
            RaggedSegmentInput(draft_segment, forward_batch),
        ]
        if normal_decode_enabled:
            segment_inputs.append(
                RaggedSegmentInput(decode_segment, decode_forward_batch)
            )
        plan = build_heterogeneous_ragged_forward_plan(
            segment_inputs
        )
        schedule = compile_ragged_schedule(plan)
        started = time.perf_counter()
        mixed_result = execute_grouped_ragged_schedule(
            target_model,
            target_runner.attn_backend,
            plan,
            shared_metadata=bool(
                self.runtime_config.diagnostics.get(
                    "shared_ragged_attention_metadata", True
                )
            ),
        )

        draft_raw_hidden = mixed_result.group_states[2].hidden_states
        draft_normalized_hidden = target_model.gpt.transformer.norm_f(
            draft_raw_hidden
        )
        draft_logits_output = target_model.logits_processor(
            draft_segment.input_ids,
            draft_normalized_hidden,
            target_model.lm_head,
            forward_batch,
        )
        draft_width = int(draft_segment.input_ids.numel() // batch.seq_lens.numel())
        selected_indices = (
            torch.arange(
                int(batch.seq_lens.numel()), device=batch.device, dtype=torch.long
            )
            * draft_width
            + batch_result.accept_lens.long()
        )
        selected_draft_logits = draft_logits_output.next_token_logits[
            selected_indices
        ]
        selected_reference_logits = reference_logits[selected_indices]
        if normal_decode_enabled:
            decode_raw_hidden = mixed_result.group_states[3].hidden_states
            decode_normalized_hidden = target_model.gpt.transformer.norm_f(
                decode_raw_hidden
            )
            decode_logits = target_model.logits_processor(
                decode_segment.input_ids,
                decode_normalized_hidden,
                target_model.lm_head,
                decode_forward_batch,
            ).next_token_logits

        row = {
            "captured_unix_s": time.time(),
            "elapsed_s": time.perf_counter() - started,
            "request_count": int(batch.seq_lens.numel()),
            "qo_indptr": plan.qo_indptr.detach().cpu().tolist(),
            "padding_rows": plan.padding_rows,
            "full_payload_rows": int(full_segment.input_ids.numel()),
            "continuation_payload_rows": int(
                continuation_segment.input_ids.numel()
            ),
            "draft_extend_payload_rows": int(draft_segment.input_ids.numel()),
            "normal_decode_payload_rows": (
                int(decode_segment.input_ids.numel())
                if normal_decode_enabled
                else 0
            ),
            "micro_step_count": len(schedule),
            "micro_step_rows": [item.row_count for item in schedule],
            "micro_step_loops": [
                [item.loop_index for item in step.active_segments]
                for step in schedule
            ],
            "kv_namespaces": plan.kv_namespaces,
            "full_hidden_error": tensor_error(
                mixed_result.group_states[0].hidden_states,
                full_stem_state.hidden_states,
            ),
            "continuation_hidden_error": tensor_error(
                mixed_result.group_states[1].hidden_states,
                reference_continuation_state.hidden_states,
            ),
            "draft_extend_hidden_error": (
                tensor_error(
                    draft_normalized_hidden,
                    reference_hidden,
                )
                if reference_hidden is not None
                else None
            ),
            "draft_extend_all_logits_error": tensor_error(
                draft_logits_output.next_token_logits,
                reference_logits,
            ),
            "draft_extend_selected_logits_error": tensor_error(
                selected_draft_logits,
                selected_reference_logits,
            ),
            "normal_decode_hidden_error": (
                tensor_error(
                    mixed_result.group_states[3].hidden_states,
                    reference_decode_state.hidden_states,
                )
                if normal_decode_enabled
                else None
            ),
            "normal_decode_logits_error": (
                tensor_error(decode_logits, reference_decode_logits)
                if normal_decode_enabled
                else None
            ),
        }
        audit_path = self.runtime_config.diagnostics.get(
            "ragged_draft_extend_audit_path"
        )
        if audit_path:
            output = Path(audit_path)
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row) + "\n")
        else:
            logger.info("DepthSpec mixed draft-extend ragged audit: %s", row)
        del target_model.depthspec_ragged_draft_extend_audit

    def _audit_target_draft_shared_prefix_kv(
        self,
        batch,
        schedule_batch,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        out_cache_locs: torch.Tensor,
        active_width: int,
    ) -> None:
        """Compare target loops0/1 KV with draft loops9/10 transition KV.

        Draft-extend transition j consumes target's prediction after accepted
        node j. Under the candidate shared-prefix mapping, its closest target
        source is accepted tree node j+1. This audit never writes either KV pool.
        """
        audit_path = self.runtime_config.diagnostics.get(
            "target_draft_kv_audit_path"
        )
        if not audit_path:
            return
        marker_path = f"{audit_path}.count"
        try:
            audit_index = int(Path(marker_path).read_text(encoding="utf-8"))
        except (FileNotFoundError, ValueError):
            audit_index = 0
        max_rows = int(
            self.runtime_config.diagnostics.get("target_draft_kv_audit_max", 256)
        )
        if audit_index >= max_rows:
            return

        next_draft_input = batch.next_draft_input
        accept_index = getattr(next_draft_input, "accept_index", None)
        candidate_tokens = getattr(next_draft_input, "candidate_tokens", None)
        candidate_positions = getattr(next_draft_input, "candidate_positions", None)
        if (
            accept_index is None
            or candidate_tokens is None
            or candidate_positions is None
        ):
            raise ValueError("Target/draft KV audit requires accepted-path metadata")
        local_accept_index = _localize_tree_accept_index(
            accept_index,
            candidate_tokens.shape[1],
        )

        target_runner = self.target_worker.model_runner
        pool = target_runner.token_to_kv_pool
        model = target_runner.model
        blocks_per_loop = int(model.config.num_hidden_layers)
        draft_first_loop = model.gpt.loop_num - 3
        request_rows = self.req_to_token_pool.req_to_token
        rows = []

        for request_index in range(len(schedule_batch.seq_lens)):
            accept_length = min(
                int(batch.accept_lens[request_index].item()),
                active_width,
            )
            seq_len = int(schedule_batch.seq_lens[request_index].item())
            req_pool_index = int(schedule_batch.req_pool_indices[request_index].item())
            for transition_index in range(max(accept_length - 1, 0)):
                flat_draft_row = request_index * active_width + transition_index
                accepted_depth = transition_index + 1
                source_tree_row = int(
                    local_accept_index[request_index, accepted_depth].item()
                )
                source_position = int(
                    candidate_positions[request_index, source_tree_row].item()
                )
                target_slot = int(request_rows[req_pool_index, source_position].item())
                draft_slot = int(out_cache_locs[flat_draft_row].item())
                draft_token = int(input_ids[flat_draft_row].item())
                target_token = int(
                    candidate_tokens[request_index, source_tree_row].item()
                )
                loop_metrics = {}
                for loop_offset in range(2):
                    target_loop = loop_offset
                    draft_loop = draft_first_loop + loop_offset
                    key_errors = []
                    value_errors = []
                    for block_index in range(blocks_per_loop):
                        target_layer = target_loop * blocks_per_loop + block_index
                        draft_layer = draft_loop * blocks_per_loop + block_index
                        target_key = pool.get_key_buffer(target_layer)[target_slot]
                        draft_key = pool.get_key_buffer(draft_layer)[draft_slot]
                        target_value = pool.get_value_buffer(target_layer)[target_slot]
                        draft_value = pool.get_value_buffer(draft_layer)[draft_slot]
                        key_difference = (target_key - draft_key).float()
                        value_difference = (target_value - draft_value).float()
                        key_errors.append(
                            {
                                "block": block_index,
                                "max_abs": float(key_difference.abs().max()),
                                "mean_abs": float(key_difference.abs().mean()),
                                "relative_l2": float(
                                    key_difference.norm()
                                    / draft_key.float().norm().clamp_min(1e-12)
                                ),
                            }
                        )
                        value_errors.append(
                            {
                                "block": block_index,
                                "max_abs": float(value_difference.abs().max()),
                                "mean_abs": float(value_difference.abs().mean()),
                                "relative_l2": float(
                                    value_difference.norm()
                                    / draft_value.float().norm().clamp_min(1e-12)
                                ),
                            }
                        )
                    loop_metrics[f"target_loop{target_loop}_draft_loop{draft_loop}"] = {
                        "key": key_errors,
                        "value": value_errors,
                    }

                rows.append(
                    {
                        "audit_index": audit_index,
                        "request_index": request_index,
                        "transition_index": transition_index,
                        "accept_length": accept_length,
                        "seq_len": seq_len,
                        "draft_position": int(positions[flat_draft_row].item()),
                        "target_source_position": source_position,
                        "position_delta": source_position
                        - int(positions[flat_draft_row].item()),
                        "draft_slot": draft_slot,
                        "target_source_slot": target_slot,
                        "draft_token": draft_token,
                        "target_source_token": target_token,
                        "token_match": draft_token == target_token,
                        "loop_metrics": loop_metrics,
                    }
                )
                audit_index += 1
                if audit_index >= max_rows:
                    break
            if audit_index >= max_rows:
                break

        if rows:
            output_path = Path(audit_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("a", encoding="utf-8") as stream:
                for row in rows:
                    stream.write(json.dumps(row) + "\n")
        Path(marker_path).write_text(str(audit_index), encoding="utf-8")

    def _compute_causal_loopformer_kv(
        self,
        model,
        token_ids: list[int],
        position_ids: list[int],
    ) -> tuple[torch.Tensor, list[list[tuple[torch.Tensor, torch.Tensor]]]]:
        """Eagerly compute two causal LoopFormer loops and return per-block KV.

        This helper is audit-only. It uses the shared block weights directly and
        does not read or write either SGLang KV pool.
        """
        gpt = model.gpt
        device = next(gpt.parameters()).device
        dtype = next(gpt.parameters()).dtype
        token_tensor = torch.tensor([token_ids], dtype=torch.long, device=device)
        position_tensor = torch.tensor([position_ids], dtype=torch.long, device=device)
        hidden = gpt.transformer.wte(token_tensor) + gpt.transformer.wpe(
            position_tensor
        )
        hidden = gpt.transformer.drop(hidden)
        current_time = 0.0
        recorded_kv: list[list[tuple[torch.Tensor, torch.Tensor]]] = []

        with torch.no_grad():
            for loop_index in range(2):
                step_size = gpt.step_schedule[loop_index]
                time_values = hidden.new_full(
                    (hidden.shape[1],), current_time, dtype=torch.float32
                )
                step_values = hidden.new_full(
                    (hidden.shape[1],), step_size, dtype=torch.float32
                )
                conditioning = gpt.time_embedder(time_values) + gpt.dt_embedder(
                    step_values
                )
                loop_kv: list[tuple[torch.Tensor, torch.Tensor]] = []
                for block in gpt.transformer.h.blocks:
                    gate_msa, gate_mlp, scale_msa, scale_mlp = block.adaLN_modulation(
                        conditioning
                    ).chunk(4, dim=-1)
                    normalized = block.norm_1(hidden[0]) * (1 + scale_msa)
                    qkv = block.attn.c_attn(normalized)
                    query, key, value = qkv.split(
                        [
                            block.attn.q_size,
                            block.attn.kv_size,
                            block.attn.kv_size,
                        ],
                        dim=-1,
                    )
                    query = query.view(
                        key.shape[0],
                        block.attn.num_heads,
                        block.attn.head_dim,
                    )
                    key = key.view(
                        key.shape[0],
                        block.attn.num_kv_heads,
                        block.attn.head_dim,
                    )
                    value = value.view(
                        value.shape[0],
                        block.attn.num_kv_heads,
                        block.attn.head_dim,
                    )
                    loop_kv.append((key, value))

                    scores = torch.einsum(
                        "qhd,khd->hqk", query.float(), key.float()
                    ) * (block.attn.head_dim**-0.5)
                    causal_mask = torch.ones(
                        key.shape[0],
                        key.shape[0],
                        dtype=torch.bool,
                        device=device,
                    ).tril()
                    scores = scores.masked_fill(~causal_mask[None], float("-inf"))
                    probabilities = torch.softmax(scores, dim=-1).to(dtype)
                    attended = torch.einsum(
                        "hqk,khd->qhd", probabilities, value
                    ).reshape(key.shape[0], -1)
                    attention_output = block.attn.c_proj(attended)

                    hidden = hidden + gate_msa[None] * attention_output[None]
                    hidden = hidden + gate_mlp[None] * block.mlp(
                        block.norm_2(hidden) * (1 + scale_mlp[None])
                    )
                recorded_kv.append(loop_kv)
                current_time += step_size

        return hidden[0], recorded_kv

    @staticmethod
    def _tensor_error(left: torch.Tensor, right: torch.Tensor) -> dict:
        difference = (left.float() - right.float()).abs()
        return {
            "max_abs": float(difference.max()),
            "mean_abs": float(difference.mean()),
            "relative_l2": float(
                (left.float() - right.float()).norm()
                / right.float().norm().clamp_min(1e-12)
            ),
        }

    def _audit_kv_source_factorial(
        self,
        batch_result,
        schedule_batch,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        out_cache_locs: torch.Tensor,
        active_width: int,
    ) -> None:
        """Attribute target/draft KV mismatch to position, root, and runtime."""
        audit_path = self.runtime_config.diagnostics.get(
            "kv_source_factorial_audit_path"
        )
        if not audit_path:
            return
        marker_path = f"{audit_path}.count"
        try:
            audit_index = int(Path(marker_path).read_text(encoding="utf-8"))
        except (FileNotFoundError, ValueError):
            audit_index = 0
        max_rows = int(
            self.runtime_config.diagnostics.get("kv_source_factorial_audit_max", 64)
        )
        if audit_index >= max_rows:
            return

        next_draft_input = batch_result.next_draft_input
        accept_index = getattr(next_draft_input, "accept_index", None)
        candidate_tokens = getattr(next_draft_input, "candidate_tokens", None)
        candidate_positions = getattr(next_draft_input, "candidate_positions", None)
        if (
            accept_index is None
            or candidate_tokens is None
            or candidate_positions is None
        ):
            raise ValueError("Factorial KV audit requires accepted-path metadata")
        local_accept_index = _localize_tree_accept_index(
            accept_index,
            candidate_tokens.shape[1],
        )

        target_runner = self.target_worker.model_runner
        model = target_runner.model
        pool = target_runner.token_to_kv_pool
        blocks_per_loop = int(model.config.num_hidden_layers)
        draft_first_loop = model.gpt.loop_num - 3
        request_rows = self.req_to_token_pool.req_to_token
        predicted_tokens = batch_result.next_token_ids.reshape(
            len(schedule_batch.seq_lens),
            self.speculative_num_draft_tokens,
        )
        rows = []

        for request_index in range(len(schedule_batch.reqs)):
            accept_length = min(
                int(batch_result.accept_lens[request_index].item()),
                active_width,
            )
            transition_count = max(accept_length - 1, 0)
            if not transition_count:
                continue
            request = schedule_batch.reqs[request_index]
            root_tree_row = int(local_accept_index[request_index, 0].item())
            root_position = int(
                candidate_positions[request_index, root_tree_row].item()
            )
            full_ids = list(request.origin_input_ids) + list(request.output_ids)
            context_ids = [int(value) for value in full_ids[:root_position]]
            context_positions = list(range(len(context_ids)))
            root_token = int(candidate_tokens[request_index, root_tree_row].item())
            transition_tokens = [
                int(predicted_tokens[request_index, index].item())
                for index in range(transition_count)
            ]

            native_tokens = context_ids + transition_tokens
            native_positions = context_positions + [
                root_position + index for index in range(transition_count)
            ]
            shifted_positions = context_positions + [
                root_position + 1 + index for index in range(transition_count)
            ]
            rooted_tokens = context_ids + [root_token] + transition_tokens
            rooted_native_positions = context_positions + [
                root_position,
                *(root_position + index for index in range(transition_count)),
            ]
            rooted_shifted_positions = context_positions + [
                root_position,
                *(root_position + 1 + index for index in range(transition_count)),
            ]

            variants = {
                "draft_native": (native_tokens, native_positions),
                "draft_plus_position": (native_tokens, shifted_positions),
                "draft_plus_root": (
                    rooted_tokens,
                    rooted_native_positions,
                ),
                "draft_plus_position_root": (
                    rooted_tokens,
                    rooted_shifted_positions,
                ),
            }
            computed = {
                name: self._compute_causal_loopformer_kv(model, tokens, pos)
                for name, (tokens, pos) in variants.items()
            }
            repeat_native = self._compute_causal_loopformer_kv(
                model,
                variants["draft_native"][0],
                variants["draft_native"][1],
            )

            for transition_index in range(transition_count):
                if audit_index >= max_rows:
                    break
                accepted_depth = transition_index + 1
                source_tree_row = int(
                    local_accept_index[request_index, accepted_depth].item()
                )
                source_position = int(
                    candidate_positions[request_index, source_tree_row].item()
                )
                req_pool_index = int(
                    schedule_batch.req_pool_indices[request_index].item()
                )
                target_slot = int(request_rows[req_pool_index, source_position].item())
                flat_draft_row = request_index * active_width + transition_index
                draft_slot = int(out_cache_locs[flat_draft_row].item())
                native_row = len(context_ids) + transition_index
                rooted_row = len(context_ids) + 1 + transition_index
                variant_rows = {
                    "draft_native": native_row,
                    "draft_plus_position": native_row,
                    "draft_plus_root": rooted_row,
                    "draft_plus_position_root": rooted_row,
                }

                variant_metrics = {}
                for variant_name, (_, variant_kv) in computed.items():
                    row_index = variant_rows[variant_name]
                    loop_metrics = {}
                    for loop_offset in range(2):
                        key_errors = []
                        value_errors = []
                        draft_key_errors = []
                        draft_value_errors = []
                        for block_index in range(blocks_per_loop):
                            target_layer = loop_offset * blocks_per_loop + block_index
                            draft_layer = (
                                draft_first_loop + loop_offset
                            ) * blocks_per_loop + block_index
                            target_key = pool.get_key_buffer(target_layer)[target_slot]
                            target_value = pool.get_value_buffer(target_layer)[
                                target_slot
                            ]
                            draft_key = pool.get_key_buffer(draft_layer)[draft_slot]
                            draft_value = pool.get_value_buffer(draft_layer)[draft_slot]
                            variant_key, variant_value = variant_kv[loop_offset][
                                block_index
                            ]
                            key_errors.append(
                                self._tensor_error(variant_key[row_index], target_key)
                            )
                            value_errors.append(
                                self._tensor_error(
                                    variant_value[row_index], target_value
                                )
                            )
                            draft_key_errors.append(
                                self._tensor_error(variant_key[row_index], draft_key)
                            )
                            draft_value_errors.append(
                                self._tensor_error(
                                    variant_value[row_index], draft_value
                                )
                            )
                        loop_metrics[f"loop{loop_offset}"] = {
                            "vs_target_key": key_errors,
                            "vs_target_value": value_errors,
                            "vs_draft_key": draft_key_errors,
                            "vs_draft_value": draft_value_errors,
                        }
                    variant_metrics[variant_name] = loop_metrics

                repeat_metrics = {}
                for loop_offset in range(2):
                    key_errors = []
                    for block_index in range(blocks_per_loop):
                        original_key, _ = computed["draft_native"][1][loop_offset][
                            block_index
                        ]
                        repeat_key, _ = repeat_native[1][loop_offset][block_index]
                        key_errors.append(
                            self._tensor_error(
                                repeat_key[native_row], original_key[native_row]
                            )
                        )
                    repeat_metrics[f"loop{loop_offset}"] = key_errors

                prefix_comparison = {}
                context_variant = computed["draft_plus_position_root"][1]
                for loop_offset in range(2):
                    loop_comparison = {}
                    for block_index in range(blocks_per_loop):
                        target_layer = loop_offset * blocks_per_loop + block_index
                        draft_layer = (
                            draft_first_loop + loop_offset
                        ) * blocks_per_loop + block_index
                        tensor_metrics = {}
                        for tensor_name in ("key", "value"):
                            custom_vs_target = []
                            custom_vs_draft = []
                            target_vs_draft = []
                            worst_custom_vs_draft = None
                            worst_position = -1
                            for context_row in range(len(context_ids)):
                                context_slot = int(
                                    request_rows[req_pool_index, context_row].item()
                                )
                                custom_key, custom_value = context_variant[loop_offset][
                                    block_index
                                ]
                                custom_tensor = (
                                    custom_key if tensor_name == "key" else custom_value
                                )
                                target_tensor = (
                                    pool.get_key_buffer(target_layer)[context_slot]
                                    if tensor_name == "key"
                                    else pool.get_value_buffer(target_layer)[
                                        context_slot
                                    ]
                                )
                                draft_tensor = (
                                    pool.get_key_buffer(draft_layer)[context_slot]
                                    if tensor_name == "key"
                                    else pool.get_value_buffer(draft_layer)[
                                        context_slot
                                    ]
                                )
                                target_error = self._tensor_error(
                                    custom_tensor[context_row], target_tensor
                                )
                                draft_error = self._tensor_error(
                                    custom_tensor[context_row], draft_tensor
                                )
                                direct_error = self._tensor_error(
                                    target_tensor, draft_tensor
                                )
                                custom_vs_target.append(target_error["relative_l2"])
                                custom_vs_draft.append(draft_error["relative_l2"])
                                target_vs_draft.append(direct_error["relative_l2"])
                                if worst_custom_vs_draft is None or (
                                    draft_error["relative_l2"] > worst_custom_vs_draft
                                ):
                                    worst_custom_vs_draft = draft_error["relative_l2"]
                                    worst_position = context_row
                            ordered_custom_vs_draft = sorted(custom_vs_draft)
                            tensor_metrics[tensor_name] = {
                                "custom_vs_target_relative_l2": {
                                    "mean": sum(custom_vs_target)
                                    / len(custom_vs_target),
                                    "max": max(custom_vs_target),
                                },
                                "custom_vs_draft_relative_l2": {
                                    "mean": sum(custom_vs_draft) / len(custom_vs_draft),
                                    "p90": ordered_custom_vs_draft[
                                        round(0.9 * (len(ordered_custom_vs_draft) - 1))
                                    ],
                                    "max": max(custom_vs_draft),
                                    "worst_position": worst_position,
                                },
                                "target_vs_draft_relative_l2": {
                                    "mean": sum(target_vs_draft) / len(target_vs_draft),
                                    "max": max(target_vs_draft),
                                },
                            }
                        loop_comparison[f"block{block_index}"] = {
                            "count": len(context_ids),
                            **tensor_metrics,
                        }
                    prefix_comparison[f"loop{loop_offset}"] = loop_comparison

                rows.append(
                    {
                        "audit_index": audit_index,
                        "request_index": request_index,
                        "transition_index": transition_index,
                        "accept_length": accept_length,
                        "context_length": len(context_ids),
                        "root_position": root_position,
                        "source_position": source_position,
                        "draft_position": int(positions[flat_draft_row].item()),
                        "root_token": root_token,
                        "transition_token": transition_tokens[transition_index],
                        "target_source_token": int(
                            candidate_tokens[request_index, source_tree_row].item()
                        ),
                        "token_match": transition_tokens[transition_index]
                        == int(candidate_tokens[request_index, source_tree_row].item()),
                        "variant_metrics": variant_metrics,
                        "deterministic_repeat_key": repeat_metrics,
                        "prefix_comparison": prefix_comparison,
                    }
                )
                audit_index += 1
            if audit_index >= max_rows:
                break

        if rows:
            output_path = Path(audit_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("a", encoding="utf-8") as stream:
                for row in rows:
                    stream.write(json.dumps(row) + "\n")
        Path(marker_path).write_text(str(audit_index), encoding="utf-8")

    def _compute_mini_transition_path(self, batch, batch_result) -> None:
        """Recompute loops9/10 only on each accepted transition path."""
        from sglang.srt.model_executor.forward_batch_info import (
            ForwardBatch,
            ForwardMode,
        )
        from sglang.srt.model_executor.forward_context import (
            ForwardContext,
            forward_context,
        )
        from sglang.srt.speculative.eagle_info import EagleDraftExtendInput

        target_runner = self.target_worker.model_runner
        target_model = target_runner.model
        relay_hidden = getattr(target_model, "depthspec_relay_hidden_states", None)
        if relay_hidden is None:
            raise ValueError("Mini transition relay requires relay hidden states")

        bs = len(batch.seq_lens)
        num_draft_tokens = self.speculative_num_draft_tokens
        predicted_tokens = batch_result.next_token_ids.reshape(
            bs,
            num_draft_tokens,
        )
        input_ids = []
        positions = []
        out_cache_locs = []
        prefix_lens = []
        extend_lens = []
        active_lengths = []
        rows = self.req_to_token_pool.req_to_token

        for request_index in range(bs):
            active_length = min(
                int(batch_result.accept_lens[request_index].item()),
                num_draft_tokens,
            )
            active_lengths.append(active_length)
            seq_len = int(batch.seq_lens[request_index].item())
            req_pool_index = int(batch.req_pool_indices[request_index].item())
            prefix_lens.append(seq_len)
            extend_lens.append(active_length)
            for transition_index in range(active_length):
                input_ids.append(
                    int(predicted_tokens[request_index, transition_index].item())
                )
                position = seq_len + transition_index
                positions.append(position)
                out_cache_locs.append(int(rows[req_pool_index, position].item()))

        if not input_ids:
            return
        device = self.device
        saved_state = {
            "input_ids": batch.input_ids,
            "out_cache_loc": batch.out_cache_loc,
            "prefix_lens": getattr(batch, "prefix_lens", None),
            "extend_lens": getattr(batch, "extend_lens", None),
            "extend_num_tokens": getattr(batch, "extend_num_tokens", None),
            "seq_lens_sum": getattr(batch, "seq_lens_sum", None),
            "forward_mode": batch.forward_mode,
            "spec_info": batch.spec_info,
        }
        old_attention_backend = target_runner.attn_backend
        target_runner.attn_backend = self.draft_extend_attn_backend
        batch.input_ids = torch.tensor(input_ids, dtype=torch.int64, device=device)
        batch.out_cache_loc = torch.tensor(
            out_cache_locs, dtype=torch.int64, device=device
        )
        batch.prefix_lens = prefix_lens
        batch.extend_lens = extend_lens
        batch.extend_num_tokens = len(input_ids)
        batch.forward_mode = ForwardMode.DRAFT_EXTEND_V2
        batch.spec_info = EagleDraftExtendInput(
            positions=torch.tensor(positions, dtype=torch.int64, device=device),
            num_tokens_per_req=num_draft_tokens,
            num_tokens_for_logprob_per_req=1,
        )

        try:
            forward_batch = ForwardBatch.init_new(
                batch,
                self.draft_runner,
                capture_hidden_mode=None,
                return_hidden_states_before_norm=False,
            )
            self.draft_extend_attn_backend.init_forward_metadata(forward_batch)
            forward_batch.mark_forward_metadata_ready()
            gpt = target_model.gpt
            with forward_context(
                ForwardContext(attn_backend=self.draft_extend_attn_backend)
            ):
                _, path_prefix_state = gpt.forward_branch(
                    forward_batch.input_ids,
                    forward_batch.positions,
                    forward_batch,
                    loop_indices=(gpt.loop_num - 3, gpt.loop_num - 2),
                    step_schedule=(0.125, 0.125),
                    return_state=True,
                )
            path_row = 0
            for request_index, active_length in enumerate(active_lengths):
                for transition_index in range(active_length):
                    flat_row = request_index * num_draft_tokens + transition_index
                    relay_hidden[flat_row].copy_(
                        path_prefix_state.hidden_states[path_row]
                    )
                    path_row += 1
        finally:
            target_runner.attn_backend = old_attention_backend
            batch.input_ids = saved_state["input_ids"]
            batch.out_cache_loc = saved_state["out_cache_loc"]
            batch.prefix_lens = saved_state["prefix_lens"]
            batch.extend_lens = saved_state["extend_lens"]
            batch.extend_num_tokens = saved_state["extend_num_tokens"]
            batch.seq_lens_sum = saved_state["seq_lens_sum"]
            batch.forward_mode = saved_state["forward_mode"]
            batch.spec_info = saved_state["spec_info"]

    def _prepare_draft_artifact_relay(self, batch, result) -> None:
        """Materialize a coarse-only initial state from accepted draft artifacts."""
        if not self.runtime_config.diagnostics.get("draft_artifact_relay", False):
            return
        if not self.unified_runner:
            return

        target_model = self.target_worker.model_runner.model
        artifacts = getattr(target_model.gpt, "depthspec_relay_artifacts", None)
        target_state = getattr(target_model, "last_prefix_state", None)
        next_draft_input = result.next_draft_input
        accept_index = getattr(next_draft_input, "accept_index", None)
        candidate_tokens = getattr(next_draft_input, "candidate_tokens", None)
        candidate_positions = getattr(next_draft_input, "candidate_positions", None)
        if (
            artifacts is None
            or target_state is None
            or accept_index is None
            or candidate_tokens is None
            or candidate_positions is None
        ):
            logger.warning(
                "DepthSpec draft artifact relay missing required state; fallback"
            )
            target_model.depthspec_relay_hidden_states = None
            return

        bs = len(batch.seq_lens)
        num_draft_tokens = self.speculative_num_draft_tokens
        local_accept_index = _localize_tree_accept_index(
            accept_index,
            candidate_tokens.shape[1],
        )
        relay_hidden = torch.zeros_like(target_state.hidden_states)
        artifact_matches = 0
        target_fallbacks = 0
        predicted_tokens = result.next_token_ids.reshape(
            bs,
            num_draft_tokens,
        )
        for request_index in range(bs):
            accept_length = min(
                int(result.accept_lens[request_index].item()),
                accept_index.shape[1],
            )
            for output_index in range(accept_length):
                # Draft-extend row j consumes target's prediction after accepted
                # node j. Match the proposal artifact for that predicted token at
                # the source node position (a transition, not the tree node).
                global_tree_row = int(
                    accept_index[request_index, output_index].item()
                )
                local_tree_row = int(
                    local_accept_index[request_index, output_index].item()
                )
                flat_row = request_index * num_draft_tokens + output_index
                position = int(
                    candidate_positions[request_index, local_tree_row].item()
                )
                token_id = int(predicted_tokens[request_index, output_index].item())
                artifact_position = position
                artifact = next(
                    (
                        item
                        for item in artifacts
                        if item["token_id"] == token_id
                        and item["position"] == artifact_position
                    ),
                    None,
                )
                if artifact is None:
                    # Target's two-loop state is only a fallback for tokens not
                    # present in the current proposal artifact set.
                    relay_hidden[flat_row].copy_(
                        target_state.hidden_states[global_tree_row]
                    )
                    target_fallbacks += 1
                else:
                    relay_hidden[flat_row].copy_(artifact["hidden_states"])
                    artifact_matches += 1

        target_model.depthspec_prefix_handoff_hidden_states = relay_hidden
        target_model.depthspec_relay_hidden_states = relay_hidden
        target_model.depthspec_relay_stats = {
            "artifact_match_count": artifact_matches,
            "target_fallback_count": target_fallbacks,
        }
        logger.info(
            "DepthSpec draft artifact relay: artifacts=%d matches=%d fallbacks=%d",
            len(artifacts),
            artifact_matches,
            target_fallbacks,
        )
        # Artifacts are only useful for this verify round; release them before the
        # next proposal repopulates the list.
        target_model.gpt.depthspec_relay_artifacts = []

    def _draft_extend_for_prefill(
        self, batch, target_hidden_states, next_token_ids, mm_input_embeds=None
    ):
        context = (
            self._use_unified_draft_forward(forward_purpose="draft_extend_prefill")
            if self.unified_runner
            else empty_context()
        )
        with context:
            return super()._draft_extend_for_prefill(
                batch,
                target_hidden_states,
                next_token_ids,
                mm_input_embeds=mm_input_embeds,
            )

    def alloc_memory_pool(
        self,
        memory_pool_config=None,
        req_to_token_pool=None,
        token_to_kv_pool_allocator=None,
    ):
        """Standalone: allocate pools without sharing embeddings."""
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        if self.unified_runner:
            target_runner = self.target_worker.model_runner
            if req_to_token_pool is not target_runner.req_to_token_pool:
                raise RuntimeError("Unified DepthSpec runner lost the target pool")
            if (
                token_to_kv_pool_allocator
                is not target_runner.token_to_kv_pool_allocator
            ):
                raise RuntimeError(
                    "Unified DepthSpec runner lost the target KV allocator"
                )
            return
        self.draft_worker.alloc_memory_pool(
            memory_pool_config=memory_pool_config,
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=token_to_kv_pool_allocator,
        )
        self.init_token_map()
        self.init_lm_head()

    def init_attention_backends(self):
        original_target_backend = None
        if self.unified_runner:
            original_target_backend = self.target_worker.model_runner.attn_backend
        with (
            self.draft_tp_context(self.draft_runner.tp_group),
            speculative_moe_backend_context(),
        ):
            super().init_attention_backends()
        if self.unified_runner:
            self._unified_target_attn_backend = original_target_backend
            self.target_attn_backend = original_target_backend
            self.target_worker.model_runner.attn_backend = original_target_backend

    def init_cuda_graphs(self):
        with (
            self.draft_tp_context(self.draft_runner.tp_group),
            speculative_moe_backend_context(),
        ):
            super().init_cuda_graphs()

    def init_lm_head(self):
        """Override to prevent sharing embeddings and lm_head with target model."""
        # For standalone worker, we don't share embeddings and lm_head
        # The draft model uses its own embeddings and lm_head
        return


class StandaloneWorkerV2(EAGLEWorkerV2):
    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        ps: ParallelState,
        nccl_port: int,
        target_worker: TpModelWorker,
    ):
        BaseSpecWorker.__init__(self)

        # Parse arguments
        self.server_args = server_args
        self.topk = server_args.speculative_eagle_topk
        self.speculative_num_steps = server_args.speculative_num_steps
        self.speculative_num_draft_tokens = server_args.speculative_num_draft_tokens
        self.gpu_id = gpu_id
        self.device = server_args.device
        self._target_worker = target_worker
        self.page_size = get_schedule().page_size
        self.speculative_algorithm = SpeculativeAlgorithm.from_string(
            server_args.speculative_algorithm
        )

        # Create our custom draft worker that doesn't share embeddings/lm_head
        self._draft_worker = StandaloneDraftWorker(
            server_args,
            gpu_id,
            ps,
            nccl_port,
            target_worker,
        )
        self.runtime_config = self._draft_worker.runtime_config
        attach_depthspec_runtime_config(self, self.runtime_config)
        self.loop_segment_executor_enabled = self.runtime_config.loop_segment_executor
        self.depthspec_segment_runtime = DepthSpecSegmentRuntime(
            topk=self.topk,
            speculative_num_steps=self.speculative_num_steps,
            num_draft_tokens=self.speculative_num_draft_tokens,
            max_rows_per_batch=int(
                self.runtime_config.diagnostics.get(
                    "segment_runtime_max_rows_per_batch",
                    256,
                )
            ),
        )
        self.depthspec_segment_bindings = DepthSpecSegmentBindingStore()
        self.depthspec_bound_scheduler = DepthSpecBoundSegmentScheduler(
            runtime=self.depthspec_segment_runtime,
            bindings=self.depthspec_segment_bindings,
        )
        self._depthspec_segment_runtime_epoch = 0
        self._depthspec_pending_normal_decode = None

        self._validate_vocab_compatibility(
            target_vocab_size=target_worker.model_runner.model_config.vocab_size,
            target_tokenizer=target_worker.tokenizer,
        )

        # Some dummy tensors
        self.num_new_pages_per_topk = torch.empty(
            (), dtype=torch.int64, device=self.device
        )
        self.extend_lens = torch.empty((), dtype=torch.int64, device=self.device)

        self.plan_stream, self.plan_stream_ctx = get_plan_stream(self.device)

        # TODO: Adaptive speculative
        self.adaptive_controller: AdaptiveController | None = None

    def _depthspec_request_modes(self, batch) -> list[str]:
        modes = []
        for request in batch.reqs:
            custom_params = getattr(
                request.sampling_params,
                "custom_params",
                None,
            )
            if not isinstance(custom_params, dict):
                modes.append("spec")
                continue
            modes.append(str(custom_params.get("depthspec_mode", "spec")))
        return modes

    def _split_schedule_batch(self, batch, indices: list[int]):
        if not indices:
            return None
        split_batch = copy_module.copy(batch)
        device_indices = torch.tensor(
            indices,
            dtype=torch.int64,
            device=batch.device,
        )
        split_batch.reqs = [batch.reqs[index] for index in indices]
        split_batch.req_pool_indices = batch.req_pool_indices[device_indices]
        split_batch.req_pool_indices_cpu = batch.req_pool_indices_cpu[
            device_indices.cpu()
        ]
        split_batch.seq_lens = batch.seq_lens[device_indices]
        split_batch.orig_seq_lens = batch.orig_seq_lens[device_indices]
        if batch.input_ids is not None:
            split_batch.input_ids = batch.input_ids[device_indices]
        if batch.seq_lens_cpu is not None:
            split_batch.seq_lens_cpu = batch.seq_lens_cpu[device_indices.cpu()]
        split_batch.out_cache_loc = (
            batch.out_cache_loc[device_indices]
            if batch.out_cache_loc is not None
            else None
        )
        if batch.out_cache_loc_dsv4 is not None:
            split_batch.out_cache_loc_dsv4 = batch.out_cache_loc_dsv4[
                device_indices
            ]
        split_batch.seq_lens_sum = None

        if batch.spec_info is not None:
            split_spec_info = copy_module.copy(batch.spec_info)
            split_spec_info.filter_batch(
                new_indices=device_indices,
                new_indices_cpu=indices,
            )
            split_batch.spec_info = split_spec_info

        from sglang.srt.sampling.sampling_batch_info import (
            SamplingBatchInfo,
        )

        split_batch.sampling_info = SamplingBatchInfo.from_schedule_batch(
            split_batch,
            vocab_size=self.target_worker.model_runner.model_config.vocab_size,
        )
        return split_batch

    def _trace_request_level_modes(
        self,
        batch,
        modes: list[str],
    ) -> None:
        trace_path = self.runtime_config.diagnostics.get(
            "request_mode_trace_path"
        )
        if not trace_path:
            return
        row = {
            "captured_unix_s": time.time(),
            "request_count": len(modes),
            "spec_count": sum(mode == "spec" for mode in modes),
            "decode_count": sum(mode == "decode" for mode in modes),
            "modes": modes,
            "spec_indices": [
                index for index, mode in enumerate(modes) if mode == "spec"
            ],
            "decode_indices": [
                index for index, mode in enumerate(modes) if mode == "decode"
            ],
            "req_pool_indices": batch.req_pool_indices.detach().cpu().tolist(),
        }
        output = Path(trace_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row) + "\n")

    def _prepare_pending_normal_decode(self, decode_batch):
        """Prepare one real normal-decode segment from a batch view."""
        from sglang.srt.mem_cache.allocation import alloc_for_decode
        from sglang.srt.model_executor.forward_batch_info import (
            CaptureHiddenMode,
            ForwardBatch,
        )

        decode_batch.out_cache_loc = alloc_for_decode(
            decode_batch,
            token_per_req=1,
        )
        for request in decode_batch.reqs:
            request.decode_batch_idx += 1
        decode_batch.seq_lens = decode_batch.seq_lens + 1
        decode_batch.orig_seq_lens = decode_batch.orig_seq_lens + 1
        if decode_batch.seq_lens_cpu is not None:
            decode_batch.seq_lens_cpu = decode_batch.seq_lens_cpu + 1

        target_runner = self.target_worker.model_runner
        last_token_ids = [
            (
                request.output_ids[-1]
                if len(request.output_ids)
                else request.origin_input_ids[-1]
            )
            for request in decode_batch.reqs
        ]
        decode_batch.input_ids = torch.tensor(
            last_token_ids,
            dtype=torch.int64,
            device=decode_batch.device,
        )
        decode_forward_batch = ForwardBatch.init_new(
            decode_batch,
            target_runner,
            capture_hidden_mode=CaptureHiddenMode.NULL,
            return_hidden_states_before_norm=False,
        )
        target_runner.attn_backend.init_forward_metadata(decode_forward_batch)
        decode_forward_batch.mark_forward_metadata_ready()
        target_model = target_runner.model
        decode_loops = tuple(
            range(int(target_model.gpt.target_loop_count))
        )
        decode_schedule = tuple(
            float(value)
            for value in target_model.gpt.step_schedule[
                : target_model.gpt.target_loop_count
            ]
        )
        decode_segment = LoopSegment(
            input_ids=decode_forward_batch.input_ids,
            positions=decode_forward_batch.positions,
            out_cache_loc=decode_forward_batch.out_cache_loc,
            seq_lens=decode_forward_batch.seq_lens,
            seq_lens_cpu=decode_forward_batch.seq_lens_cpu,
            seq_lens_sum=decode_forward_batch.seq_lens_sum,
            spec_info=None,
            req_pool_indices=decode_forward_batch.req_pool_indices,
            phase="normal_decode",
            loop_indices=decode_loops,
            step_schedule=decode_schedule,
            attention_contract="causal-decode",
        )
        return SimpleNamespace(
            segment=decode_segment,
            forward_batch=decode_forward_batch,
            next_token_ids=None,
        )

    def _run_pending_normal_decode(self, pending_decode):
        """Execute a prepared normal-decode segment through the target model."""
        if pending_decode.next_token_ids is not None:
            return pending_decode.next_token_ids
        from sglang.srt.model_executor.forward_context import (
            ForwardContext,
            forward_context,
        )

        target_runner = self.target_worker.model_runner
        with forward_context(
            ForwardContext(attn_backend=target_runner.attn_backend)
        ):
            decode_logits_output = target_runner.model.forward(
                pending_decode.forward_batch.input_ids,
                pending_decode.forward_batch.positions,
                pending_decode.forward_batch,
            )
        pending_decode.next_token_ids = target_runner.sample(
            decode_logits_output,
            pending_decode.forward_batch,
        )
        return pending_decode.next_token_ids

    def _forward_normal_decode_only(self, batch):
        """Run a request-level decode-only batch without speculative draft state."""
        from sglang.srt.managers.utils import GenerationBatchResult
        from sglang.srt.model_executor.forward_batch_info import ForwardMode

        decode_batch = self._split_schedule_batch(
            batch,
            list(range(len(batch.reqs))),
        )
        decode_batch.forward_mode = ForwardMode.DECODE
        decode_batch.spec_info = None
        pending_decode = self._prepare_pending_normal_decode(decode_batch)
        next_token_ids = self._run_pending_normal_decode(pending_decode)
        request_count = len(batch.reqs)
        stride = self.speculative_num_draft_tokens
        flattened_tokens = torch.zeros(
            request_count * stride,
            dtype=next_token_ids.dtype,
            device=self.device,
        )
        for request_index, token in enumerate(next_token_ids.tolist()):
            start = request_index * stride
            flattened_tokens[start : start + stride] = token
        accept_lens = torch.ones(
            request_count,
            dtype=torch.int32,
            device=self.device,
        )
        from sglang.srt.speculative.eagle_info import EagleDraftInput

        next_draft_input = EagleDraftInput(
            bonus_tokens=next_token_ids.to(dtype=torch.int32),
            topk_p=torch.zeros(
                (request_count, self.topk),
                dtype=torch.float32,
                device=self.device,
            ),
            topk_index=torch.zeros(
                (request_count, self.topk),
                dtype=torch.int64,
                device=self.device,
            ),
        )
        return GenerationBatchResult(
            logits_output=None,
            next_token_ids=flattened_tokens,
            speculative_num_draft_tokens=stride,
            next_draft_input=next_draft_input,
            accept_lens=accept_lens,
            new_seq_lens=batch.seq_lens + accept_lens,
        )

    def _merge_request_level_results(
        self,
        batch,
        modes: list[str],
        spec_indices: list[int],
        decode_indices: list[int],
        spec_result,
        decode_result,
    ):
        from sglang.srt.managers.utils import GenerationBatchResult

        stride = self.speculative_num_draft_tokens
        request_count = len(modes)
        original_position = {
            index: position for position, index in enumerate(spec_indices)
        }
        decode_position = {
            index: position for position, index in enumerate(decode_indices)
        }
        next_token_ids = torch.zeros(
            request_count * stride,
            dtype=spec_result.next_token_ids.dtype,
            device=self.device,
        )
        accept_lens = torch.zeros(
            request_count,
            dtype=spec_result.accept_lens.dtype,
            device=self.device,
        )
        for request_index in spec_indices:
            source = original_position[request_index]
            target_start = request_index * stride
            source_start = source * stride
            next_token_ids[target_start : target_start + stride] = (
                spec_result.next_token_ids[source_start : source_start + stride]
            )
            accept_lens[request_index] = spec_result.accept_lens[source]
        for request_index in decode_indices:
            source = decode_position[request_index]
            token = decode_result.next_token_ids[source].to(
                dtype=next_token_ids.dtype
            )
            target_start = request_index * stride
            next_token_ids[target_start : target_start + stride] = token
            accept_lens[request_index] = 1

        source_draft_input = spec_result.next_draft_input
        merged_draft_input = copy_module.copy(source_draft_input)
        merged_draft_input.bonus_tokens = torch.zeros(
            request_count,
            dtype=source_draft_input.bonus_tokens.dtype,
            device=self.device,
        )
        merged_draft_input.topk_p = torch.zeros(
            (request_count, source_draft_input.topk_p.shape[1]),
            dtype=source_draft_input.topk_p.dtype,
            device=self.device,
        )
        merged_draft_input.topk_index = torch.zeros(
            (request_count, source_draft_input.topk_index.shape[1]),
            dtype=source_draft_input.topk_index.dtype,
            device=self.device,
        )
        for request_index in spec_indices:
            source = original_position[request_index]
            merged_draft_input.bonus_tokens[request_index] = (
                source_draft_input.bonus_tokens[source]
            )
            merged_draft_input.topk_p[request_index] = (
                source_draft_input.topk_p[source]
            )
            merged_draft_input.topk_index[request_index] = (
                source_draft_input.topk_index[source]
            )
        for request_index in decode_indices:
            source = decode_position[request_index]
            merged_draft_input.bonus_tokens[request_index] = (
                decode_result.next_token_ids[source]
            )

        source_accept_index = getattr(
            source_draft_input,
            "accept_index",
            None,
        )
        if source_accept_index is not None:
            merged_accept_index = torch.full(
                (
                    request_count,
                    source_accept_index.shape[1],
                ),
                -1,
                dtype=source_accept_index.dtype,
                device=self.device,
            )
            for request_index in spec_indices:
                source = original_position[request_index]
                merged_accept_index[request_index] = (
                    source_accept_index[source]
                )
            merged_draft_input.accept_index = merged_accept_index

        source_candidates = getattr(
            source_draft_input,
            "candidate_tokens",
            None,
        )
        source_positions = getattr(
            source_draft_input,
            "candidate_positions",
            None,
        )
        if source_candidates is not None and source_positions is not None:
            merged_candidates = source_candidates[0].repeat(
                request_count,
                1,
            )
            merged_positions = source_positions[0].repeat(
                request_count,
                1,
            )
            for request_index in spec_indices:
                source = original_position[request_index]
                merged_candidates[request_index] = source_candidates[source]
                merged_positions[request_index] = source_positions[source]
            for request_index in decode_indices:
                source = decode_position[request_index]
                merged_candidates[request_index, 0] = (
                    decode_result.next_token_ids[source]
                )
                merged_candidates[request_index, 1:] = (
                    merged_candidates[request_index, 0]
                )
                merged_positions[request_index, 1:] = (
                    merged_positions[request_index, 0]
                )
            merged_draft_input.candidate_tokens = merged_candidates
            merged_draft_input.candidate_positions = merged_positions

        return GenerationBatchResult(
            logits_output=spec_result.logits_output,
            next_token_ids=next_token_ids,
            can_run_cuda_graph=spec_result.can_run_cuda_graph,
            speculative_num_draft_tokens=stride,
            next_draft_input=merged_draft_input,
            accept_lens=accept_lens,
            new_seq_lens=batch.seq_lens + accept_lens,
        )

    def forward_batch_generation(
        self,
        batch,
        on_publish=None,
        grammar_barrier=None,
    ):
        modes = self._depthspec_request_modes(batch)
        request_level_mode = bool(
            self.runtime_config.diagnostics.get(
                "request_level_depthspec_mode", False
            )
        )
        mixed_mode = (
            request_level_mode
            and not batch.forward_mode.is_extend()
            and not batch.is_extend_in_batch
            and "spec" in modes
            and "decode" in modes
        )
        decode_only_mode = (
            request_level_mode
            and not batch.forward_mode.is_extend()
            and not batch.is_extend_in_batch
            and bool(modes)
            and all(mode == "decode" for mode in modes)
        )
        if decode_only_mode:
            self._trace_request_level_modes(batch, modes)
            return self._forward_normal_decode_only(batch)
        if not mixed_mode:
            return super().forward_batch_generation(
                batch,
                on_publish=on_publish,
                grammar_barrier=grammar_barrier,
            )

        if any(mode not in {"spec", "decode"} for mode in modes):
            raise ValueError(
                f"Unsupported DepthSpec request modes: {sorted(set(modes))}"
            )
        if batch.return_logprob or batch.return_hidden_states:
            raise ValueError(
                "Request-level DepthSpec decode mode does not yet support "
 "logprobs or hidden-state returns"
            )
        if batch.has_grammar:
            raise ValueError(
                "Request-level DepthSpec decode mode does not yet support grammar"
            )

        self._trace_request_level_modes(batch, modes)
        spec_indices = [
            index for index, mode in enumerate(modes) if mode == "spec"
        ]
        decode_indices = [
            index for index, mode in enumerate(modes) if mode == "decode"
        ]
        spec_batch = self._split_schedule_batch(batch, spec_indices)
        decode_batch = self._split_schedule_batch(batch, decode_indices)
        if spec_batch is None or decode_batch is None:
            raise RuntimeError("Mixed DepthSpec batch split produced no requests")

        from sglang.srt.model_executor.forward_batch_info import ForwardMode

        decode_batch.forward_mode = ForwardMode.DECODE
        decode_batch.spec_info = None
        self._depthspec_pending_normal_decode = (
            self._prepare_pending_normal_decode(decode_batch)
        )
        spec_result = super().forward_batch_generation(
            spec_batch,
            on_publish=on_publish,
            grammar_barrier=grammar_barrier,
        )
        pending_decode = self._depthspec_pending_normal_decode
        self._depthspec_pending_normal_decode = None
        if pending_decode is None:
            raise RuntimeError("Mixed DepthSpec decode segment disappeared")
        self._run_pending_normal_decode(pending_decode)
        decode_result = SimpleNamespace(
            next_token_ids=pending_decode.next_token_ids,
        )
        return self._merge_request_level_results(
            batch,
            modes,
            spec_indices,
            decode_indices,
            spec_result,
            decode_result,
        )

    def _validate_vocab_compatibility(
        self,
        target_vocab_size: int,
        target_tokenizer,
    ) -> None:
        """Raise ValueError if the draft and target vocabularies are incompatible."""
        draft_vocab_size = self._draft_worker.draft_runner.model_config.vocab_size
        draft_tokenizer = self._draft_worker.draft_worker.tokenizer
        if target_vocab_size != draft_vocab_size:
            raise ValueError(
                f"STANDALONE speculative decoding requires the draft model to share the "
                f"same vocabulary as the target model, but got "
                f"target vocab_size={target_vocab_size} and "
                f"draft vocab_size={draft_vocab_size}. "
                f"Use a draft model with a matching vocabulary, or a speculative "
                f"algorithm that supports heterogeneous vocabularies."
            )
        if (
            target_tokenizer is not None
            and draft_tokenizer is not None
            and hasattr(target_tokenizer, "get_vocab")
            and hasattr(draft_tokenizer, "get_vocab")
            and target_tokenizer.get_vocab() != draft_tokenizer.get_vocab()
        ):
            raise ValueError(
                "STANDALONE speculative decoding requires the draft model to share the "
                "same vocabulary as the target model, but the two tokenizers have "
                "different token-to-id mappings even though their vocab sizes match. "
                "Use a draft model with a matching vocabulary, or a speculative "
                "algorithm that supports heterogeneous vocabularies."
            )

    def _audit_tree_verify_input(self, batch, verify_input) -> None:
        """Optionally dump the exact tree metadata consumed by verification."""
        audit_path = self.runtime_config.diagnostics.get("tree_audit_path")
        if not audit_path:
            return
        max_rounds = int(
            self.runtime_config.diagnostics.get("tree_audit_max_rounds", 64)
        )
        marker_path = f"{audit_path}.count"
        try:
            round_index = int(Path(marker_path).read_text(encoding="utf-8"))
        except (FileNotFoundError, ValueError):
            round_index = 0
        if round_index >= max_rounds:
            return

        seq_len = int(batch.seq_lens[0].item())
        draft_token_num = int(verify_input.draft_token_num)
        full_mask = verify_input.custom_mask.detach().cpu()
        total_kv_width = seq_len + draft_token_num
        tree_masks = []
        for row in range(draft_token_num):
            row_start = row * total_kv_width + seq_len
            tree_masks.append(
                full_mask[row_start : row_start + draft_token_num].tolist()
            )

        row = {
            "round_index": round_index,
            "batch_size": int(batch.seq_lens.numel()),
            "seq_len_before_verify": seq_len,
            "spec_steps": int(verify_input.spec_steps),
            "topk": int(verify_input.topk),
            "draft_token_num": draft_token_num,
            "draft_tokens": verify_input.draft_token.detach().cpu().tolist(),
            "positions": verify_input.positions.detach().cpu().tolist(),
            "tree_mask_qlen_only": tree_masks,
            "retrieve_index": verify_input.retrieve_index.detach().cpu().tolist(),
            "retrieve_next_token": verify_input.retrieve_next_token.detach()
            .cpu()
            .tolist(),
            "retrieve_next_sibling": verify_input.retrieve_next_sibling.detach()
            .cpu()
            .tolist(),
        }
        Path(audit_path).parent.mkdir(parents=True, exist_ok=True)
        with Path(audit_path).open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row) + "\n")
        Path(marker_path).write_text(str(round_index + 1), encoding="utf-8")

    def _audit_draft_prefix_artifacts(self, batch, verify_input, result) -> None:
        """Compare draft prefix artifacts with target verify prefix states."""
        audit_path = self.runtime_config.diagnostics.get(
            "prefix_artifact_audit_path"
        )
        if not audit_path:
            return
        marker_path = f"{audit_path}.count"
        try:
            audit_index = int(Path(marker_path).read_text(encoding="utf-8"))
        except (FileNotFoundError, ValueError):
            audit_index = 0
        max_audits = int(
            self.runtime_config.diagnostics.get("prefix_artifact_audit_max", 128)
        )
        if audit_index >= max_audits:
            return

        model = self.target_worker.model_runner.model
        target_state = getattr(model, "last_prefix_state", None)
        artifacts = getattr(
            model.gpt,
            "depthspec_proposal_prefix_artifacts",
            [],
        )
        if target_state is None:
            row = {
                "audit_index": audit_index,
                "status": "missing_target_prefix_state",
                "artifact_count": len(artifacts),
            }
        else:
            target_hidden = target_state.hidden_states
            candidate_tokens = verify_input.draft_token
            candidate_positions = verify_input.positions
            rows = []
            exact_errors = []
            token_only_errors = []
            aligned_token_errors = []
            for row_index, (token_id, position) in enumerate(
                zip(
                    candidate_tokens.tolist(),
                    candidate_positions.tolist(),
                    strict=True,
                )
            ):
                target_row_hidden = target_hidden[row_index].float()
                exact_matches = [
                    artifact
                    for artifact in artifacts
                    if artifact["token_id"] == int(token_id)
                    and artifact["position"] == int(position)
                ]
                token_matches = [
                    artifact
                    for artifact in artifacts
                    if artifact["token_id"] == int(token_id)
                ]
                exact_candidates = []
                for artifact in exact_matches:
                    difference = (
                        artifact["hidden_states"].float() - target_row_hidden
                    ).abs()
                    error = float(difference.max())
                    exact_errors.append(error)
                    exact_candidates.append(
                        {
                            "artifact_position": artifact["position"],
                            "max_abs_hidden_error": error,
                            "mean_abs_hidden_error": float(difference.mean()),
                        }
                    )
                token_only_candidates = []
                for artifact in token_matches:
                    difference = (
                        artifact["hidden_states"].float() - target_row_hidden
                    ).abs()
                    error = float(difference.max())
                    token_only_errors.append(error)
                    token_only_candidates.append(
                        {
                            "artifact_position": artifact["position"],
                            "position_delta": artifact["position"] - int(position),
                            "max_abs_hidden_error": error,
                        }
                    )
                    if "aligned_hidden_states" in artifact:
                        aligned_difference = (
                            artifact["aligned_hidden_states"].float()
                            - target_row_hidden
                        ).abs()
                        aligned_error = float(aligned_difference.max())
                        aligned_token_errors.append(aligned_error)
                        token_only_candidates[-1]["aligned_position"] = artifact[
                            "aligned_position"
                        ]
                        token_only_candidates[-1]["aligned_position_match"] = artifact[
                            "aligned_position"
                        ] == int(position)
                        token_only_candidates[-1]["aligned_max_abs_hidden_error"] = (
                            aligned_error
                        )
                rows.append(
                    {
                        "verify_row": row_index,
                        "token_id": int(token_id),
                        "verify_position": int(position),
                        "exact_match_count": len(exact_matches),
                        "token_match_count": len(token_matches),
                        "exact_candidates": exact_candidates,
                        "token_only_candidates": token_only_candidates,
                    }
                )

            row = {
                "audit_index": audit_index,
                "status": "complete",
                "verify_row_count": len(rows),
                "artifact_count": len(artifacts),
                "exact_match_row_count": sum(
                    item["exact_match_count"] > 0 for item in rows
                ),
                "token_match_row_count": sum(
                    item["token_match_count"] > 0 for item in rows
                ),
                "max_exact_abs_hidden_error": (
                    max(exact_errors) if exact_errors else None
                ),
                "mean_exact_abs_hidden_error": (
                    sum(exact_errors) / len(exact_errors) if exact_errors else None
                ),
                "max_token_only_abs_hidden_error": (
                    max(token_only_errors) if token_only_errors else None
                ),
                "max_aligned_token_only_abs_hidden_error": (
                    max(aligned_token_errors) if aligned_token_errors else None
                ),
                "mean_aligned_token_only_abs_hidden_error": (
                    sum(aligned_token_errors) / len(aligned_token_errors)
                    if aligned_token_errors
                    else None
                ),
                "rows": rows,
                "artifacts": [
                    {
                        "token_id": artifact["token_id"],
                        "position": artifact["position"],
                    }
                    for artifact in artifacts
                ],
            }

        Path(audit_path).parent.mkdir(parents=True, exist_ok=True)
        with Path(audit_path).open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row) + "\n")
        Path(marker_path).write_text(str(audit_index + 1), encoding="utf-8")

        root_dump_dir = self.runtime_config.diagnostics.get(
            "prefix_root_audit_dump_dir"
        )
        if root_dump_dir and target_state is not None:
            next_draft_input = result.next_draft_input
            accept_index = getattr(next_draft_input, "accept_index", None)
            if accept_index is None:
                raise ValueError("Root-aware audit requires accept_index")
            candidate_tokens = verify_input.draft_token.reshape(
                verify_input.draft_token.shape[0] // verify_input.draft_token_num,
                verify_input.draft_token_num,
            )
            candidate_positions = verify_input.positions.reshape(candidate_tokens.shape)
            local_accept_index = _localize_tree_accept_index(
                accept_index,
                candidate_tokens.shape[1],
            )
            manifest_path = Path(root_dump_dir) / "manifest.jsonl"
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            for request_index, request in enumerate(batch.reqs):
                accept_length = int(result.accept_lens[request_index].item())
                if accept_length < 2:
                    continue
                root_index = int(local_accept_index[request_index, 0].item())
                child_index = int(local_accept_index[request_index, 1].item())
                root_position = int(
                    candidate_positions[request_index, root_index].item()
                )
                full_ids = list(request.origin_input_ids) + list(request.output_ids)
                context_ids = [int(value) for value in full_ids[:root_position]]
                root_token = int(candidate_tokens[request_index, root_index].item())
                candidate_token = int(
                    candidate_tokens[request_index, child_index].item()
                )
                case_path = (
                    Path(root_dump_dir)
                    / f"case_{audit_index:06d}_{request_index:03d}.pt"
                )
                torch.save(
                    {
                        "audit_index": audit_index,
                        "request_index": request_index,
                        "context_ids": context_ids,
                        "root_token": root_token,
                        "candidate_token": candidate_token,
                        "root_position": int(
                            candidate_positions[request_index, root_index].item()
                        ),
                        "candidate_position": int(
                            candidate_positions[request_index, child_index].item()
                        ),
                        "target_hidden": target_state.hidden_states[child_index].cpu(),
                        "batch_seq_len": int(batch.seq_lens[request_index].item()),
                        "fill_ids_length": len(request.get_fill_ids()),
                        "origin_plus_output_length": len(full_ids),
                    },
                    case_path,
                )
                with manifest_path.open("a", encoding="utf-8") as stream:
                    stream.write(
                        json.dumps(
                            {
                                "audit_index": audit_index,
                                "request_index": request_index,
                                "case_path": str(case_path),
                                "context_length": len(context_ids),
                                "root_position": root_position,
                                "batch_seq_len": int(
                                    batch.seq_lens[request_index].item()
                                ),
                                "fill_ids_length": len(request.get_fill_ids()),
                                "origin_plus_output_length": len(full_ids),
                                "root_token": root_token,
                                "candidate_token": candidate_token,
                            }
                        )
                        + "\n"
                    )

        model.gpt.depthspec_capture_prefix_artifact = False
        model.gpt.depthspec_proposal_prefix_artifacts = []
        model.depthspec_proposal_prefix_artifacts = []

    def verify(self, batch, grammar_barrier=None):
        self._audit_tree_verify_input(batch, batch.spec_info)
        started = time.perf_counter()
        profile_path = self.runtime_config.diagnostics.get("verify_profile_path")
        profile_skip = int(
            self.runtime_config.diagnostics.get("verify_profile_skip", 20)
        )
        profile_active = int(
            self.runtime_config.diagnostics.get("verify_profile_active", 10)
        )
        profile_include_prepare = bool(
            self.runtime_config.diagnostics.get(
                "verify_profile_include_prepare", False
            )
        )
        verify_profiler = None
        should_stop_profiler = False
        profile_index = None
        if profile_path:
            if not hasattr(self, "_depthspec_verify_profile_count"):
                self._depthspec_verify_profile_count = 0
            profile_index = self._depthspec_verify_profile_count
            self._depthspec_verify_profile_count += 1
            if profile_skip <= profile_index < profile_skip + profile_active:
                verify_profiler = torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ],
                    with_stack=False,
                    record_shapes=False,
                    profile_memory=False,
                )
                if profile_include_prepare:
                    verify_profiler.start()
                    should_stop_profiler = (
                        profile_index == profile_skip + profile_active - 1
                    )
        if self.runtime_config.canonical_stem:
            canonical_started = time.perf_counter()
            with self._depthspec_profile_scope("depthspec.prepare_canonical"):
                self._prepare_canonical_stem_oracle(batch)
            _append_stage_timing(
                "canonical_stem_precompute",
                int(batch.seq_lens.numel()),
                time.perf_counter() - canonical_started,
            )
        if profile_path:
            if (
                verify_profiler is not None
                and not profile_include_prepare
            ):
                verify_profiler.start()
                should_stop_profiler = (
                    profile_index == profile_skip + profile_active - 1
                )
        target_runner = self.target_worker.model_runner
        decode_graph_runner = target_runner.decode_cuda_graph_runner
        if self.runtime_config.target_verify_graph_mode == "disabled":
            target_runner.decode_cuda_graph_runner = None
        try:
            result = super().verify(batch, grammar_barrier=grammar_barrier)
        finally:
            target_runner.decode_cuda_graph_runner = decode_graph_runner
            if verify_profiler is not None:
                verify_profiler.step()
                if should_stop_profiler:
                    verify_profiler.stop()
                    profile_output = Path(profile_path)
                    profile_output.parent.mkdir(parents=True, exist_ok=True)
                    verify_profiler.export_chrome_trace(str(profile_output))
        _append_stage_timing(
            "target_verify", int(batch.seq_lens.numel()), time.perf_counter() - started
        )
        self._audit_draft_prefix_artifacts(batch, batch.spec_info, result)
        self._draft_worker._prepare_draft_artifact_relay(batch, result)
        return result

    def _prepare_canonical_stem_oracle(self, batch):
        """Precompute a target-compatible two-loop stem after proposal building."""
        from sglang.srt.model_executor.forward_context import (
            ForwardContext,
            forward_context,
        )
        from sglang.srt.speculative.eagle_utils import eagle_prepare_for_verify

        target_runner = self.target_worker.model_runner
        target_model = target_runner.model
        if hasattr(target_model, "depthspec_prefetched_verify_hidden_states"):
            del target_model.depthspec_prefetched_verify_hidden_states

        if self.runtime_config.canonical_partial_stem:
            self._prepare_partial_only_canonical_stem(
                batch,
                target_model,
                target_runner,
            )
            return

        verify_forward_batch, can_run_graph = eagle_prepare_for_verify(
            batch.spec_info,
            self.req_to_token_pool,
            batch,
            self.target_worker,
        )
        if can_run_graph:
            raise ValueError("Canonical stem oracle unexpectedly selected a graph")
        if self.runtime_config.diagnostics.get("merged_full_stem", False):
            merged_segments = self._split_verify_forward_into_segments(
                verify_forward_batch,
                "authoritative_full_stem",
                sequence_lens=batch.seq_lens,
                sequence_lens_cpu=batch.seq_lens_cpu,
            )
            self._last_merged_segment_mask_numels = [
                int(segment.spec_info.custom_mask.numel())
                for segment in merged_segments
            ]
            self._last_merged_segment_seq_lens = [
                int(item.seq_lens[0].item()) for item in merged_segments
            ]
            target_runner.attn_backend.init_forward_metadata(verify_forward_batch)
            verify_forward_batch.mark_forward_metadata_ready()
            with forward_context(
                ForwardContext(attn_backend=target_runner.attn_backend)
            ):
                _, reference_state = target_model.gpt.forward_branch(
                    verify_forward_batch.input_ids,
                    verify_forward_batch.positions,
                    verify_forward_batch,
                    loop_indices=(0, 1),
                    step_schedule=(0.125, 0.125),
                    return_state=True,
                )
            reference_metadata = target_runner.attn_backend.forward_metadata
            reference_kv_indptr = (
                reference_metadata.kv_indptr.clone()
                if reference_metadata.kv_indptr is not None
                else None
            )
            reference_kv_indices = (
                reference_metadata.kv_indices.clone()
                if reference_metadata.kv_indices is not None
                else None
            )
            reference_qo_indptr = (
                reference_metadata.qo_indptr.clone()
                if reference_metadata.qo_indptr is not None
                else None
            )
            reference_mask_indptr = (
                reference_metadata.mask_indptr.clone()
                if reference_metadata.mask_indptr is not None
                else None
            )
            reference_custom_mask = verify_forward_batch.spec_info.custom_mask.clone()
            stem_state, merged_forward_batch = (
                LoopSegmentExecutor(self, batch).execute_many(merged_segments)
            )
            audit_path = self.runtime_config.diagnostics.get("merged_stem_audit_path")
            if audit_path:
                seq_lens = verify_forward_batch.seq_lens.detach().cpu().tolist()
                positions = verify_forward_batch.positions.detach().cpu().tolist()
                width = int(
                    getattr(
                        verify_forward_batch.spec_info,
                        "num_tokens_per_req",
                        verify_forward_batch.spec_info.draft_token_num,
                    )
                )
                per_request_errors = [
                    self._canonical_tensor_error(
                        stem_state.hidden_states[
                            request_index * width : (request_index + 1) * width
                        ],
                        reference_state.hidden_states[
                            request_index * width : (request_index + 1) * width
                        ],
                    )
                    for request_index in range(len(seq_lens))
                ]
                hidden_error = self._canonical_tensor_error(
                    stem_state.hidden_states,
                    reference_state.hidden_states,
                )
                row = {
                    "request_count": int(batch.seq_lens.numel()),
                    "row_count": int(verify_forward_batch.input_ids.shape[0]),
                    "seq_lens": seq_lens,
                    "positions": positions,
                    "custom_mask_numel": int(
                        verify_forward_batch.spec_info.custom_mask.numel()
                    ),
                    "reference_custom_mask_numel": int(reference_custom_mask.numel()),
                    "merged_custom_mask_numel": int(
                        merged_forward_batch.spec_info.custom_mask.numel()
                    ),
                    "segment_custom_mask_numels": getattr(
                        self, "_last_merged_segment_mask_numels", []
                    ),
                    "segment_seq_lens": getattr(
                        self, "_last_merged_segment_seq_lens", []
                    ),
                    "expected_compact_mask_numel": sum(
                        (int(length) + width) * width for length in seq_lens
                    ),
                    "per_request_hidden_errors": per_request_errors,
                    "hidden_error": hidden_error,
                    "metadata_audit": {
                        "kv_indptr_error": (
                            self._canonical_tensor_error(
                                target_runner.attn_backend.forward_metadata.kv_indptr,
                                reference_kv_indptr,
                            )
                            if reference_kv_indptr is not None
                            and target_runner.attn_backend.forward_metadata.kv_indptr
                            is not None
                            else None
                        ),
                        "kv_indices_error": (
                            self._canonical_tensor_error(
                                target_runner.attn_backend.forward_metadata.kv_indices,
                                reference_kv_indices,
                            )
                            if reference_kv_indices is not None
                            and target_runner.attn_backend.forward_metadata.kv_indices
                            is not None
                            else None
                        ),
                        "qo_indptr_error": (
                            self._canonical_tensor_error(
                                target_runner.attn_backend.forward_metadata.qo_indptr,
                                reference_qo_indptr,
                            )
                            if reference_qo_indptr is not None
                            and target_runner.attn_backend.forward_metadata.qo_indptr
                            is not None
                            else None
                        ),
                        "mask_indptr_error": (
                            self._canonical_tensor_error(
                                target_runner.attn_backend.forward_metadata.mask_indptr,
                                reference_mask_indptr,
                            )
                            if reference_mask_indptr is not None
                            and target_runner.attn_backend.forward_metadata.mask_indptr
                            is not None
                            else None
                        ),
                        "custom_mask_numel_equal": (
                            merged_forward_batch.spec_info.custom_mask.numel()
                            == reference_custom_mask.numel()
                        ),
                        "custom_mask_all_equal": torch.equal(
                            merged_forward_batch.spec_info.custom_mask,
                            reference_custom_mask,
                        ),
                    },
                }
                output = Path(audit_path)
                output.parent.mkdir(parents=True, exist_ok=True)
                with output.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(row) + "\n")
                stem_state = reference_state
            target_model.depthspec_canonical_stem_state = stem_state

        target_runner.attn_backend.init_forward_metadata(verify_forward_batch)
        verify_forward_batch.mark_forward_metadata_ready()
        with forward_context(ForwardContext(attn_backend=target_runner.attn_backend)):
            full_stem_started = time.perf_counter()
            piecewise_runner = None
            if self.runtime_config.canonical_full_stem_piecewise_graph:
                draft_worker = getattr(self, "_draft_worker", self)
                piecewise_runner = getattr(
                    draft_worker,
                    "canonical_stem_piecewise_graph_runners",
                    {},
                ).get(int(verify_forward_batch.input_ids.shape[0]))
            if piecewise_runner is None:
                _, stem_state = target_model.gpt.forward_branch(
                    verify_forward_batch.input_ids,
                    verify_forward_batch.positions,
                    verify_forward_batch,
                    loop_indices=(0, 1),
                    step_schedule=(0.125, 0.125),
                    return_state=True,
                )
            else:
                _, stem_state = piecewise_runner.execute(
                    verify_forward_batch.input_ids,
                    verify_forward_batch.positions,
                    verify_forward_batch,
                    return_state=True,
                )
            _append_segment_trace(
                phase="authoritative_full_stem",
                request_count=int(batch.seq_lens.numel()),
                row_count=int(verify_forward_batch.input_ids.shape[0]),
                loop_indices=(0, 1),
                step_schedule=(0.125, 0.125),
                attention_contract="tree-custom",
                graph_hit=piecewise_runner is not None,
                elapsed_s=time.perf_counter() - full_stem_started,
            )
        target_model.depthspec_canonical_stem_state = stem_state
        if hasattr(target_model, "depthspec_partial_stem_state"):
            if self.runtime_config.diagnostics.get(
                "heterogeneous_ragged_schedule_audit", False
            ) or self.runtime_config.diagnostics.get(
                "live_heterogeneous_ragged_schedule", False
            ):
                self._audit_heterogeneous_ragged_schedule(
                    batch,
                    verify_forward_batch,
                    stem_state,
                )
            if self.runtime_config.diagnostics.get(
                "grouped_ragged_stem_audit", False
            ):
                self._audit_grouped_ragged_stem(
                    batch,
                    verify_forward_batch,
                    stem_state,
                )
            if self.runtime_config.diagnostics.get("ragged_mixed_stem_audit", False):
                self._audit_ragged_mixed_stem(
                    batch,
                    verify_forward_batch,
                    stem_state,
                )
            del target_model.depthspec_partial_stem_state
        if self.runtime_config.diagnostics.get("partial_stem_audit_path"):
            self._audit_partial_canonical_stem(
                target_model,
                target_runner,
                verify_forward_batch,
                stem_state,
            )

    @contextmanager
    def _depthspec_profile_scope(self, name: str):
        """Wrap a stage only while inclusive verify profiling is enabled."""
        if self.runtime_config.diagnostics.get(
            "verify_profile_include_prepare", False
        ):
            with torch.profiler.record_function(name):
                yield
        else:
            yield

    def _audit_heterogeneous_ragged_schedule(
        self,
        batch,
        verify_forward_batch,
        full_stem_state,
    ) -> None:
        """Validate dense merging across different segment loop schedules."""
        from sglang.srt.model_executor.forward_context import (
            ForwardContext,
            forward_context,
        )

        target_runner = self.target_worker.model_runner
        target_model = target_runner.model
        gpt = target_model.gpt
        live_mode = bool(
            self.runtime_config.diagnostics.get(
                "live_heterogeneous_ragged_schedule", False
            )
        )
        compute_reference = not live_mode or bool(
            self.runtime_config.diagnostics.get(
                "live_reference_continuation", False
            )
        )
        continuation_loops = tuple(
            range(2, int(gpt.target_loop_count))
        )
        continuation_schedule = tuple(
            float(value) for value in gpt.step_schedule[2 : gpt.target_loop_count]
        )

        def make_segment(phase, loops, schedule):
            return LoopSegment(
                input_ids=verify_forward_batch.input_ids,
                positions=verify_forward_batch.positions,
                out_cache_loc=verify_forward_batch.out_cache_loc,
                seq_lens=batch.seq_lens,
                seq_lens_cpu=(
                    batch.seq_lens_cpu
                    if batch.seq_lens_cpu is not None
                    else batch.seq_lens.cpu()
                ),
                seq_lens_sum=verify_forward_batch.spec_info.seq_lens_sum
                if verify_forward_batch.spec_info.seq_lens_sum is not None
                else int(batch.seq_lens.sum().item()),
                spec_info=verify_forward_batch.spec_info,
                req_pool_indices=batch.req_pool_indices,
                phase=phase,
                loop_indices=loops,
                step_schedule=schedule,
                attention_contract="tree-custom",
            )

        full_segment = make_segment(
            "authoritative_full_stem",
            (0, 1),
            (0.125, 0.125),
        )
        continuation_segment = make_segment(
            "target_continuation",
            continuation_loops,
            continuation_schedule,
        )
        schedule_full_lane = bool(
            self.runtime_config.diagnostics.get(
                "live_schedule_full_lane", True
            )
        )
        lightweight_live_graph = (
            live_mode
            and not schedule_full_lane
            and not self.runtime_config.diagnostics.get(
                "heterogeneous_ragged_schedule_audit", False
            )
            and not self.runtime_config.diagnostics.get(
                "ragged_draft_extend_audit", False
            )
            and self.runtime_config.diagnostics.get(
                "live_continuation_piecewise_graph", False
            )
            and self.runtime_config.diagnostics.get(
                "live_lightweight_binding", False
            )
        )
        if lightweight_live_graph:
            pending_decode = self._depthspec_pending_normal_decode
            mixed_decode_enabled = bool(
                self.runtime_config.diagnostics.get(
                    "live_mixed_normal_decode", False
                )
            )
            if mixed_decode_enabled and pending_decode is not None:
                segment_runtime = self.depthspec_segment_runtime
                binding_store = self.depthspec_segment_bindings
                bound_scheduler = self.depthspec_bound_scheduler
                runtime_epoch = self._depthspec_segment_runtime_epoch
                self._depthspec_segment_runtime_epoch += 1
                continuation_request_id = runtime_epoch * 100
                normal_request_id = continuation_request_id + 1
                segment_seed = (
                    f"r{continuation_request_id}-k0-"
                    "authoritative_full_stem"
                )
                segment_runtime.mark_seed_completed(segment_seed)
                segment_runtime.register_request(
                    continuation_request_id,
                    phase="target_continuation",
                    round_id=0,
                    ready_tick=0,
                    lane=0,
                    request_count=int(batch.seq_lens.numel()),
                )
                segment_runtime.register_request(
                    normal_request_id,
                    phase="normal_decode",
                    round_id=0,
                    ready_tick=0,
                    lane=1,
                    request_count=int(
                        pending_decode.segment.seq_lens.shape[0]
                    ),
                )
                continuation_descriptor = (
                    segment_runtime.descriptor_for_request(
                        continuation_request_id
                    )
                )
                normal_descriptor = segment_runtime.descriptor_for_request(
                    normal_request_id
                )
                binding_store.bind(
                    continuation_descriptor,
                    RaggedSegmentInput(
                        continuation_segment,
                        verify_forward_batch,
                        initial_hidden_states=full_stem_state.hidden_states,
                        initial_time=full_stem_state.current_time,
                    ),
                )
                binding_store.bind(
                    normal_descriptor,
                    RaggedSegmentInput(
                        pending_decode.segment,
                        pending_decode.forward_batch,
                    ),
                )
                bound_batch = bound_scheduler.materialize_batch(tick=0)
                if bound_batch is None or len(bound_batch.segments) != 2:
                    raise RuntimeError(
                        "Mixed live continuation/decode was not schedulable"
                    )
                mixed_plan = bound_batch.forward_plan
                started = time.perf_counter()
                with self._depthspec_profile_scope("depthspec.mixed_execute"):
                    mixed_result = execute_grouped_ragged_schedule(
                        target_model,
                        target_runner.attn_backend,
                    mixed_plan,
                    shared_metadata=False,
                    save_kv_cache=True,
                    per_group_metadata_snapshots=True,
                    block_graphs_by_row=(
                        getattr(
                            target_model,
                            "depthspec_mixed_piecewise_graphs",
                            None,
                        )
                        if self.runtime_config.diagnostics.get(
                            "mixed_piecewise_graph", False
                        )
                        else None
                    ),
                )
                continuation_rows = int(
                    continuation_segment.input_ids.shape[0]
                )
                decode_rows = int(
                    pending_decode.segment.input_ids.shape[0]
                )
                target_model.depthspec_prefetched_verify_hidden_states = (
                    mixed_result.normalized_hidden_states[
                        :continuation_rows
                    ].contiguous()
                )
                decode_normalized_hidden = (
                    mixed_result.normalized_hidden_states[
                        continuation_rows : continuation_rows + decode_rows
                    ]
                )
                decode_logits_output = target_model.logits_processor(
                    pending_decode.segment.input_ids,
                    decode_normalized_hidden,
                    target_model.lm_head,
                    pending_decode.forward_batch,
                )
                pending_decode.next_token_ids = target_runner.sample(
                    decode_logits_output,
                    pending_decode.forward_batch,
                )
                next_phases = bound_scheduler.complete_batch(bound_batch)
                expected_phases = {
                    continuation_descriptor.segment_id: (
                        "target_compatible_draft_extend"
                    ),
                    normal_descriptor.segment_id: "normal_decode",
                }
                if next_phases != expected_phases:
                    raise RuntimeError(
                        "Mixed live continuation/decode phase mismatch"
                    )
                segment_runtime.retire_request(
                    continuation_request_id,
                    forget_history=True,
                )
                segment_runtime.retire_request(
                    normal_request_id,
                    forget_history=True,
                )
                if (
                    segment_runtime.active_count()
                    or binding_store.ready_count()
                ):
                    raise RuntimeError(
                        "Mixed live continuation/decode leaked state"
                    )
                mixed_trace_path = self.runtime_config.diagnostics.get(
                    "mixed_forward_trace_path"
                )
                if mixed_trace_path:
                    trace_row = {
                        "captured_unix_s": time.time(),
                        "elapsed_s": time.perf_counter() - started,
                        "continuation_request_count": int(
                            batch.seq_lens.numel()
                        ),
                        "normal_decode_request_count": int(
                            pending_decode.segment.seq_lens.shape[0]
                        ),
                        "continuation_req_pool_indices": (
                            batch.req_pool_indices.detach().cpu().tolist()
                        ),
                        "normal_decode_req_pool_indices": (
                            pending_decode.segment.req_pool_indices
                            .detach()
                            .cpu()
                            .tolist()
                        ),
                        "continuation_rows": continuation_rows,
                        "normal_decode_rows": decode_rows,
                        "padding_rows": mixed_plan.padding_rows,
                        "qo_indptr": mixed_plan.qo_indptr.detach().cpu().tolist(),
                    }
                    output = Path(mixed_trace_path)
                    output.parent.mkdir(parents=True, exist_ok=True)
                    with output.open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(trace_row) + "\n")
                return
            continuation_runner = target_model.depthspec_target_continuation_piecewise_runners.get(
                int(continuation_segment.input_ids.shape[0])
            )
            if continuation_runner is not None:
                segment_runtime = self.depthspec_segment_runtime
                binding_store = self.depthspec_segment_bindings
                runtime_epoch = self._depthspec_segment_runtime_epoch
                self._depthspec_segment_runtime_epoch += 1
                continuation_request_id = runtime_epoch
                segment_seed = (
                    f"r{continuation_request_id}-k0-"
                    "authoritative_full_stem"
                )
                segment_runtime.mark_seed_completed(segment_seed)
                segment_runtime.register_request(
                    continuation_request_id,
                    phase="target_continuation",
                    round_id=0,
                    ready_tick=0,
                    lane=0,
                    request_count=int(batch.seq_lens.numel()),
                )
                continuation_descriptor = (
                    segment_runtime.descriptor_for_request(
                        continuation_request_id
                    )
                )
                binding_store.bind(
                    continuation_descriptor,
                    RaggedSegmentInput(
                        continuation_segment,
                        verify_forward_batch,
                        initial_hidden_states=full_stem_state.hidden_states,
                        initial_time=full_stem_state.current_time,
                    ),
                )
                descriptor_batch = segment_runtime.materialize_batch(
                    tick=0,
                    allowed_segment_ids={continuation_descriptor.segment_id},
                )
                if descriptor_batch is None or len(descriptor_batch.segments) != 1:
                    raise RuntimeError(
                        "Lightweight live continuation was not schedulable"
                    )
                with forward_context(
                    ForwardContext(attn_backend=target_runner.attn_backend)
                ):
                    normalized_hidden, continuation_state = (
                        continuation_runner.execute(
                            continuation_segment.input_ids,
                            continuation_segment.positions,
                            verify_forward_batch,
                            initial_hidden_states=full_stem_state.hidden_states,
                            return_state=True,
                        )
                    )
                target_model.depthspec_prefetched_verify_hidden_states = (
                    normalized_hidden.contiguous()
                )
                segment_runtime.complete_segment(
                    continuation_descriptor.segment_id
                )
                binding_store.release(continuation_descriptor.segment_id)
                segment_runtime.retire_request(
                    continuation_request_id,
                    forget_history=True,
                )
                if (
                    segment_runtime.active_count()
                    or binding_store.ready_count()
                ):
                    raise RuntimeError(
                        "Lightweight live continuation leaked scheduler state"
                    )
                return
        if self.runtime_config.diagnostics.get("ragged_draft_extend_audit", False):
            target_model.depthspec_ragged_draft_extend_audit = {
                "verify_forward_batch": verify_forward_batch,
                "full_segment": full_segment,
                "continuation_segment": continuation_segment,
                "full_stem_state": full_stem_state,
                "reference_continuation_state": None,
            }

        target_runner.attn_backend.init_forward_metadata(verify_forward_batch)
        verify_forward_batch.mark_forward_metadata_ready()
        pool = target_runner.token_to_kv_pool
        slots = verify_forward_batch.out_cache_loc.detach().cpu().tolist()
        blocks_per_loop = int(target_model.config.num_hidden_layers)
        saved_kvs = {}
        if compute_reference:
            for loop_index in continuation_loops:
                for block_index in range(blocks_per_loop):
                    layer_id = loop_index * blocks_per_loop + block_index
                    saved_kvs[(loop_index, block_index)] = (
                        pool.get_key_buffer(layer_id)[slots].clone(),
                        pool.get_value_buffer(layer_id)[slots].clone(),
                    )
        reference_continuation_state = None
        try:
            if compute_reference:
                with forward_context(
                    ForwardContext(attn_backend=target_runner.attn_backend)
                ):
                    _, reference_continuation_state = (
                        gpt.forward_with_continuation(
                            verify_forward_batch.input_ids,
                            verify_forward_batch.positions,
                            verify_forward_batch,
                            continuation_state=full_stem_state,
                            start_loop_index=2,
                            end_loop_index=int(gpt.target_loop_count),
                        )
                    )
                if self.runtime_config.diagnostics.get(
                    "ragged_draft_extend_audit", False
                ):
                    target_model.depthspec_ragged_draft_extend_audit[
                        "reference_continuation_state"
                    ] = reference_continuation_state
        finally:
            if compute_reference:
                for loop_index in continuation_loops:
                    for block_index in range(blocks_per_loop):
                        layer_id = loop_index * blocks_per_loop + block_index
                        key, value = saved_kvs[(loop_index, block_index)]
                        pool.get_key_buffer(layer_id)[slots].copy_(key)
                        pool.get_value_buffer(layer_id)[slots].copy_(value)

        segment_runtime = self.depthspec_segment_runtime
        binding_store = self.depthspec_segment_bindings
        bound_scheduler = self.depthspec_bound_scheduler
        if segment_runtime.active_count() or binding_store.ready_count():
            raise RuntimeError(
                "Persistent DepthSpec segment runtime has leaked active state"
            )
        runtime_epoch = self._depthspec_segment_runtime_epoch
        self._depthspec_segment_runtime_epoch += 1
        full_request_id = runtime_epoch * 2
        continuation_request_id = full_request_id + 1
        if schedule_full_lane:
            segment_runtime.mark_seed_completed(
                f"r{full_request_id}-k0-proposal_coarse"
            )
        segment_runtime.mark_seed_completed(
            f"r{continuation_request_id}-k0-authoritative_full_stem"
        )
        if schedule_full_lane:
            segment_runtime.register_request(
                full_request_id,
                phase="authoritative_full_stem",
                round_id=0,
                ready_tick=0,
                lane=0,
                request_count=int(batch.seq_lens.numel()),
            )
        segment_runtime.register_request(
            continuation_request_id,
            phase="target_continuation",
            round_id=0,
            ready_tick=0,
            lane=1,
            request_count=int(batch.seq_lens.numel()),
        )
        bound_segments_before = binding_store.statistics.bound_segments
        released_segments_before = binding_store.statistics.released_segments
        full_descriptor = (
            segment_runtime.descriptor_for_request(full_request_id)
            if schedule_full_lane
            else None
        )
        continuation_descriptor = segment_runtime.descriptor_for_request(
            continuation_request_id
        )
        if full_descriptor is not None:
            binding_store.bind(
                full_descriptor,
                RaggedSegmentInput(full_segment, verify_forward_batch),
            )
        binding_store.bind(
            continuation_descriptor,
            RaggedSegmentInput(
                continuation_segment,
                verify_forward_batch,
                initial_hidden_states=full_stem_state.hidden_states,
                initial_time=full_stem_state.current_time,
            ),
        )
        bound_batch = bound_scheduler.materialize_batch(tick=0)
        if bound_batch is None:
            raise RuntimeError(
                "Bound heterogeneous scheduler produced no batch"
            )
        plan = bound_batch.forward_plan
        schedule = compile_ragged_schedule(plan)
        continuation_runner = None
        if live_mode and not schedule_full_lane:
            continuation_runner = (
                target_model.depthspec_target_continuation_piecewise_runners.get(
                    int(continuation_segment.input_ids.shape[0])
                )
                if self.runtime_config.diagnostics.get(
                    "live_continuation_piecewise_graph", False
                )
                and hasattr(
                    target_model,
                    "depthspec_target_continuation_piecewise_runners",
                )
                else None
            )
        started = time.perf_counter()
        continuation_state = None
        continuation_normalized_hidden = None
        if continuation_runner is not None:
            with self._depthspec_profile_scope(
                "depthspec.continuation_piecewise_graph"
            ), forward_context(
                ForwardContext(attn_backend=target_runner.attn_backend)
            ):
                continuation_normalized_hidden, continuation_state = (
                    continuation_runner.execute(
                        continuation_segment.input_ids,
                        continuation_segment.positions,
                        verify_forward_batch,
                        initial_hidden_states=full_stem_state.hidden_states,
                        return_state=True,
                    )
                )
        else:
            with self._depthspec_profile_scope("depthspec.ragged_execute"):
                mixed_result = execute_grouped_ragged_schedule(
                    target_model,
                    target_runner.attn_backend,
                    plan,
                    shared_metadata=bool(
                        self.runtime_config.diagnostics.get(
                            "shared_ragged_attention_metadata", True
                        )
                    ),
                    save_kv_cache=live_mode,
                )
        full_rows = int(full_segment.input_ids.numel())
        continuation_rows = int(continuation_segment.input_ids.numel())
        continuation_row_start = full_rows if schedule_full_lane else 0
        if live_mode:
            if continuation_runner is not None:
                target_model.depthspec_prefetched_verify_hidden_states = (
                    continuation_normalized_hidden.contiguous()
                )
            else:
                target_model.depthspec_prefetched_verify_hidden_states = (
                    mixed_result.normalized_hidden_states[
                        continuation_row_start : continuation_row_start
                        + continuation_rows
                    ].contiguous()
                )
        with self._depthspec_profile_scope("depthspec.bound_complete"):
            next_phases = bound_scheduler.complete_batch(bound_batch)
        expected_next_phases = {
            continuation_descriptor.segment_id: (
                "target_compatible_draft_extend"
            ),
        }
        if full_descriptor is not None:
            expected_next_phases[full_descriptor.segment_id] = (
                "target_continuation"
            )
        if next_phases != expected_next_phases:
            raise RuntimeError(
                "Bound heterogeneous scheduler phase mismatch"
            )
        if schedule_full_lane:
            segment_runtime.retire_request(
                full_request_id,
                forget_history=True,
            )
        segment_runtime.retire_request(
            continuation_request_id,
            forget_history=True,
        )
        if segment_runtime.active_count() or binding_store.ready_count():
            raise RuntimeError(
                "DepthSpec segment runtime did not retire cleanly"
            )
        audit_enabled = bool(
            self.runtime_config.diagnostics.get(
                "heterogeneous_ragged_schedule_audit", False
            )
        )
        if audit_enabled:
            row = {
                "captured_unix_s": time.time(),
                "elapsed_s": time.perf_counter() - started,
                "request_count": int(batch.seq_lens.numel()),
                "qo_indptr": plan.qo_indptr.detach().cpu().tolist(),
                "padding_rows": plan.padding_rows,
                "full_payload_rows": full_rows if schedule_full_lane else 0,
                "continuation_payload_rows": continuation_rows,
                "micro_step_count": len(schedule),
                "micro_step_rows": [item.row_count for item in schedule],
                "micro_step_loops": [
                    [item.loop_index for item in step.active_segments]
                    for step in schedule
                ],
                "scheduler_statistics": {
                    "runtime": {
                        "materialized_segments": len(bound_batch.segments),
                        "materialized_batches": 1,
                        "registered_requests_total": (
                            segment_runtime.statistics.registered_requests
                        ),
                        "completed_segments_total": (
                            segment_runtime.statistics.completed_segments
                        ),
                    },
                    "bindings": {
                        "bound_segments": (
                            binding_store.statistics.bound_segments
                            - bound_segments_before
                        ),
                        "released_segments": (
                            binding_store.statistics.released_segments
                            - released_segments_before
                        ),
                        "bound_segments_total": (
                            binding_store.statistics.bound_segments
                        ),
                        "released_segments_total": (
                            binding_store.statistics.released_segments
                        ),
                    },
                },
                "segment_runtime_epoch": runtime_epoch,
                "live_consumption": live_mode,
                "execution_mode": (
                    "continuation_piecewise_graph"
                    if continuation_runner is not None
                    else "ragged_schedule"
                ),
                "schedule_full_lane": schedule_full_lane,
                "full_hidden_error": (
                    self._canonical_tensor_error(
                        mixed_result.group_states[0].hidden_states,
                        full_stem_state.hidden_states,
                    )
                    if schedule_full_lane
                    else None
                ),
                "continuation_hidden_error": (
                    self._canonical_tensor_error(
                        (
                            continuation_state.hidden_states
                            if continuation_runner is not None
                            else mixed_result.group_states[
                                1 if schedule_full_lane else 0
                            ].hidden_states
                        ),
                        reference_continuation_state.hidden_states,
                    )
                    if reference_continuation_state is not None
                    else None
                ),
            }
            audit_path = self.runtime_config.diagnostics.get(
                "heterogeneous_ragged_schedule_audit_path"
            )
            if audit_path:
                output = Path(audit_path)
                output.parent.mkdir(parents=True, exist_ok=True)
                with output.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(row) + "\n")
            else:
                logger.info("DepthSpec heterogeneous ragged audit: %s", row)

    def _audit_grouped_ragged_stem(
        self,
        batch,
        verify_forward_batch,
        full_stem_state,
    ) -> None:
        """Validate dense-merged, attention-grouped ragged stem execution."""
        target_model = self.target_worker.model_runner.model
        partial = getattr(target_model, "depthspec_partial_stem_state", None)
        if (
            partial is None
            or "segment" not in partial
            or "forward_batch" not in partial
        ):
            raise ValueError("Grouped ragged audit requires partial segment metadata")

        full_segment = LoopSegment(
            input_ids=verify_forward_batch.input_ids,
            positions=verify_forward_batch.positions,
            out_cache_loc=verify_forward_batch.out_cache_loc,
            seq_lens=batch.seq_lens,
            seq_lens_cpu=(
                batch.seq_lens_cpu
                if batch.seq_lens_cpu is not None
                else batch.seq_lens.cpu()
            ),
            seq_lens_sum=verify_forward_batch.spec_info.seq_lens_sum
            if verify_forward_batch.spec_info.seq_lens_sum is not None
            else int(batch.seq_lens.sum().item()),
            spec_info=verify_forward_batch.spec_info,
            req_pool_indices=batch.req_pool_indices,
            phase="authoritative_full_stem",
            loop_indices=(0, 1),
            step_schedule=(0.125, 0.125),
            attention_contract="tree-custom",
        )
        plan = build_ragged_forward_plan(
            [
                (partial["segment"], partial["forward_batch"]),
                (full_segment, verify_forward_batch),
            ],
            loop_indices=(0, 1),
            step_schedule=(0.125, 0.125),
        )
        started = time.perf_counter()
        _, mixed_state = execute_grouped_ragged_stem(
            target_model,
            self.target_worker.model_runner.attn_backend,
            plan,
            full_stem_state,
        )
        partial_rows = int(partial["segment"].input_ids.numel())
        full_rows = int(full_segment.input_ids.numel())
        row = {
            "captured_unix_s": time.time(),
            "elapsed_s": time.perf_counter() - started,
            "request_count": int(batch.seq_lens.numel()),
            "qo_indptr": plan.qo_indptr.detach().cpu().tolist(),
            "partial_payload_rows": partial_rows,
            "full_payload_rows": full_rows,
            "execution_rows": plan.row_count,
            "padding_rows": plan.padding_rows,
            "partial_hidden_error": self._canonical_tensor_error(
                mixed_state.hidden_states[:partial_rows],
                partial["state"].hidden_states,
            ),
            "full_hidden_error": self._canonical_tensor_error(
                mixed_state.hidden_states[partial_rows : partial_rows + full_rows],
                full_stem_state.hidden_states,
            ),
        }
        audit_path = self.runtime_config.diagnostics.get(
            "grouped_ragged_stem_audit_path"
        )
        if audit_path:
            output = Path(audit_path)
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row) + "\n")
        else:
            logger.info("DepthSpec grouped ragged stem audit: %s", row)

    def _audit_ragged_mixed_stem(
        self,
        batch,
        verify_forward_batch,
        full_stem_state,
    ) -> None:
        """Compare a no-padding partial/full mixed forward with references."""
        target_model = self.target_worker.model_runner.model
        partial = getattr(target_model, "depthspec_partial_stem_state", None)
        if partial is None or "segment" not in partial:
            raise ValueError("Ragged mixed stem audit requires a partial segment")

        full_segment = LoopSegment(
            input_ids=verify_forward_batch.input_ids,
            positions=verify_forward_batch.positions,
            out_cache_loc=verify_forward_batch.out_cache_loc,
            seq_lens=batch.seq_lens,
            seq_lens_cpu=(
                batch.seq_lens_cpu
                if batch.seq_lens_cpu is not None
                else batch.seq_lens.cpu()
            ),
            seq_lens_sum=verify_forward_batch.spec_info.seq_lens_sum
            if verify_forward_batch.spec_info.seq_lens_sum is not None
            else int(batch.seq_lens.sum().item()),
            spec_info=verify_forward_batch.spec_info,
            req_pool_indices=batch.req_pool_indices,
            phase="authoritative_full_stem",
            loop_indices=(0, 1),
            step_schedule=(0.125, 0.125),
            attention_contract="tree-custom",
        )
        partial_segment = partial["segment"]
        scope = self.runtime_config.diagnostics.get("ragged_mixed_scope", "both")
        if scope not in {"partial", "full", "both"}:
            raise ValueError(f"Unsupported ragged mixed audit scope: {scope}")
        audit_segments = {
            "partial": [partial_segment],
            "full": [full_segment],
            "both": [partial_segment, full_segment],
        }[scope]
        started = time.perf_counter()
        mixed_state, mixed_forward_batch = LoopSegmentExecutor(
            self,
            batch,
        ).execute_ragged_many(audit_segments)
        partial_rows = (
            int(partial_segment.input_ids.numel())
            if scope in {"partial", "both"}
            else 0
        )
        full_rows = (
            int(full_segment.input_ids.numel()) if scope in {"full", "both"} else 0
        )
        partial_mixed = mixed_state.hidden_states[:partial_rows]
        full_mixed = mixed_state.hidden_states[partial_rows : partial_rows + full_rows]
        row = {
            "captured_unix_s": time.time(),
            "elapsed_s": time.perf_counter() - started,
            "request_count": int(batch.seq_lens.numel()),
            "scope": scope,
            "partial_payload_rows": partial_rows,
            "full_payload_rows": full_rows,
            "execution_rows": int(mixed_forward_batch.input_ids.numel()),
            "padding_rows": 0,
            "partial_hidden_error": (
                self._canonical_tensor_error(
                    partial_mixed,
                    partial["state"].hidden_states,
                )
                if scope in {"partial", "both"}
                else None
            ),
            "full_hidden_error": (
                self._canonical_tensor_error(
                    full_mixed,
                    full_stem_state.hidden_states,
                )
                if scope in {"full", "both"}
                else None
            ),
        }
        audit_path = self.runtime_config.diagnostics.get(
            "ragged_mixed_stem_audit_path"
        )
        if audit_path:
            output = Path(audit_path)
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row) + "\n")
        else:
            logger.info("DepthSpec ragged mixed stem audit: %s", row)

    def _split_verify_forward_into_segments(
        self,
        verify_forward_batch,
        phase_prefix: str,
        sequence_lens: torch.Tensor | None = None,
        sequence_lens_cpu: torch.Tensor | None = None,
    ) -> list[LoopSegment]:
        """Split a fixed-width verify batch into one segment per request."""
        spec_info = verify_forward_batch.spec_info
        width = int(spec_info.num_tokens_per_req)
        seq_lens = (
            sequence_lens
            if sequence_lens is not None
            else verify_forward_batch.seq_lens
        )
        seq_lens_cpu = (
            sequence_lens_cpu
            if sequence_lens_cpu is not None
            else verify_forward_batch.seq_lens_cpu
        )
        if seq_lens_cpu is None:
            seq_lens_cpu = seq_lens.cpu()
        request_count = int(seq_lens.shape[0])
        segments = []
        mask_offset = 0
        for request_index in range(request_count):
            start = request_index * width
            end = start + width
            sequence_length = int(seq_lens[request_index].item())
            mask_width = sequence_length + int(
                getattr(spec_info, "num_kv_tokens_per_req", width)
            )
            mask_start = mask_offset
            mask_end = mask_start + mask_width * width
            mask_offset = mask_end
            request_spec = replace(
                spec_info,
                draft_token=spec_info.draft_token[start:end],
                custom_mask=spec_info.custom_mask[
                    mask_start:mask_end
                ],
                positions=spec_info.positions[start:end],
                retrieve_index=spec_info.retrieve_index[
                    request_index : request_index + 1
                ],
                retrieve_next_token=spec_info.retrieve_next_token[
                    request_index : request_index + 1
                ],
                retrieve_next_sibling=spec_info.retrieve_next_sibling[
                    request_index : request_index + 1
                ],
                seq_lens_sum=sequence_length,
                seq_lens_cpu=seq_lens_cpu[request_index : request_index + 1],
            )
            segments.append(
                LoopSegment(
                    input_ids=verify_forward_batch.input_ids[start:end],
                    positions=verify_forward_batch.positions[start:end],
                    out_cache_loc=verify_forward_batch.out_cache_loc[start:end],
                    seq_lens=seq_lens[request_index : request_index + 1],
                    seq_lens_cpu=seq_lens_cpu[request_index : request_index + 1],
                    seq_lens_sum=sequence_length,
                    spec_info=request_spec,
                    req_pool_indices=verify_forward_batch.req_pool_indices[
                        request_index : request_index + 1
                    ],
                    phase=f"{phase_prefix}_{request_index}",
                    loop_indices=(0, 1),
                    step_schedule=(0.125, 0.125),
                    attention_contract="tree-custom",
                )
            )
        return segments

    @staticmethod
    def _canonical_tensor_error(left: torch.Tensor, right: torch.Tensor) -> dict:
        difference = (left.float() - right.float()).abs()
        return {
            "max_abs": float(difference.max()),
            "mean_abs": float(difference.mean()),
            "relative_l2": float(
                (left.float() - right.float()).norm()
                / right.float().norm().clamp_min(1e-12)
            ),
        }

    def _audit_partial_canonical_stem(
        self,
        target_model,
        target_runner,
        verify_forward_batch,
        stem_state,
    ) -> None:
        """Compare an early partial stem with the completed full-tree stem."""
        audit_path = self.runtime_config.diagnostics.get("partial_stem_audit_path")
        if not audit_path:
            return
        partial = getattr(target_model, "depthspec_partial_stem_state", None)
        if partial is None:
            raise ValueError("Partial stem audit ran without a cached partial state")
        if partial["slots"] is None or partial["kvs"] is None:
            raise ValueError("Partial stem audit ran without an audit snapshot")

        row_count = int(partial["state"].hidden_states.shape[0])
        pool = target_runner.token_to_kv_pool
        full_tokens = verify_forward_batch.input_ids.detach().cpu().tolist()
        full_positions = verify_forward_batch.positions.detach().cpu().tolist()
        current_slots = verify_forward_batch.out_cache_loc.tolist()
        partial_keys = {
            (token, position): index
            for index, (token, position) in enumerate(
                zip(partial["tokens"], partial["positions"])
            )
        }
        mapping = []
        hidden_errors = []
        kv_errors = {}

        def add_kv_error(loop_offset, block_index, tensor_name, error):
            key = f"loop{loop_offset}/block{block_index}"
            kv_errors.setdefault(key, {})[tensor_name] = error

        for full_row, row_key in enumerate(zip(full_tokens, full_positions)):
            partial_row = partial_keys.get(row_key)
            if partial_row is None:
                continue
            mapping.append((full_row, partial_row))
            hidden_errors.append(
                self._canonical_tensor_error(
                    partial["state"].hidden_states[partial_row],
                    stem_state.hidden_states[full_row],
                )
            )
            for (loop_offset, block_index), (
                partial_key,
                partial_value,
            ) in partial["kvs"].items():
                layer_id = (
                    loop_offset * int(target_model.config.num_hidden_layers)
                    + block_index
                )
                full_key = pool.get_key_buffer(layer_id)[current_slots[full_row]]
                full_value = pool.get_value_buffer(layer_id)[current_slots[full_row]]
                add_kv_error(
                    loop_offset,
                    block_index,
                    "key",
                    self._canonical_tensor_error(partial_key[partial_row], full_key),
                )
                add_kv_error(
                    loop_offset,
                    block_index,
                    "value",
                    self._canonical_tensor_error(
                        partial_value[partial_row], full_value
                    ),
                )

        mapped_slots_match = all(
            current_slots[full_row] == partial["slots"][partial_row]
            for full_row, partial_row in mapping
        )
        use_partial = (
            self.runtime_config.use_partial_stem and bool(mapping) and mapped_slots_match
        )
        if use_partial:
            for full_row, partial_row in mapping:
                stem_state.hidden_states[full_row].copy_(
                    partial["state"].hidden_states[partial_row]
                )

        max_hidden_error = max(
            (error["relative_l2"] for error in hidden_errors), default=0.0
        )
        max_key_error = max(
            (
                metrics["key"]["relative_l2"]
                for metrics in kv_errors.values()
                if "key" in metrics
            ),
            default=0.0,
        )
        max_value_error = max(
            (
                metrics["value"]["relative_l2"]
                for metrics in kv_errors.values()
                if "value" in metrics
            ),
            default=0.0,
        )

        marker_path = f"{audit_path}.count"
        try:
            audit_index = int(Path(marker_path).read_text(encoding="utf-8"))
        except (FileNotFoundError, ValueError):
            audit_index = 0
        row = {
            "audit_index": audit_index,
            "row_count": row_count,
            "full_row_count": len(full_tokens),
            "mapped_row_count": len(mapping),
            "coverage_fraction": len(mapping) / len(full_tokens),
            "mapping": mapping,
            "mapped_slots_match": mapped_slots_match,
            "used_partial": use_partial,
            "full_tokens": full_tokens,
            "full_positions": full_positions,
            "partial_tokens": partial["tokens"],
            "partial_positions": partial["positions"],
            "max_hidden_error": {
                "relative_l2": max_hidden_error,
            },
            "max_key_error": {
                "relative_l2": max_key_error,
            },
            "max_value_error": {
                "relative_l2": max_value_error,
            },
            "kv_errors": kv_errors,
        }
        output = Path(audit_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row) + "\n")
        Path(marker_path).write_text(str(audit_index + 1), encoding="utf-8")
        del target_model.depthspec_partial_stem_state

    def _prepare_full_tree_missing_rows(
        self,
        batch,
        target_model,
        target_runner,
    ) -> None:
        """Compute a query subset while retaining the full tree KV metadata."""
        from sglang.srt.model_executor.forward_context import (
            ForwardContext,
            forward_context,
        )
        from sglang.srt.speculative.eagle_info import EagleVerifyInput
        from sglang.srt.speculative.eagle_utils import eagle_prepare_for_verify

        partial = getattr(target_model, "depthspec_partial_stem_state", None)
        if partial is None:
            raise ValueError("Full-tree missing rows require an early partial state")

        verify_input = batch.spec_info
        batch_size = int(batch.seq_lens.numel())
        width = int(verify_input.draft_token_num)
        full_token_tensor = verify_input.draft_token.reshape(batch_size, width)
        full_position_tensor = verify_input.positions.reshape(batch_size, width)
        full_tokens = full_token_tensor.tolist()
        full_positions = full_position_tensor.tolist()
        if len(partial["tokens"]) % batch_size != 0:
            raise ValueError("Partial stem rows are not divisible by batch size")
        partial_width = len(partial["tokens"]) // batch_size
        mapping = {}
        missing_rows_by_request = []
        for request_index in range(batch_size):
            partial_start = request_index * partial_width
            available = {}
            for partial_local_row in range(partial_width):
                partial_row = partial_start + partial_local_row
                key = (
                    partial["tokens"][partial_row],
                    partial["positions"][partial_row],
                )
                available.setdefault(key, []).append(partial_row)
            request_missing_rows = []
            for full_local_row, row_key in enumerate(
                zip(
                    full_tokens[request_index],
                    full_positions[request_index],
                    strict=True,
                )
            ):
                candidates = available.get(row_key, [])
                full_row = request_index * width + full_local_row
                if candidates:
                    mapping[full_row] = candidates.pop(0)
                else:
                    request_missing_rows.append(full_local_row)
            missing_rows_by_request.append(request_missing_rows)
        if not mapping:
            raise ValueError("Full-tree missing rows found no reusable early rows")
        missing_counts = [len(rows) for rows in missing_rows_by_request]
        query_count = max(missing_counts)
        final_hidden = partial["state"].hidden_states.new_empty(
            (batch_size * width, partial["state"].hidden_states.shape[1])
        )
        if query_count == 0:
            for full_row, partial_row in mapping.items():
                final_hidden[full_row].copy_(
                    partial["state"].hidden_states[partial_row]
                )
            final_state = type(partial["state"])(
                hidden_states=final_hidden,
                current_time=partial["state"].current_time,
                next_loop_index=partial["state"].next_loop_index,
            )
            target_model.depthspec_canonical_stem_state = final_state
            return

        request_rows = self.req_to_token_pool.req_to_token[
            batch.req_pool_indices.long()
        ]
        tree_offsets = torch.arange(width, device=batch.device, dtype=torch.long)
        full_pool_positions = batch.seq_lens[:, None].long() + tree_offsets[None, :]
        full_slots_tensor = torch.gather(request_rows, 1, full_pool_positions)
        query_rows_by_request = []
        for request_index, request_missing_rows in enumerate(
            missing_rows_by_request
        ):
            query_rows = list(request_missing_rows)
            if len(query_rows) < query_count:
                reusable_rows = [
                    full_local_row
                    for full_local_row in range(width)
                    if request_index * width + full_local_row in mapping
                ]
                if not reusable_rows:
                    raise ValueError(
                        "Full-tree padding requires at least one reusable row"
                    )
                query_rows.extend(
                    reusable_rows[index % len(reusable_rows)]
                    for index in range(query_count - len(query_rows))
                )
            query_rows_by_request.append(query_rows)
        query_rows_tensor = torch.tensor(
            query_rows_by_request,
            dtype=torch.long,
            device=batch.device,
        )
        if self.runtime_config.diagnostics.get("cross_request_shape_audit", False):
            logger.info(
                "DepthSpec cross-request missing-row audit: req_pool=%s "
                "seq_lens=%s missing=%s query_rows=%s",
                batch.req_pool_indices.tolist(),
                batch.seq_lens.tolist(),
                missing_rows_by_request,
                query_rows_by_request,
            )
        query_slots_tensor = torch.gather(
            full_slots_tensor, 1, query_rows_tensor
        )
        valid_query_counts = torch.tensor(
            missing_counts,
            dtype=torch.long,
            device=batch.device,
        )
        query_offsets = torch.arange(
            query_count,
            dtype=torch.long,
            device=batch.device,
        )
        padding_mask = query_offsets[None, :] >= valid_query_counts[:, None]
        query_slots_tensor.masked_fill_(padding_mask, 0)
        query_token_tensor = torch.gather(
            full_token_tensor, 1, query_rows_tensor
        )
        query_position_tensor = torch.gather(
            full_position_tensor, 1, query_rows_tensor
        )

        if batch.seq_lens_cpu is None:
            sequence_lengths = batch.seq_lens.detach().cpu().tolist()
        else:
            sequence_lengths = batch.seq_lens_cpu.tolist()
        selected_masks = []
        mask_offset = 0
        for request_index, sequence_length in enumerate(sequence_lengths):
            row_width = int(sequence_length) + width
            request_mask_size = width * row_width
            request_mask = verify_input.custom_mask[
                mask_offset : mask_offset + request_mask_size
            ].reshape(width, row_width)
            selected_masks.append(
                request_mask[query_rows_tensor[request_index]].reshape(-1)
            )
            mask_offset += request_mask_size
        if mask_offset != int(verify_input.custom_mask.numel()):
            raise ValueError(
                "Full-tree missing-row mask layout mismatch: "
                f"{mask_offset} != {verify_input.custom_mask.numel()}"
            )
        selected_mask = torch.cat(selected_masks)
        post_seq_lens = batch.seq_lens + width
        post_seq_lens_cpu = (
            batch.seq_lens_cpu + width
            if batch.seq_lens_cpu is not None
            else batch.seq_lens.cpu() + width
        )
        subset_input = EagleVerifyInput(
            draft_token=query_token_tensor.reshape(-1),
            custom_mask=selected_mask,
            positions=query_position_tensor.reshape(-1),
            retrieve_index=verify_input.retrieve_index,
            retrieve_next_token=verify_input.retrieve_next_token,
            retrieve_next_sibling=verify_input.retrieve_next_sibling,
            retrieve_cum_len=verify_input.retrieve_cum_len,
            spec_steps=verify_input.spec_steps,
            topk=verify_input.topk,
            draft_token_num=query_count,
            num_tokens_per_req=query_count,
            num_kv_tokens_per_req=width,
            depthspec_preserve_full_tree_kv=True,
            capture_hidden_mode=verify_input.capture_hidden_mode,
            seq_lens_sum=int(post_seq_lens.sum().item()),
            seq_lens_cpu=post_seq_lens_cpu,
        )
        segment = LoopSegment(
            input_ids=subset_input.draft_token,
            positions=subset_input.positions,
            out_cache_loc=query_slots_tensor.reshape(-1),
            seq_lens=post_seq_lens,
            seq_lens_cpu=post_seq_lens_cpu,
            seq_lens_sum=subset_input.seq_lens_sum,
            spec_info=subset_input,
            req_pool_indices=batch.req_pool_indices,
            phase="full_tree_missing_rows",
            loop_indices=(0, 1),
            step_schedule=(0.125, 0.125),
            attention_contract="tree-custom-full-kv",
        )
        started = time.perf_counter()
        missing_state, _ = LoopSegmentExecutor(self, batch).execute(segment)
        _append_stage_timing(
            "full_tree_missing_rows",
            batch_size,
            time.perf_counter() - started,
        )

        for full_row, partial_row in mapping.items():
            final_hidden[full_row].copy_(partial["state"].hidden_states[partial_row])
        missing_full_rows = [
            request_index * width + full_local_row
            for request_index, request_missing_rows in enumerate(
                missing_rows_by_request
            )
            for full_local_row in request_missing_rows
        ]
        missing_state_rows = [
            request_index * query_count + query_local_row
            for request_index, request_missing_rows in enumerate(
                missing_rows_by_request
            )
            for query_local_row in range(len(request_missing_rows))
        ]
        final_hidden.index_copy_(
            0,
            torch.tensor(missing_full_rows, dtype=torch.long, device=batch.device),
            missing_state.hidden_states[
                torch.tensor(
                    missing_state_rows,
                    dtype=torch.long,
                    device=batch.device,
                )
            ],
        )
        final_state = type(partial["state"])(
            hidden_states=final_hidden,
            current_time=missing_state.current_time,
            next_loop_index=missing_state.next_loop_index,
        )
        target_model.depthspec_canonical_stem_state = final_state

        full_comparison = None
        if self.runtime_config.diagnostics.get("audit_partial_only_full", False):
            if batch_size != 1:
                raise ValueError(
                    "Full-tree numerical audit currently requires batch size 1"
                )
            pool = target_runner.token_to_kv_pool
            full_slots = full_slots_tensor.reshape(-1).tolist()
            blocks_per_loop = int(target_model.config.num_hidden_layers)
            partial_only_kvs = {}
            for loop_offset in range(2):
                for block_index in range(blocks_per_loop):
                    layer_id = loop_offset * blocks_per_loop + block_index
                    partial_only_kvs[(loop_offset, block_index)] = (
                        pool.get_key_buffer(layer_id)[full_slots].clone(),
                        pool.get_value_buffer(layer_id)[full_slots].clone(),
                    )
            saved_full_probe_state = {
                name: getattr(batch, name, None)
                for name in (
                    "input_ids",
                    "out_cache_loc",
                    "out_cache_loc_dsv4",
                    "seq_lens",
                    "seq_lens_cpu",
                    "seq_lens_sum",
                    "forward_mode",
                    "spec_info",
                )
            }
            try:
                full_forward_batch, can_run_graph = eagle_prepare_for_verify(
                    batch.spec_info,
                    self.req_to_token_pool,
                    batch,
                    self.target_worker,
                )
                if can_run_graph:
                    raise ValueError(
                        "Full-tree subset full audit unexpectedly selected a graph"
                    )
                target_runner.attn_backend.init_forward_metadata(full_forward_batch)
                full_forward_batch.mark_forward_metadata_ready()
                with forward_context(
                    ForwardContext(attn_backend=target_runner.attn_backend)
                ):
                    _, full_state = target_model.gpt.forward_branch(
                        full_forward_batch.input_ids,
                        full_forward_batch.positions,
                        full_forward_batch,
                        loop_indices=(0, 1),
                        step_schedule=(0.125, 0.125),
                        return_state=True,
                    )
            finally:
                for name, value in saved_full_probe_state.items():
                    setattr(batch, name, value)

            hidden_errors = [
                self._canonical_tensor_error(
                    final_hidden[full_row], full_state.hidden_states[full_row]
                )
                for full_row in range(width)
            ]
            kv_errors = {}
            for (loop_offset, block_index), (
                partial_key,
                partial_value,
            ) in partial_only_kvs.items():
                layer_id = loop_offset * blocks_per_loop + block_index
                full_key = pool.get_key_buffer(layer_id)[full_slots]
                full_value = pool.get_value_buffer(layer_id)[full_slots]
                kv_errors[f"loop{loop_offset}/block{block_index}"] = {
                    "key": self._canonical_tensor_error(partial_key, full_key),
                    "value": self._canonical_tensor_error(partial_value, full_value),
                }
            full_comparison = {
                "hidden_errors": hidden_errors,
                "kv_errors": kv_errors,
                "max_hidden_relative_l2": max(
                    error["relative_l2"] for error in hidden_errors
                ),
                "max_key_relative_l2": max(
                    metrics["key"]["relative_l2"] for metrics in kv_errors.values()
                ),
                "max_value_relative_l2": max(
                    metrics["value"]["relative_l2"] for metrics in kv_errors.values()
                ),
            }

        audit_path = self.runtime_config.diagnostics.get("partial_stem_audit_path")
        if audit_path:
            marker_path = f"{audit_path}.count"
            try:
                audit_index = int(Path(marker_path).read_text(encoding="utf-8"))
            except (FileNotFoundError, ValueError):
                audit_index = 0
            row = {
                "audit_index": audit_index,
                "mode": "full_tree_missing_rows",
                "batch_size": batch_size,
                "full_row_count": batch_size * width,
                "mapped_row_count": len(mapping),
                "missing_row_count": sum(
                    len(rows) for rows in missing_rows_by_request
                ),
                "coverage_fraction": len(mapping) / (batch_size * width),
                "mapping": {
                    str(full_row): partial_row
                    for full_row, partial_row in mapping.items()
                },
                "missing_rows_by_request": missing_rows_by_request,
                "query_rows_by_request": query_rows_by_request,
                "query_slots": query_slots_tensor.reshape(-1).tolist(),
                "query_count_per_request": query_count,
                "kv_tokens_per_req": width,
            }
            if full_comparison is not None:
                row["full_comparison"] = full_comparison
            output = Path(audit_path)
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row) + "\n")
            Path(marker_path).write_text(str(audit_index + 1), encoding="utf-8")
        del target_model.depthspec_partial_stem_state

    def _prepare_partial_only_canonical_stem(
        self,
        batch,
        target_model,
        target_runner,
    ) -> None:
        """Compute only final-tree rows missed by the early partial stem."""
        if self.loop_segment_executor_enabled and self.runtime_config.full_tree_missing_rows:
            self._prepare_full_tree_missing_rows(batch, target_model, target_runner)
            return

        from sglang.srt.model_executor.forward_context import (
            ForwardContext,
            forward_context,
        )
        from sglang.srt.speculative.eagle_info import EagleVerifyInput
        from sglang.srt.speculative.eagle_utils import eagle_prepare_for_verify

        if int(batch.seq_lens.numel()) != 1:
            raise ValueError("Partial-only stem pilot currently requires batch size 1")
        partial = getattr(target_model, "depthspec_partial_stem_state", None)
        if partial is None:
            raise ValueError("Partial-only stem requires an early partial state")

        verify_input = batch.spec_info
        request_index = 0
        sequence_length = int(batch.seq_lens[request_index].item())
        width = int(verify_input.draft_token_num)
        full_tokens = verify_input.draft_token.reshape(width).tolist()
        full_positions = verify_input.positions.reshape(width).tolist()
        partial_keys = {
            (token, position): index
            for index, (token, position) in enumerate(
                zip(partial["tokens"], partial["positions"])
            )
        }
        mapping = {}
        missing_rows = []
        for full_row, row_key in enumerate(zip(full_tokens, full_positions)):
            partial_row = partial_keys.get(row_key)
            if partial_row is not None:
                mapping[full_row] = partial_row
            else:
                missing_rows.append(full_row)
        if not mapping:
            raise ValueError("Partial-only stem found no reusable early rows")

        request_rows = self.req_to_token_pool.req_to_token[
            batch.req_pool_indices[request_index]
        ]
        full_mask = verify_input.custom_mask
        mask_segments = []
        synthetic_seq_lens = []
        missing_tokens = []
        missing_positions = []
        missing_slots = []

        for full_row in missing_rows:
            row_start = full_row * (sequence_length + width)
            tree_visible = [
                tree_row
                for tree_row in range(width)
                if tree_row != full_row
                and bool(full_mask[row_start + sequence_length + tree_row].item())
            ]
            if not tree_visible:
                raise ValueError(
                    f"Partial-only stem row {full_row} has no visible tree ancestor"
                )
            deepest_tree_prefix = max(
                tree_row if tree_row < full_row else -1 for tree_row in tree_visible
            )
            if deepest_tree_prefix < 0:
                raise ValueError(
                    f"Partial-only stem row {full_row} has no prior tree ancestor"
                )
            prefix_length = sequence_length + deepest_tree_prefix + 1
            row_mask = torch.zeros(
                prefix_length + 1, dtype=torch.bool, device=batch.device
            )
            row_mask[:sequence_length] = True
            for tree_row in tree_visible:
                if tree_row < full_row:
                    row_mask[sequence_length + tree_row] = True
            row_mask[-1] = True
            mask_segments.append(row_mask)
            synthetic_seq_lens.append(prefix_length)
            missing_tokens.append(full_tokens[full_row])
            missing_positions.append(full_positions[full_row])
            missing_slots.append(int(request_rows[sequence_length + full_row].item()))

        saved_state = {
            name: getattr(batch, name, None)
            for name in (
                "input_ids",
                "out_cache_loc",
                "out_cache_loc_dsv4",
                "req_pool_indices",
                "seq_lens",
                "seq_lens_cpu",
                "seq_lens_sum",
                "forward_mode",
                "spec_info",
            )
        }
        missing_states = []
        try:
            for missing_index, full_row in enumerate(missing_rows):
                synthetic_input = EagleVerifyInput(
                    draft_token=torch.tensor(
                        [missing_tokens[missing_index]],
                        dtype=torch.long,
                        device=batch.device,
                    ),
                    custom_mask=mask_segments[missing_index],
                    positions=torch.tensor(
                        [missing_positions[missing_index]],
                        dtype=torch.long,
                        device=batch.device,
                    ),
                    retrieve_index=torch.zeros(
                        (1, 1), dtype=torch.int64, device=batch.device
                    ),
                    retrieve_next_token=torch.full(
                        (1, 1), -1, dtype=torch.int64, device=batch.device
                    ),
                    retrieve_next_sibling=torch.full(
                        (1, 1), -1, dtype=torch.int64, device=batch.device
                    ),
                    retrieve_cum_len=None,
                    spec_steps=0,
                    topk=1,
                    draft_token_num=1,
                    capture_hidden_mode=None,
                    seq_lens_sum=synthetic_seq_lens[missing_index],
                    seq_lens_cpu=torch.tensor(
                        [synthetic_seq_lens[missing_index]],
                        dtype=torch.int64,
                        device="cpu",
                    ),
                )
                batch.input_ids = synthetic_input.draft_token
                batch.req_pool_indices = saved_state["req_pool_indices"]
                batch.seq_lens = torch.tensor(
                    [synthetic_seq_lens[missing_index]],
                    dtype=torch.int64,
                    device=batch.device,
                )
                batch.seq_lens_cpu = synthetic_input.seq_lens_cpu
                batch.seq_lens_sum = synthetic_input.seq_lens_sum
                batch.spec_info = synthetic_input
                missing_forward_batch, can_run_graph = eagle_prepare_for_verify(
                    synthetic_input,
                    self.req_to_token_pool,
                    batch,
                    self.target_worker,
                )
                if can_run_graph:
                    raise ValueError("Partial-only stem unexpectedly selected a graph")
                missing_forward_batch.out_cache_loc = torch.tensor(
                    [missing_slots[missing_index]],
                    dtype=torch.long,
                    device=batch.device,
                )
                target_runner.attn_backend.init_forward_metadata(missing_forward_batch)
                missing_forward_batch.mark_forward_metadata_ready()
                with forward_context(
                    ForwardContext(attn_backend=target_runner.attn_backend)
                ):
                    _, missing_state = target_model.gpt.forward_branch(
                        missing_forward_batch.input_ids,
                        missing_forward_batch.positions,
                        missing_forward_batch,
                        loop_indices=(0, 1),
                        step_schedule=(0.125, 0.125),
                        return_state=True,
                    )
                missing_states.append(missing_state)
        finally:
            for name, value in saved_state.items():
                setattr(batch, name, value)

        final_hidden = torch.empty_like(
            partial["state"].hidden_states[0].repeat(width, 1)
        )
        for full_row, partial_row in mapping.items():
            final_hidden[full_row].copy_(partial["state"].hidden_states[partial_row])
        missing_index = 0
        for full_row in missing_rows:
            final_hidden[full_row].copy_(missing_states[missing_index].hidden_states[0])
            missing_index += 1
        final_state = type(partial["state"])(
            hidden_states=final_hidden,
            current_time=missing_states[0].current_time,
            next_loop_index=missing_states[0].next_loop_index,
        )
        target_model.depthspec_canonical_stem_state = final_state

        full_comparison = None
        if self.runtime_config.diagnostics.get("audit_partial_only_full", False):
            full_slots = [
                int(request_rows[sequence_length + full_row].item())
                for full_row in range(width)
            ]
            partial_only_kvs = {}
            pool = target_runner.token_to_kv_pool
            blocks_per_loop = int(target_model.config.num_hidden_layers)
            for loop_offset in range(2):
                for block_index in range(blocks_per_loop):
                    layer_id = loop_offset * blocks_per_loop + block_index
                    partial_only_kvs[(loop_offset, block_index)] = (
                        pool.get_key_buffer(layer_id)[full_slots].clone(),
                        pool.get_value_buffer(layer_id)[full_slots].clone(),
                    )
            saved_full_probe_state = {
                name: getattr(batch, name, None)
                for name in (
                    "input_ids",
                    "out_cache_loc",
                    "out_cache_loc_dsv4",
                    "req_pool_indices",
                    "seq_lens",
                    "seq_lens_cpu",
                    "seq_lens_sum",
                    "forward_mode",
                    "spec_info",
                )
            }
            try:
                full_forward_batch, full_can_run_graph = eagle_prepare_for_verify(
                    batch.spec_info,
                    self.req_to_token_pool,
                    batch,
                    self.target_worker,
                )
                if full_can_run_graph:
                    raise ValueError("Partial-only full audit selected a graph")
                target_runner.attn_backend.init_forward_metadata(full_forward_batch)
                full_forward_batch.mark_forward_metadata_ready()
                with forward_context(
                    ForwardContext(attn_backend=target_runner.attn_backend)
                ):
                    _, full_state = target_model.gpt.forward_branch(
                        full_forward_batch.input_ids,
                        full_forward_batch.positions,
                        full_forward_batch,
                        loop_indices=(0, 1),
                        step_schedule=(0.125, 0.125),
                        return_state=True,
                    )
            finally:
                for name, value in saved_full_probe_state.items():
                    setattr(batch, name, value)

            hidden_errors = [
                self._canonical_tensor_error(
                    final_hidden[full_row],
                    full_state.hidden_states[full_row],
                )
                for full_row in range(width)
            ]
            kv_errors = {}
            for (loop_offset, block_index), (
                partial_key,
                partial_value,
            ) in partial_only_kvs.items():
                layer_id = loop_offset * blocks_per_loop + block_index
                full_key = pool.get_key_buffer(layer_id)[full_slots]
                full_value = pool.get_value_buffer(layer_id)[full_slots]
                kv_errors[f"loop{loop_offset}/block{block_index}"] = {
                    "key": self._canonical_tensor_error(partial_key, full_key),
                    "value": self._canonical_tensor_error(partial_value, full_value),
                }
            full_comparison = {
                "hidden_errors": hidden_errors,
                "kv_errors": kv_errors,
                "max_hidden_relative_l2": max(
                    error["relative_l2"] for error in hidden_errors
                ),
                "max_key_relative_l2": max(
                    metrics["key"]["relative_l2"] for metrics in kv_errors.values()
                ),
                "max_value_relative_l2": max(
                    metrics["value"]["relative_l2"] for metrics in kv_errors.values()
                ),
            }
            if self.runtime_config.diagnostics.get(
                "partial_only_full_fallback", False
            ):
                target_model.depthspec_canonical_stem_state = full_state

        audit_path = self.runtime_config.diagnostics.get("partial_stem_audit_path")
        if audit_path:
            marker_path = f"{audit_path}.count"
            try:
                audit_index = int(Path(marker_path).read_text(encoding="utf-8"))
            except (FileNotFoundError, ValueError):
                audit_index = 0
            row = {
                "audit_index": audit_index,
                "mode": "partial_only",
                "row_count": len(partial["tokens"]),
                "full_row_count": width,
                "mapped_row_count": len(mapping),
                "missing_row_count": len(missing_rows),
                "coverage_fraction": len(mapping) / width,
                "mapping": {
                    str(full_row): partial_row
                    for full_row, partial_row in mapping.items()
                },
                "missing_rows": missing_rows,
                "missing_slots": missing_slots,
                "synthetic_seq_lens": synthetic_seq_lens,
            }
            if full_comparison is not None:
                row["full_comparison"] = full_comparison
            output = Path(audit_path)
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row) + "\n")
            Path(marker_path).write_text(str(audit_index + 1), encoding="utf-8")
        del target_model.depthspec_partial_stem_state
