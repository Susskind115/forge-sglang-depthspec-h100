"""Resolve DepthSpec runtime behavior once during worker initialization."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)

DepthSpecProfile = Literal["auto", "legacy", "canonical-faas", "diagnostic"]
TargetVerifyGraphMode = Literal["auto", "disabled", "enabled"]

_LOOPFORMER_ARCHITECTURE = "LoopFormerSGLangForCausalLM"
_LEGACY_BEHAVIOR_ENV_NAMES = (
    "DEPTHSPEC_CONTINUATION_MODE",
    "DEPTHSPEC_PREFIX_SHIFT_AUDIT",
    "DEPTHSPEC_DRAFT_RELAY_ORACLE_MODE",
    "DEPTHSPEC_UNIFIED_RUNNER",
    "DEPTHSPEC_LOOP_SEGMENT_EXECUTOR",
    "DEPTHSPEC_PROPOSAL_STEM_ARTIFACTS",
    "DEPTHSPEC_DRAFT_ACTIVE_PATH_GRAPH",
    "DEPTHSPEC_CANONICAL_PROPOSAL_REPLACEMENT",
    "DEPTHSPEC_STANDALONE_TARGET_ALIGNED_LAYOUT",
    "DEPTHSPEC_CANONICAL_STEM_ORACLE",
    "DEPTHSPEC_PARTIAL_STEM_AUDIT",
    "DEPTHSPEC_PARTIAL_STEM_ONLY",
    "DEPTHSPEC_FULL_TREE_MISSING_ROWS",
    "DEPTHSPEC_DRAFT_ACTIVE_PATH_EXTEND",
    "DEPTHSPEC_DRAFT_ACTIVE_PATH_FIXED",
    "DEPTHSPEC_STANDALONE_TARGET_VERIFY_DRAFT_EXTEND",
    "DEPTHSPEC_USE_PARTIAL_STEM",
    "DEPTHSPEC_ENABLE_TARGET_VERIFY_GRAPH",
    "DEPTHSPEC_DISABLE_TARGET_VERIFY_GRAPH",
)
_LEGACY_DIAGNOSTIC_ENV_MAP = {
    "DEPTHSPEC_STAGE_TIMING_PATH": "stage_timing_path",
    "DEPTHSPEC_VERIFY_PHASE_TIMING_PATH": "verify_phase_timing_path",
    "DEPTHSPEC_PREFIX_ARTIFACT_AUDIT": "prefix_artifact_audit",
    "DEPTHSPEC_DRAFT_ARTIFACT_RELAY": "draft_artifact_relay",
    "DEPTHSPEC_STANDALONE_LAYOUT_AUDIT_PATH": "standalone_layout_audit_path",
    "DEPTHSPEC_DRAFT_RELAY_MINI_BONUS": "draft_relay_mini_bonus",
    "DEPTHSPEC_DRAFT_RELAY_MINI_PATH": "draft_relay_mini_path",
    "DEPTHSPEC_PREFIX_STATE_HANDOFF": "prefix_state_handoff",
    "DEPTHSPEC_TARGET_FIRST_LEVEL_RELAY": "target_first_level_relay",
    "DEPTHSPEC_ACTIVE_GRAPH_AUDIT": "active_graph_audit",
    "DEPTHSPEC_TARGET_DRAFT_KV_AUDIT_PATH": "target_draft_kv_audit_path",
    "DEPTHSPEC_TARGET_DRAFT_KV_AUDIT_MAX": "target_draft_kv_audit_max",
    "DEPTHSPEC_KV_SOURCE_FACTORIAL_AUDIT_PATH": "kv_source_factorial_audit_path",
    "DEPTHSPEC_KV_SOURCE_FACTORIAL_AUDIT_MAX": "kv_source_factorial_audit_max",
    "DEPTHSPEC_TREE_AUDIT_PATH": "tree_audit_path",
    "DEPTHSPEC_TREE_AUDIT_MAX_ROUNDS": "tree_audit_max_rounds",
    "DEPTHSPEC_PREFIX_ARTIFACT_AUDIT_PATH": "prefix_artifact_audit_path",
    "DEPTHSPEC_PREFIX_ARTIFACT_AUDIT_MAX": "prefix_artifact_audit_max",
    "DEPTHSPEC_PREFIX_ROOT_AUDIT_DUMP_DIR": "prefix_root_audit_dump_dir",
    "DEPTHSPEC_VERIFY_PROFILE_PATH": "verify_profile_path",
    "DEPTHSPEC_VERIFY_PROFILE_SKIP": "verify_profile_skip",
    "DEPTHSPEC_VERIFY_PROFILE_ACTIVE": "verify_profile_active",
    "DEPTHSPEC_PARTIAL_STEM_AUDIT_PATH": "partial_stem_audit_path",
    "DEPTHSPEC_AUDIT_PARTIAL_ONLY_FULL": "audit_partial_only_full",
    "DEPTHSPEC_PARTIAL_ONLY_FULL_FALLBACK": "partial_only_full_fallback",
    "DEPTHSPEC_TARGET_GRAPH_AUDIT_PATH": "target_graph_audit_path",
    "DEPTHSPEC_VERIFY_ALIGN_NORMAL_METADATA": "verify_align_normal_metadata",
    "DEPTHSPEC_PREFIX_ARTIFACT_POSITION_OFFSET": "prefix_artifact_position_offset",
    "DEPTHSPEC_TARGET_PREFIX_DUMP_DIR": "target_prefix_dump_dir",
    "DEPTHSPEC_TARGET_PREFIX_DUMP_MAX": "target_prefix_dump_max",
    "DEPTHSPEC_PREFIX_HANDOFF_AUDIT_PATH": "prefix_handoff_audit_path",
    "DEPTHSPEC_PREFIX_HANDOFF_AUDIT_MAX": "prefix_handoff_audit_max",
    "DEPTHSPEC_PREFIX_SHIFT_POSITION_OFFSET": "prefix_shift_position_offset",
}
_LEGACY_DIAGNOSTIC_BOOL_ENV_NAMES = {
    "DEPTHSPEC_PREFIX_ARTIFACT_AUDIT",
    "DEPTHSPEC_DRAFT_ARTIFACT_RELAY",
    "DEPTHSPEC_DRAFT_RELAY_MINI_BONUS",
    "DEPTHSPEC_DRAFT_RELAY_MINI_PATH",
    "DEPTHSPEC_PREFIX_STATE_HANDOFF",
    "DEPTHSPEC_TARGET_FIRST_LEVEL_RELAY",
    "DEPTHSPEC_ACTIVE_GRAPH_AUDIT",
    "DEPTHSPEC_AUDIT_PARTIAL_ONLY_FULL",
    "DEPTHSPEC_PARTIAL_ONLY_FULL_FALLBACK",
    "DEPTHSPEC_VERIFY_ALIGN_NORMAL_METADATA",
}
_LEGACY_DIAGNOSTIC_INT_ENV_NAMES = {
    "DEPTHSPEC_TARGET_DRAFT_KV_AUDIT_MAX",
    "DEPTHSPEC_KV_SOURCE_FACTORIAL_AUDIT_MAX",
    "DEPTHSPEC_TREE_AUDIT_MAX_ROUNDS",
    "DEPTHSPEC_PREFIX_ARTIFACT_AUDIT_MAX",
    "DEPTHSPEC_VERIFY_PROFILE_SKIP",
    "DEPTHSPEC_VERIFY_PROFILE_ACTIVE",
    "DEPTHSPEC_PREFIX_ARTIFACT_POSITION_OFFSET",
    "DEPTHSPEC_TARGET_PREFIX_DUMP_MAX",
    "DEPTHSPEC_PREFIX_HANDOFF_AUDIT_MAX",
    "DEPTHSPEC_PREFIX_SHIFT_POSITION_OFFSET",
}


@dataclass(frozen=True)
class DepthSpecDiagnostics:
    """Cold-path diagnostic options loaded from one JSON file."""

    values: dict[str, Any] = field(default_factory=dict)
    legacy_environment: tuple[str, ...] = ()

    @classmethod
    def load(cls, path: str | None) -> "DepthSpecDiagnostics":
        if path is None:
            payload = {}
        else:
            config_path = Path(path)
            payload = json.loads(config_path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError(
                    "DepthSpec diagnostics config must contain a JSON object, "
                    f"got {type(payload).__name__}"
                )
        values = dict(payload)
        legacy_environment = []
        for env_name, option_name in _LEGACY_DIAGNOSTIC_ENV_MAP.items():
            raw_value = os.environ.get(env_name)
            if raw_value is None:
                continue
            legacy_environment.append(env_name)
            if option_name in values:
                continue
            if env_name in _LEGACY_DIAGNOSTIC_BOOL_ENV_NAMES:
                values[option_name] = raw_value == "1"
            elif env_name in _LEGACY_DIAGNOSTIC_INT_ENV_NAMES:
                values[option_name] = int(raw_value)
            else:
                values[option_name] = raw_value
        return cls(
            values=values,
            legacy_environment=tuple(legacy_environment),
        )

    def get(self, name: str, default: Any = None) -> Any:
        return self.values.get(name, default)


@dataclass(frozen=True)
class DepthSpecRuntimeConfig:
    """A frozen execution policy shared by all DepthSpec runtime modules."""

    requested_profile: DepthSpecProfile
    effective_profile: str
    architecture: str | None
    verified_shape: bool
    unified_runner: bool
    draft_active_path_extend: bool
    draft_active_path_fixed: bool
    draft_active_path_graph: bool
    target_aligned_layout: bool
    target_verify_draft_extend: bool
    target_compatible_piecewise_graph: bool
    target_continuation_piecewise_graph: bool
    canonical_stem_piecewise_graph: bool
    canonical_full_stem_piecewise_graph: bool
    canonical_graph_max_batch_size: int
    canonical_proposal_replacement: bool
    canonical_proposal_graph: bool
    canonical_stem: bool
    canonical_partial_stem: bool
    loop_segment_executor: bool
    full_tree_missing_rows: bool
    proposal_stem_artifacts: bool
    use_partial_stem: bool
    continuation_mode: Literal["normal", "split"]
    prefix_shift_audit: bool
    draft_relay_oracle_mode: str | None
    target_verify_graph_mode: TargetVerifyGraphMode
    diagnostics: DepthSpecDiagnostics = field(default_factory=DepthSpecDiagnostics)
    legacy_environment: tuple[str, ...] = ()

    @classmethod
    def resolve(
        cls,
        server_args: Any,
        architecture: str | None,
    ) -> "DepthSpecRuntimeConfig":
        requested_profile = getattr(server_args, "depthspec_runtime_profile", "auto")
        if requested_profile not in {
            "auto",
            "legacy",
            "canonical-faas",
            "diagnostic",
        }:
            raise ValueError(
                f"Unsupported DepthSpec runtime profile: {requested_profile!r}"
            )

        diagnostics = DepthSpecDiagnostics.load(
            getattr(server_args, "depthspec_diagnostics_config", None)
        )
        verified_shape = (
            str(getattr(server_args, "speculative_algorithm", "")).upper()
            == "STANDALONE"
            and architecture == _LOOPFORMER_ARCHITECTURE
            and getattr(server_args, "speculative_num_steps", None) == 2
            and getattr(server_args, "speculative_eagle_topk", None) == 3
            and getattr(server_args, "speculative_num_draft_tokens", None) == 6
        )
        behavior_environment = tuple(
            name for name in _LEGACY_BEHAVIOR_ENV_NAMES if name in os.environ
        )
        legacy_environment = behavior_environment + diagnostics.legacy_environment
        if requested_profile == "auto" and behavior_environment:
            config = cls._from_legacy_environment(
                server_args=server_args,
                architecture=architecture,
                verified_shape=verified_shape,
                diagnostics=diagnostics,
                legacy_environment=legacy_environment,
            )
            logger.warning(
                "DepthSpec translated deprecated environment switches at startup: %s. "
                "Use --depthspec-runtime-profile instead.",
                ", ".join(legacy_environment),
            )
            return config
        if diagnostics.legacy_environment:
            logger.warning(
                "DepthSpec translated deprecated diagnostic environment switches "
                "at startup: %s. Use --depthspec-diagnostics-config instead.",
                ", ".join(diagnostics.legacy_environment),
            )

        if requested_profile == "legacy" or (
            requested_profile == "auto" and not verified_shape
        ):
            return cls._legacy(
                requested_profile=requested_profile,
                architecture=architecture,
                verified_shape=verified_shape,
                diagnostics=diagnostics,
                legacy_environment=legacy_environment,
            )

        if requested_profile == "canonical-faas":
            if not verified_shape:
                raise ValueError(
                    "canonical-faas currently requires LoopFormer STANDALONE "
                    "with topk=3, steps=2, and num_draft_tokens=6"
                )
            use_partial_stem = bool(
                diagnostics.get(
                    "canonical_partial_stem",
                    getattr(server_args, "max_running_requests", None) == 1,
                )
            )
            full_stem_piecewise_graph = bool(
                diagnostics.get(
                    "canonical_full_stem_piecewise_graph",
                    not use_partial_stem,
                )
            )
            canonical_graph_max_batch_size = min(
                int(getattr(server_args, "max_running_requests", 1) or 1),
                int(diagnostics.get("canonical_graph_max_batch_size", 16)),
            )
            return cls(
                requested_profile=requested_profile,
                effective_profile="canonical-faas",
                architecture=architecture,
                verified_shape=True,
                unified_runner=True,
                draft_active_path_extend=True,
                draft_active_path_fixed=True,
                draft_active_path_graph=False,
                target_aligned_layout=True,
                target_verify_draft_extend=True,
                target_compatible_piecewise_graph=bool(
                    diagnostics.get("target_compatible_piecewise_graph", True)
                ),
                target_continuation_piecewise_graph=bool(
                    diagnostics.get("target_continuation_piecewise_graph", True)
                ),
                canonical_stem_piecewise_graph=bool(
                    diagnostics.get("canonical_stem_piecewise_graph", True)
                ),
                canonical_full_stem_piecewise_graph=full_stem_piecewise_graph,
                canonical_graph_max_batch_size=canonical_graph_max_batch_size,
                canonical_proposal_replacement=True,
                canonical_proposal_graph=bool(
                    diagnostics.get("canonical_proposal_graph", True)
                ),
                canonical_stem=True,
                canonical_partial_stem=use_partial_stem,
                loop_segment_executor=True,
                full_tree_missing_rows=use_partial_stem,
                proposal_stem_artifacts=False,
                use_partial_stem=False,
                continuation_mode="normal",
                prefix_shift_audit=False,
                draft_relay_oracle_mode=None,
                target_verify_graph_mode="disabled",
                diagnostics=diagnostics,
                legacy_environment=legacy_environment,
            )

        # Both auto and diagnostic use the proven selective policy. Diagnostic
        # behavior itself is configured through one cold-path JSON object.
        active_path_graph = getattr(server_args, "max_running_requests", None) == 1
        return cls(
            requested_profile=requested_profile,
            effective_profile=(
                "selective-diagnostic"
                if requested_profile == "diagnostic"
                else "selective"
            ),
            architecture=architecture,
            verified_shape=True,
            unified_runner=True,
            draft_active_path_extend=True,
            draft_active_path_fixed=True,
            draft_active_path_graph=active_path_graph,
            target_aligned_layout=False,
            target_verify_draft_extend=False,
            target_compatible_piecewise_graph=False,
            target_continuation_piecewise_graph=False,
            canonical_stem_piecewise_graph=False,
            canonical_full_stem_piecewise_graph=False,
            canonical_graph_max_batch_size=0,
            canonical_proposal_replacement=False,
            canonical_proposal_graph=False,
            canonical_stem=False,
            canonical_partial_stem=False,
            loop_segment_executor=False,
            full_tree_missing_rows=False,
            proposal_stem_artifacts=False,
            use_partial_stem=False,
            continuation_mode="normal",
            prefix_shift_audit=False,
            draft_relay_oracle_mode=None,
            target_verify_graph_mode="disabled",
            diagnostics=diagnostics,
            legacy_environment=legacy_environment,
        )

    @classmethod
    def _legacy(
        cls,
        *,
        requested_profile: DepthSpecProfile,
        architecture: str | None,
        verified_shape: bool,
        diagnostics: DepthSpecDiagnostics,
        legacy_environment: tuple[str, ...] = (),
    ) -> "DepthSpecRuntimeConfig":
        return cls(
            requested_profile=requested_profile,
            effective_profile="legacy",
            architecture=architecture,
            verified_shape=verified_shape,
            unified_runner=False,
            draft_active_path_extend=False,
            draft_active_path_fixed=False,
            draft_active_path_graph=False,
            target_aligned_layout=False,
            target_verify_draft_extend=False,
            target_compatible_piecewise_graph=False,
            target_continuation_piecewise_graph=False,
            canonical_stem_piecewise_graph=False,
            canonical_full_stem_piecewise_graph=False,
            canonical_graph_max_batch_size=0,
            canonical_proposal_replacement=False,
            canonical_proposal_graph=False,
            canonical_stem=False,
            canonical_partial_stem=False,
            loop_segment_executor=False,
            full_tree_missing_rows=False,
            proposal_stem_artifacts=False,
            use_partial_stem=False,
            continuation_mode="normal",
            prefix_shift_audit=False,
            draft_relay_oracle_mode=None,
            target_verify_graph_mode="auto",
            diagnostics=diagnostics,
            legacy_environment=legacy_environment,
        )

    @classmethod
    def _from_legacy_environment(
        cls,
        *,
        server_args: Any,
        architecture: str | None,
        verified_shape: bool,
        diagnostics: DepthSpecDiagnostics,
        legacy_environment: tuple[str, ...],
    ) -> "DepthSpecRuntimeConfig":
        def enabled(name: str) -> bool:
            return os.environ.get(name) == "1"

        if enabled("DEPTHSPEC_DISABLE_TARGET_VERIFY_GRAPH"):
            graph_mode: TargetVerifyGraphMode = "disabled"
        elif enabled("DEPTHSPEC_ENABLE_TARGET_VERIFY_GRAPH"):
            graph_mode = "enabled"
        else:
            graph_mode = "auto"
        continuation_mode = os.environ.get("DEPTHSPEC_CONTINUATION_MODE", "normal")
        if continuation_mode not in {"normal", "split"}:
            raise ValueError(
                f"Invalid DEPTHSPEC_CONTINUATION_MODE={continuation_mode!r}"
            )
        config = cls(
            requested_profile="auto",
            effective_profile="legacy-environment",
            architecture=architecture,
            verified_shape=verified_shape,
            unified_runner=enabled("DEPTHSPEC_UNIFIED_RUNNER"),
            draft_active_path_extend=enabled(
                "DEPTHSPEC_DRAFT_ACTIVE_PATH_EXTEND"
            ),
            draft_active_path_fixed=enabled("DEPTHSPEC_DRAFT_ACTIVE_PATH_FIXED"),
            draft_active_path_graph=enabled("DEPTHSPEC_DRAFT_ACTIVE_PATH_GRAPH"),
            target_aligned_layout=enabled(
                "DEPTHSPEC_STANDALONE_TARGET_ALIGNED_LAYOUT"
            ),
            target_verify_draft_extend=enabled(
                "DEPTHSPEC_STANDALONE_TARGET_VERIFY_DRAFT_EXTEND"
            ),
            target_compatible_piecewise_graph=False,
            target_continuation_piecewise_graph=False,
            canonical_stem_piecewise_graph=False,
            canonical_full_stem_piecewise_graph=False,
            canonical_graph_max_batch_size=0,
            canonical_proposal_replacement=enabled(
                "DEPTHSPEC_CANONICAL_PROPOSAL_REPLACEMENT"
            ),
            canonical_proposal_graph=False,
            canonical_stem=enabled("DEPTHSPEC_CANONICAL_STEM_ORACLE"),
            canonical_partial_stem=(
                enabled("DEPTHSPEC_PARTIAL_STEM_AUDIT")
                and enabled("DEPTHSPEC_PARTIAL_STEM_ONLY")
            ),
            loop_segment_executor=enabled("DEPTHSPEC_LOOP_SEGMENT_EXECUTOR"),
            full_tree_missing_rows=enabled("DEPTHSPEC_FULL_TREE_MISSING_ROWS"),
            proposal_stem_artifacts=enabled("DEPTHSPEC_PROPOSAL_STEM_ARTIFACTS"),
            use_partial_stem=enabled("DEPTHSPEC_USE_PARTIAL_STEM"),
            continuation_mode=continuation_mode,
            prefix_shift_audit=enabled("DEPTHSPEC_PREFIX_SHIFT_AUDIT"),
            draft_relay_oracle_mode=os.environ.get(
                "DEPTHSPEC_DRAFT_RELAY_ORACLE_MODE"
            ),
            target_verify_graph_mode=graph_mode,
            diagnostics=diagnostics,
            legacy_environment=legacy_environment,
        )
        if config.unified_runner and architecture != _LOOPFORMER_ARCHITECTURE:
            raise ValueError(
                "DEPTHSPEC_UNIFIED_RUNNER is only valid for the LoopFormer adapter"
            )
        if config.canonical_proposal_replacement and (
            not verified_shape or getattr(server_args, "max_running_requests", None) != 1
        ):
            raise ValueError(
                "Legacy canonical proposal switches require the validated "
                "LoopFormer topk=3, steps=2, n6, max-running-requests=1 shape"
            )
        return config

    def to_log_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["diagnostics"] = sorted(self.diagnostics.values)
        return payload


_LEGACY_FALLBACK = DepthSpecRuntimeConfig._legacy(
    requested_profile="legacy",
    architecture=None,
    verified_shape=False,
    diagnostics=DepthSpecDiagnostics(),
)
_ACTIVE_RUNTIME_CONFIG = _LEGACY_FALLBACK


def activate_depthspec_runtime_config(config: DepthSpecRuntimeConfig) -> None:
    """Publish one immutable policy for helpers without a direct worker handle."""

    global _ACTIVE_RUNTIME_CONFIG
    _ACTIVE_RUNTIME_CONFIG = config


def get_active_depthspec_runtime_config() -> DepthSpecRuntimeConfig:
    """Return the process-local DepthSpec policy resolved during worker startup."""

    return _ACTIVE_RUNTIME_CONFIG


def attach_depthspec_runtime_config(owner: Any, config: DepthSpecRuntimeConfig) -> None:
    """Attach the resolved policy to a runtime object without changing its type."""

    setattr(owner, "depthspec_runtime_config", config)


def get_depthspec_runtime_config(owner: Any) -> DepthSpecRuntimeConfig:
    """Return a previously attached policy or the non-DepthSpec fallback."""

    config = getattr(owner, "depthspec_runtime_config", None)
    if config is not None:
        return config
    model_runner = getattr(owner, "model_runner", None)
    config = getattr(model_runner, "depthspec_runtime_config", None)
    if config is not None:
        return config
    server_args = getattr(owner, "server_args", None)
    config = getattr(server_args, "depthspec_runtime_config", None)
    if config is not None:
        return config
    return _LEGACY_FALLBACK
