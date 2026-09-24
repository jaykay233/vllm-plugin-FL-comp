# Copyright (c) 2026 BAAI. All rights reserved.

"""
Unit tests for ``vllm_fl.platform``'s ``VLLM_FL_CUDAGRAPH_ONLY`` switch.

Only ``PIECEWISE`` cudagraph modes need ``torch.compile``
(``CUDAGraphMode.requires_piecewise_compilation()``), so ``FULL`` and
``FULL_DECODE_ONLY`` can be replayed from an eager model. Dropping inductor also
restores ``custom_ops=['all']``, which is what brings the FlagGems out-of-tree
chain (``forward_oot -> CachedOp -> flag_gems``) back for every FL op.

These tests only exercise config mutation, so no accelerator work happens. The
import of ``vllm_fl.platform`` itself is guarded with ``importorskip`` because it
probes the local device at import time.
"""

from __future__ import annotations

import pytest

pytest.importorskip("vllm")

from vllm.config import CompilationMode, CUDAGraphMode  # noqa: E402


@pytest.fixture
def fl_platform():
    """Import vllm_fl.platform lazily so collection works on CPU-only hosts."""
    return pytest.importorskip("vllm_fl.platform")


class _FakeCompilationConfig:
    """Just enough of vllm.config.CompilationConfig for the helper under test."""

    def __init__(
        self,
        mode: CompilationMode,
        cudagraph_mode: CUDAGraphMode,
        custom_ops: list[str] | None = None,
        ir_enable_torch_wrap: bool = False,
    ) -> None:
        self.mode = mode
        self.cudagraph_mode = cudagraph_mode
        self.custom_ops = ["none"] if custom_ops is None else custom_ops
        self.ir_enable_torch_wrap = ir_enable_torch_wrap


class TestEnvParsing:
    """``_cudagraph_only_enabled`` accepts the usual truthy spellings."""

    @pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "On", " true "])
    def test_truthy(self, fl_platform, monkeypatch, value):
        monkeypatch.setenv(fl_platform.ENV_CUDAGRAPH_ONLY, value)
        assert fl_platform._cudagraph_only_enabled() is True

    @pytest.mark.parametrize("value", ["0", "false", "no", "off", "", "maybe"])
    def test_falsy(self, fl_platform, monkeypatch, value):
        monkeypatch.setenv(fl_platform.ENV_CUDAGRAPH_ONLY, value)
        assert fl_platform._cudagraph_only_enabled() is False

    def test_unset_is_off(self, fl_platform, monkeypatch):
        monkeypatch.delenv(fl_platform.ENV_CUDAGRAPH_ONLY, raising=False)
        assert fl_platform._cudagraph_only_enabled() is False


class TestApplyCudagraphOnly:
    """The rewrite itself."""

    def test_full_and_piecewise_keeps_full_half(self, fl_platform):
        config = _FakeCompilationConfig(
            CompilationMode.VLLM_COMPILE, CUDAGraphMode.FULL_AND_PIECEWISE
        )

        assert fl_platform._apply_cudagraph_only(config) is True
        assert config.mode == CompilationMode.NONE
        # Piecewise is the only half that demands compilation, so it is dropped
        # and mixed/prefill batches fall back to eager.
        assert config.cudagraph_mode == CUDAGraphMode.FULL_DECODE_ONLY

    @pytest.mark.parametrize(
        "cg_mode", [CUDAGraphMode.FULL, CUDAGraphMode.FULL_DECODE_ONLY]
    )
    def test_full_modes_are_preserved(self, fl_platform, cg_mode):
        config = _FakeCompilationConfig(CompilationMode.VLLM_COMPILE, cg_mode)

        assert fl_platform._apply_cudagraph_only(config) is True
        assert config.mode == CompilationMode.NONE
        assert config.cudagraph_mode == cg_mode

    def test_restores_oot_chain(self, fl_platform):
        """``custom_ops=['all']`` is what makes FlagGems reachable again."""
        config = _FakeCompilationConfig(
            CompilationMode.VLLM_COMPILE,
            CUDAGraphMode.FULL_AND_PIECEWISE,
            ir_enable_torch_wrap=True,
        )

        assert fl_platform._apply_cudagraph_only(config) is True
        assert config.custom_ops == ["all"]
        assert "none" not in config.custom_ops
        assert config.ir_enable_torch_wrap is False

    def test_keeps_explicit_op_overrides(self, fl_platform):
        """A user's "+op" / "-op" entries survive the rewrite."""
        config = _FakeCompilationConfig(
            CompilationMode.VLLM_COMPILE,
            CUDAGraphMode.FULL_DECODE_ONLY,
            custom_ops=["none", "+quant_fp8", "-rms_norm"],
        )

        assert fl_platform._apply_cudagraph_only(config) is True
        assert config.custom_ops == ["+quant_fp8", "-rms_norm", "all"]

    @pytest.mark.parametrize("cg_mode", [CUDAGraphMode.PIECEWISE, CUDAGraphMode.NONE])
    def test_unsupported_modes_are_left_alone(self, fl_platform, cg_mode):
        config = _FakeCompilationConfig(CompilationMode.VLLM_COMPILE, cg_mode)

        assert fl_platform._apply_cudagraph_only(config) is False
        assert config.mode == CompilationMode.VLLM_COMPILE
        assert config.cudagraph_mode == cg_mode
        assert config.custom_ops == ["none"]
