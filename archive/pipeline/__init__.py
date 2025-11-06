"""Modular pipeline package for GCaMP analysis."""

__all__ = ["run_pipeline"]


def run_pipeline(*args, **kwargs):
	from .main import run_pipeline as _run_pipeline

	return _run_pipeline(*args, **kwargs)
