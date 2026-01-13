"""
Utilities for CASCADE spike inference model.
"""

from __future__ import annotations
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, TYPE_CHECKING
import logging

if TYPE_CHECKING:
    from data_classes.video import Video

logger = logging.getLogger(__name__)


def load_cascade_model(model_name: str = "GC8_EXC_30Hz_smoothing25ms_high_noise", 
                       model_dir: str = "C:/Users/mzinn1/Desktop/Scripts/GCaMP-analysis/Pretrained_models") -> Any:
    """
    Load a pretrained CASCADE model.
    
    Args:
        model_name: Name of the pretrained model
        model_dir: Directory containing CASCADE models
        
    Returns:
        Loaded CASCADE model (CascadePredictor instance)
    """
    try:
        # Try to import cascade2p
        import sys
        cascade_path = Path(__file__).parent.parent / "Cascade"
        if str(cascade_path) not in sys.path:
            sys.path.insert(0, str(cascade_path))
            
        from cascade2p.cascade_wrapper import CascadePredictor
        
        # Load the model using CascadePredictor
        model = CascadePredictor(model_name=model_name, model_folder=model_dir)
        logger.info(f"Loaded CASCADE model: {model_name}")
        return model
        
    except ImportError as e:
        logger.error(f"Failed to import cascade2p: {e}")
        logger.error("Make sure CASCADE is properly installed")
        raise
    except Exception as e:
        logger.error(f"Failed to load CASCADE model: {e}")
        raise


class CascadeWrapper:
    """
    Wrapper for CASCADE spike inference model.
    
    Provides batch prediction and caching functionality.
    """
    
    def __init__(self, model_name: str, model_dir: str, batch_size: int = 64):
        """
        Initialize CASCADE wrapper.
        
        Args:
            model_name: Name of the pretrained model
            model_dir: Directory containing CASCADE models
            batch_size: Batch size for prediction
        """
        self.model_name = model_name
        self.model_dir = model_dir
        self.batch_size = batch_size
        self.model = None
        
    def load(self):
        """Load the CASCADE model."""
        if self.model is None:
            self.model = load_cascade_model(self.model_name, self.model_dir)
            
    def predict(self, traces: np.ndarray, frame_rate: float) -> np.ndarray:
        """
        Predict spike probabilities from fluorescence traces.
        
        Args:
            traces: Fluorescence traces (n_neurons, n_frames)
            frame_rate: Recording frame rate in Hz
            
        Returns:
            Spike probability traces (n_neurons, n_frames)
        """
        if self.model is None:
            self.load()
            
        # CASCADE expects 2D input (neurons, frames)
        if traces.ndim == 1:
            traces = traces.reshape(1, -1)
            
        # Run prediction
        try:
            from cascade2p import checks
            spike_probs = checks.predict(
                self.model,
                traces,
                frame_rate,
                batch_size=self.batch_size
            )
            return spike_probs
            
        except Exception as e:
            logger.error(f"CASCADE prediction failed: {e}")
            raise


def batch_predict_cascade(
    traces: np.ndarray,
    frame_rate: float,
    model: CascadeWrapper,
    batch_size: int = 64
) -> np.ndarray:
    """
    Run CASCADE prediction in batches.
    
    Args:
        traces: Fluorescence traces (n_neurons, n_frames)
        frame_rate: Recording frame rate in Hz
        model: CASCADE model wrapper
        batch_size: Number of neurons per batch
        
    Returns:
        Spike probability traces (n_neurons, n_frames)
    """
    n_neurons = traces.shape[0]
    n_frames = traces.shape[1]
    spike_probs = np.zeros_like(traces)
    
    # Process in batches
    for i in range(0, n_neurons, batch_size):
        end_idx = min(i + batch_size, n_neurons)
        batch_traces = traces[i:end_idx]
        
        logger.debug(f"Processing CASCADE batch {i//batch_size + 1}/{(n_neurons + batch_size - 1)//batch_size}")
        spike_probs[i:end_idx] = model.predict(batch_traces, frame_rate)
        
    return spike_probs


def load_cascade_predictions(video: Video, cache_dir: Optional[Path] = None) -> np.ndarray:
    """
    Load cached CASCADE predictions or compute them.
    
    Args:
        video: Video object with fluorescence traces
        cache_dir: Directory to cache predictions (optional)
        
    Returns:
        Spike probability traces (n_neurons, n_frames)
    """
    # Check cache if directory provided
    if cache_dir is not None:
        cache_file = cache_dir / f"{video.name}_cascade.npy"
        if cache_file.exists():
            logger.info(f"Loading cached CASCADE predictions from {cache_file}")
            return np.load(cache_file)
    
    # If no cache or cache miss, would need to compute
    # But this requires the model, so just raise error
    raise FileNotFoundError(
        f"No cached CASCADE predictions found for {video.name}. "
        "Please run CASCADE prediction first."
    )
