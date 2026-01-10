"""
Base class for radar post-processing components.
"""
from typing import Dict, Any, Optional
from abc import abstractmethod
import torch

from witwin.components import Component, component, bool_field
from witwin.utils.logging import get_logger

logger = get_logger("RadarPostProcessor")


@component(name="RadarPostProcessor", category="Radar/PostProcessing", abstract=True)
class RadarPostProcessor(Component):
    """
    Base class for radar post-processing components.

    This component receives raw radar signals and applies processing algorithms.
    Subclasses should override the `process()` method to implement specific
    processing algorithms (e.g., Range-Doppler, point cloud, etc.).

    Post-processors are passive - they are triggered by the Radar component
    when rendering completes, not manually by the user.
    """

    # Processing control
    enabled = bool_field(True, description="Enable this post-processing")

    # Internal storage for raw signal data
    _raw_signal: Optional[torch.Tensor] = None
    _signal_metadata: Dict[str, Any] = {}

    def __init__(self, **kwargs):
        """Initialize the post-processing component."""
        super().__init__(**kwargs)
        self._raw_signal = None
        self._signal_metadata = {}

    def set_raw_signal(
        self,
        signal: torch.Tensor,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Set the raw signal data for processing.

        Called by the Radar component after rendering.

        Args:
            signal: Raw radar signal tensor. Shape depends on radar mode:
                    - Non-MIMO: [adc_samples] complex tensor
                    - MIMO: [num_tx, num_rx, adc_samples] complex tensor
            metadata: Metadata dict containing radar parameters
        """
        logger.info(f"set_raw_signal called: shape={signal.shape}, enabled={self.enabled}")
        self._raw_signal = signal
        self._signal_metadata = metadata or {}

        if self.enabled:
            logger.info("Calling _run_process...")
            self._run_process()
        else:
            logger.info("Processing disabled, skipping")

    def get_raw_signal(self) -> Optional[torch.Tensor]:
        """Get the current raw signal tensor."""
        return self._raw_signal

    def get_metadata(self) -> Dict[str, Any]:
        """Get the signal metadata dictionary."""
        return self._signal_metadata

    @abstractmethod
    def process(self, signal: torch.Tensor, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the raw radar signal.

        Subclasses must implement this method to perform specific processing.

        Args:
            signal: Raw radar signal tensor
            params: Radar parameters dictionary

        Returns:
            Dictionary containing processing results.
        """
        raise NotImplementedError("Subclasses must implement process()")

    def _run_process(self) -> str:
        """Execute the post-processing pipeline."""
        logger.info(f"_run_process called, enabled={self.enabled}, has_signal={self._raw_signal is not None}")

        if not self.enabled:
            return "Post-processing is disabled"

        if self._raw_signal is None:
            logger.warning("No raw signal available for processing")
            return "No raw signal available"

        params = self._signal_metadata

        try:
            logger.info("Calling process()...")
            result = self.process(self._raw_signal, params)
            logger.info(f"process() returned: {list(result.keys()) if result else None}")
            self._on_process_complete(result)
            logger.info("_on_process_complete done")
            return "Processing complete"
        except Exception as e:
            import traceback
            logger.error(f"Processing failed: {e}")
            logger.error(traceback.format_exc())
            return f"Processing failed: {e}"

    def _on_process_complete(self, result: Dict[str, Any]):
        """
        Called after processing is complete.

        Subclasses can override this to update visualizations.
        """
        pass

    def to_dict(self) -> Dict[str, Any]:
        """Convert component to dictionary for serialization."""
        return super().to_dict()

    def from_dict(self, data: Dict[str, Any]):
        """Load component from dictionary."""
        super().from_dict(data)
