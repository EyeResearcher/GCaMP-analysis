"""Valley class for negative events."""
import numpy as np

class Valley:
    """Represents a valley (negative deflection) in fluorescence."""
    
    def __init__(self,
                 frame_index: int,
                 depth: float,
                 width: int):
        """
        Initialize Valley.
        
        Parameters:
            frame_index: Frame index of valley minimum
            depth: Depth of valley (negative value)
            width: Width in frames
        """
        self.frame_index = frame_index
        self.depth = depth
        self.width = width
        
    def __repr__(self):
        return f"Valley(frame={self.frame_index}, depth={self.depth:.2f}, width={self.width})"