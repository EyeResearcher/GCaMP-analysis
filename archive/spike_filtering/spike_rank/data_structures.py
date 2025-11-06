from dataclasses import dataclass
@dataclass
class Peak:
    def __init__(self, index: int, value: float):
        """
        Args:
            index (int): index of the peak
            value (float): value of the peak
        """
        self.index = index
        self.value = value
        self.normalized_value = None  # will be assigned later
        self.rank_score = {}
        self.valley_score = {}
        self.rank_score_preonly = {}
@dataclass
class Valley:
    def __init__(self, index, value):
        """
        Args:
            index (int): index of the peak
            value (float): value of the peak
        """
        self.index = index
        self.value = value
        self.normalized_value = None
        self.ranks = {}
        self.depths = {}    
        self.normal_ranks = {}
        self.weight = None
        self.weight_preonly = None
        self.weighted_ranks = {}
        self.weighted_depths = {}
        self.previous_peak = None
        self.next_peak = None
        self.previous_auc = None
        self.next_auc = None
        self.previous_depth = None
        self.next_depth = None
        self.sum_auc = None
        self.average_auc = None
        self.sum_depth = None
        self.average_depth = None
        self.previous_depth_sharpness = None
        self.next_depth_sharpness = None
        self.sum_depth_sharpness = None
        self.average_depth_sharpness = None