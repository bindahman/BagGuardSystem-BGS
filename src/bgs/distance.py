from typing import List, Optional, Tuple

import numpy as np

from .config import BGSConfig


class DistanceEstimator:
    def __init__(self):
        self.config = BGSConfig

    @staticmethod
    def _bbox_center(bbox: List[float]) -> Tuple[float, float]:
        x1, y1, x2, y2 = bbox
        return (x1 + x2) / 2, (y1 + y2) / 2

    def estimate_depth(self, bbox_height_pixels: float, real_height_meters: float = None) -> float:
        if real_height_meters is None:
            real_height_meters = self.config.ASSUMED_PERSON_HEIGHT
        if bbox_height_pixels < 1:
            return 10.0
        return (real_height_meters * self.config.FOCAL_LENGTH) / bbox_height_pixels

    def estimate_position_3d(
        self,
        bbox: List[float],
        is_person: bool = False,
        reference_depth: Optional[float] = None,
    ) -> Tuple[float, float, float]:
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        bbox_height = y2 - y1

        if is_person and bbox_height > 50:
            depth = self.estimate_depth(bbox_height, self.config.ASSUMED_PERSON_HEIGHT)
        elif reference_depth is not None:
            depth = reference_depth
        else:
            depth = self.estimate_depth(bbox_height, self.config.ASSUMED_BAG_HEIGHT)

        image_center_x = self.config.IMAGE_WIDTH / 2
        image_center_y = self.config.IMAGE_HEIGHT / 2
        x_meters = ((center_x - image_center_x) * depth) / self.config.FOCAL_LENGTH
        y_meters = ((center_y - image_center_y) * depth) / self.config.FOCAL_LENGTH
        return x_meters, y_meters, depth

    def calculate_distance(
        self,
        pos1: Tuple[float, float, float],
        pos2: Tuple[float, float, float],
    ) -> float:
        x1, y1, z1 = pos1
        x2, y2, z2 = pos2
        return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)

    def calculate_ground_distance(
        self,
        pos1: Tuple[float, float, float],
        pos2: Tuple[float, float, float],
    ) -> float:
        x1, _, z1 = pos1
        x2, _, z2 = pos2
        return float(np.sqrt((x2 - x1) ** 2 + (z2 - z1) ** 2))

    def calculate_image_distance(self, bbox_a: List[float], bbox_b: List[float]) -> float:
        ax, ay = self._bbox_center(bbox_a)
        bx, by = self._bbox_center(bbox_b)
        return float(np.hypot(ax - bx, ay - by))