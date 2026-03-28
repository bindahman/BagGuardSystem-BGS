"""
═══════════════════════════════════════════════════════════════════════════════
BAG GUARD SYSTEM (BGS) - COMPLETE SPECIFICATION IMPLEMENTATION
═══════════════════════════════════════════════════════════════════════════════

FINAL YEAR PROJECT - COMPUTER SCIENCE
System Focus: Unattended Luggage Detection with Ownership Persistence

FULL IMPLEMENTATION OF BGS SPECIFICATION:
✅ Monocular Distance Estimation (Trigonometry-based)
✅ Ownership Persistence Logic (Trend-based, NOT frame-by-frame)
✅ Stable ID Tracking (ByteTrack optimized)
✅ Three-State Bag Status (OK, POTENTIAL, UNATTENDED)
✅ Ownership Locking & Confirmation
✅ Rolling Distance History & Smoothing
✅ Visual Distance Lines & Meter Labels
✅ Frozen Parameters (Reproducibility)
✅ Professional Debug Overlay
✅ Modular Architecture

Optimized for: yolo26x.pt model (stable person IDs)
Author: BGS Implementation Team
Version: 3.0 - Full Specification Compliant
Date: February 2026
═══════════════════════════════════════════════════════════════════════════════
"""

import argparse
import json
import pickle
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
import time
from collections import deque
from dataclasses import dataclass
from types import SimpleNamespace
from typing import List, Dict, Tuple, Optional, Deque
import sys
import os
import torch


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1: FROZEN CONFIGURATION (Section 9 - Parameter Management)
# ═══════════════════════════════════════════════════════════════════════════
class BGSConfig:
    """
    FROZEN SYSTEM PARAMETERS (Section 9 - Parameter Management)
    
    All parameters are fixed during runtime for:
    - Scientific reproducibility
    - Fair evaluation
    - Stable system behaviour
    
    Based on BGS Specification Requirements:
    - Section 4: Detection
    - Section 5: Tracking
    - Section 6: Distance Estimation
    - Section 7: Ownership Persistence
    - Section 8: Unattended Detection
    """
    
    # ═══════════════════════════════════════════════════════════════════════
    # SECTION 4.2: DETECTION REQUIREMENTS
    # ═══════════════════════════════════════════════════════════════════════
    TARGET_CLASSES = {
        0: 'person',
        24: 'backpack',
        26: 'handbag',
        28: 'suitcase'
    }
    CLASS_IDS = [0, 24, 26, 28]
    EXPECTED_CLASS_IDS = {0, 24, 26, 28}
    EXPECTED_TARGET_CLASSES = {
        0: 'person',
        24: 'backpack',
        26: 'handbag',
        28: 'suitcase'
    }
    PERSON_CLASS_ID = 0
    BAG_CLASS_IDS = [24, 26, 28]
    DETECTION_CONFIDENCE = 0.20  # Fixed confidence threshold
    PERSON_CONF = DETECTION_CONFIDENCE
    BAG_CONF = DETECTION_CONFIDENCE - 0.05
    IOU_THRESHOLD = 0.45
    
    # ═══════════════════════════════════════════════════════════════════════
    # SECTION 5: TRACKING REQUIREMENTS (CRITICAL)
    # ═══════════════════════════════════════════════════════════════════════
    TRACKER = "bytetrack.yaml"   # Section 5.1: ByteTrack default
    PERSIST = True
    # Frozen tracking parameters (Section 5.3)
    TRACK_BUFFER = 90
    TRACK_THRESH = 0.25
    MATCH_THRESH = 0.6
    BAG_MATCH_THRESHOLD_PX = 100  # pixels - spatial matching threshold
    BAG_FRAME_MEMORY = 30         # frames - how long to keep bag history
    
    # ═══════════════════════════════════════════════════════════════════════
    # SECTION 6.2: DISTANCE ESTIMATION (Monocular Trigonometry)
    # ═══════════════════════════════════════════════════════════════════════
    CAMERA_HFOV = 60.0              # Horizontal field of view (degrees)
    ASSUMED_PERSON_HEIGHT = 1.70    # meters (Section 6.2)
    ASSUMED_BAG_HEIGHT = 0.45       # meters (bag depth fallback)
    IMAGE_WIDTH = 1280              # pixels
    IMAGE_HEIGHT = 720              # pixels
    
    # Calculate focal length from HFOV
    FOCAL_LENGTH = (IMAGE_WIDTH / 2) / np.tan(np.radians(CAMERA_HFOV / 2))
    
    # ═══════════════════════════════════════════════════════════════════════
    # SECTION 7: OWNERSHIP PERSISTENCE LOGIC
    # ═══════════════════════════════════════════════════════════════════════
    DISTANCE_HISTORY_SIZE = 30
    ASSIGNMENT_DISTANCE = 2.5
    CONFIRMATION_TIME = 2.0
    OWNERSHIP_LOCK_TIME = 10.0
    SWITCH_DISTANCE_IMPROVEMENT = 1.5
    
    # ═══════════════════════════════════════════════════════════════════════
    # SECTION 8: UNATTENDED BAG LOGIC
    # ═══════════════════════════════════════════════════════════════════════
    POTENTIAL_THRESHOLD = 5.0
    UNATTENDED_THRESHOLD = 10.0
    OWNERSHIP_RELEASE_GRACE = 5.0

    # Two-layer identity persistence (registry)
    REID_MAX_AGE_FRAMES = 900
    REID_MATCH_MAX_AGE_FRAMES = 240
    REID_APPEARANCE_MAX_AGE_FRAMES = 600
    REID_CENTROID_THRESH = 80
    REID_BAG_CENTROID_THRESH = 60
    REID_IOU_THRESH = 0.25
    REID_PERSON_MODEL_NAME = "osnet_x1_0"
    REID_PERSON_MODEL_PATH = "models/reid/osnet_x1_0_msmt17.pt"
    REID_PERSON_EMBED_THRESHOLD = 0.58
    REID_PERSON_INPUT_SIZE = (256, 128)
    REID_PERSON_MIN_CROP_SIZE = 32
    REID_PERSON_SIZE_RATIO_MIN = 0.40
    REID_PERSON_ASPECT_RATIO_MAX_DIFF = 0.55
    REID_APPEARANCE_UPDATE_WEIGHT = 0.20
    REID_PERSON_GALLERY_SIZE = 12
    REID_PERSON_APPEARANCE_PRIORITY = 0.04
    REID_BAG_APPEARANCE_MAX_AGE_FRAMES = 600
    REID_BAG_EMBED_THRESHOLD = 0.72
    REID_BAG_MIN_CROP_SIZE = 20
    REID_BAG_SIZE_RATIO_MIN = 0.45
    REID_BAG_ASPECT_RATIO_MAX_DIFF = 0.90
    REID_BAG_GALLERY_SIZE = 8
    REID_BAG_OWNER_MATCH_BONUS = 0.08
    REID_BAG_APPEARANCE_PRIORITY = 0.05
    REID_PERSIST_PATH = "logs/person_reid/person_registry.pkl"
    REID_PERSIST_LOG_DIR = "logs/person_reid"
    REID_PERSIST_INTERVAL_FRAMES = 30
    REID_PERSIST_MAX_AGE_SECONDS = 43200.0
    REID_FRAME_IMAGE_DIRNAME = "frames"
    REID_FRAME_METADATA_NAME = "frames.jsonl"
    REID_FRAME_IMAGE_EXT = ".jpg"
    
    # ═══════════════════════════════════════════════════════════════════════
    # SECTION 10: VISUALIZATION
    # ═══════════════════════════════════════════════════════════════════════
    COLOR_PERSON = (50, 205, 50)
    COLOR_BAG_OK = (255, 165, 0)
    COLOR_BAG_POTENTIAL = (0, 165, 255)
    COLOR_BAG_UNATTENDED = (0, 0, 255)
    COLOR_DISTANCE_LINE = (255, 255, 0)
    COLOR_TEXT = (255, 255, 255)
    COLOR_BG = (30, 30, 30)
    COLOR_HIGHLIGHT = (0, 255, 255)
    SHOW_DISTANCE_LINES = True
    SHOW_DISTANCE_LABELS = True
    SHOW_DEBUG_OVERLAY = True
    LINE_THICKNESS = 3
    FONT_SCALE = 0.7
    FONT = cv2.FONT_HERSHEY_SIMPLEX


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2: DISTANCE ESTIMATOR (Section 6.2 - Monocular Trigonometry)
# ═══════════════════════════════════════════════════════════════════════════
class DistanceEstimator:
    def __init__(self):
        self.config = BGSConfig

    def estimate_depth(self, bbox_height_pixels: float, real_height_meters: float = None) -> float:
        if real_height_meters is None:
            real_height_meters = self.config.ASSUMED_PERSON_HEIGHT
        if bbox_height_pixels < 1:
            return 10.0
        return (real_height_meters * self.config.FOCAL_LENGTH) / bbox_height_pixels

    def estimate_position_3d(self, bbox: List[float],
                            is_person: bool = False,
                            reference_depth: Optional[float] = None) -> Tuple[float, float, float]:
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

    def calculate_distance(self, pos1: Tuple[float, float, float],
                          pos2: Tuple[float, float, float]) -> float:
        x1, y1, z1 = pos1
        x2, y2, z2 = pos2
        return np.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)


class PersonReIDEmbedder:
    """OSNet-based feature extractor for person re-identification."""

    def __init__(self, device: str, model_name: str, model_path: str = ""):
        self.config = BGSConfig
        self.device = "cuda" if str(device).startswith("cuda") else "cpu"
        self.model_name = model_name
        self.model_path = model_path
        self.extractor = None

        try:
            from torchreid.reid.utils.feature_extractor import FeatureExtractor

            self.extractor = FeatureExtractor(
                model_name=self.model_name,
                model_path=self.model_path,
                image_size=self.config.REID_PERSON_INPUT_SIZE,
                device=self.device,
                verbose=False,
            )
            source = self.model_path if self.model_path else f"pretrained {self.model_name}"
            print(f"✓ Person re-ID extractor loaded: {source}")
        except Exception as e:
            print(f"⚠️  Person re-ID disabled: {e}")

    @property
    def enabled(self) -> bool:
        return self.extractor is not None

    def crop_person(self, frame: np.ndarray, bbox: List[float]) -> Optional[np.ndarray]:
        frame_h, frame_w = frame.shape[:2]
        x1 = max(0, min(frame_w - 1, int(bbox[0])))
        y1 = max(0, min(frame_h - 1, int(bbox[1])))
        x2 = max(0, min(frame_w, int(bbox[2])))
        y2 = max(0, min(frame_h, int(bbox[3])))

        if x2 <= x1 or y2 <= y1:
            return None

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        crop_h, crop_w = crop.shape[:2]
        if crop_h < self.config.REID_PERSON_MIN_CROP_SIZE or crop_w < self.config.REID_PERSON_MIN_CROP_SIZE:
            return None

        inset_x = max(1, int(crop_w * 0.18))
        top_trim = max(1, int(crop_h * 0.10))
        bottom_trim = max(1, int(crop_h * 0.05))
        center = crop[top_trim:crop_h - bottom_trim, inset_x:crop_w - inset_x]
        if center.size != 0:
            crop = center

        return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

    def extract(self, crops: List[np.ndarray]) -> List[Optional[np.ndarray]]:
        if not crops or not self.enabled:
            return [None] * len(crops)
        try:
            features = self.extractor(crops)
        except Exception as e:
            print(f"⚠️  Person re-ID inference failed: {e}")
            return [None] * len(crops)

        embeddings = []
        for feature in features:
            vector = feature.detach().cpu().numpy().astype(np.float32)
            norm = float(np.linalg.norm(vector))
            embeddings.append(vector / norm if norm > 0 else None)
        return embeddings


class BagReIDEmbedder:
    """Lightweight handcrafted appearance extractor for bag re-identification."""

    def __init__(self):
        self.config = BGSConfig

    @property
    def enabled(self) -> bool:
        return True

    def crop_bag(self, frame: np.ndarray, bbox: List[float]) -> Optional[np.ndarray]:
        frame_h, frame_w = frame.shape[:2]
        x1 = max(0, min(frame_w - 1, int(bbox[0])))
        y1 = max(0, min(frame_h - 1, int(bbox[1])))
        x2 = max(0, min(frame_w, int(bbox[2])))
        y2 = max(0, min(frame_h, int(bbox[3])))

        if x2 <= x1 or y2 <= y1:
            return None

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        crop_h, crop_w = crop.shape[:2]
        if crop_h < self.config.REID_BAG_MIN_CROP_SIZE or crop_w < self.config.REID_BAG_MIN_CROP_SIZE:
            return None

        return crop

    def extract_one(self, crop: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if crop is None or crop.size == 0:
            return None

        resized = cv2.resize(crop, (64, 64), interpolation=cv2.INTER_LINEAR)
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        hsv_hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256]).flatten()
        gray_hist = cv2.calcHist([gray], [0], None, [16], [0, 256]).flatten()
        edges = cv2.Canny(gray, 80, 160)
        edge_density = np.asarray([float(np.count_nonzero(edges)) / float(edges.size)], dtype=np.float32)
        aspect_ratio = np.asarray([float(crop.shape[1]) / max(1.0, float(crop.shape[0]))], dtype=np.float32)

        vector = np.concatenate([
            hsv_hist.astype(np.float32),
            gray_hist.astype(np.float32),
            edge_density,
            aspect_ratio,
        ])
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm > 0 else None


class BGSRegistry:
    """Stable-ID registry for person and bag tracks."""

    PERSON_BASE_ID = 1_000_000
    BAG_BASE_ID = 2_000_000

    def __init__(self, persist_path: Optional[Path] = None, persist_log_dir: Optional[Path] = None):
        self.config = BGSConfig
        self.entries: Dict[int, Dict] = {}
        self.next_person_id = 1
        self.next_bag_id = 1
        self.used_ids_by_frame: Dict[int, set] = {}
        self.persist_path = Path(persist_path) if persist_path is not None else None
        self.persist_log_dir = Path(persist_log_dir) if persist_log_dir is not None else None
        self.loaded_person_count = 0
        self.loaded_bag_count = 0
        self.load_persistent_entries()

    def resolve(self, bbox: List[float], class_id: int, frame_number: int,
                bt_id: Optional[int] = None,
                appearance: Optional[np.ndarray] = None,
                owner_hint: Optional[int] = None) -> Tuple[int, Dict[str, float | str]]:
        self._expire(frame_number)
        used_ids = self.used_ids_by_frame.setdefault(frame_number, set())
        centroid_thresh = (
            self.config.REID_CENTROID_THRESH
            if class_id == self.config.PERSON_CLASS_ID
            else self.config.REID_BAG_CENTROID_THRESH
        )

        if bt_id is not None:
            for stable_id, entry in self.entries.items():
                if stable_id in used_ids:
                    continue
                if entry['class_id'] == class_id and bt_id in entry['bt_ids']:
                    self._update(stable_id, bbox, frame_number, bt_id, appearance)
                    used_ids.add(stable_id)
                    return stable_id, {'reason': 'track', 'score': 1.0}

        if class_id == self.config.PERSON_CLASS_ID and appearance is not None:
            appearance_id, appearance_score = self._match_person_by_appearance(
                bbox, appearance, frame_number, used_ids
            )
            if appearance_id is not None:
                geometry_id, geometry_meta = self._match_by_geometry(
                    bbox, class_id, frame_number, used_ids, centroid_thresh
                )
                if geometry_id is not None and geometry_id != appearance_id:
                    geometry_iou = float(geometry_meta.get('iou', 0.0))
                    appearance_margin = appearance_score - self.config.REID_PERSON_EMBED_THRESHOLD
                    if geometry_iou >= self.config.REID_IOU_THRESH and appearance_margin < self.config.REID_PERSON_APPEARANCE_PRIORITY:
                        self._update(geometry_id, bbox, frame_number, bt_id, appearance)
                        used_ids.add(geometry_id)
                        return geometry_id, geometry_meta

                self._update(appearance_id, bbox, frame_number, bt_id, appearance)
                used_ids.add(appearance_id)
                return appearance_id, {'reason': 'reid', 'score': appearance_score}

        if class_id in self.config.BAG_CLASS_IDS and appearance is not None:
            appearance_id, appearance_score = self._match_bag_by_appearance(
                bbox, class_id, appearance, frame_number, used_ids, owner_hint
            )
            if appearance_id is not None:
                geometry_id, geometry_meta = self._match_by_geometry(
                    bbox, class_id, frame_number, used_ids, centroid_thresh
                )
                if geometry_id is not None and geometry_id != appearance_id:
                    geometry_iou = float(geometry_meta.get('iou', 0.0))
                    appearance_margin = appearance_score - self.config.REID_BAG_EMBED_THRESHOLD
                    if geometry_iou >= self.config.REID_IOU_THRESH and appearance_margin < self.config.REID_BAG_APPEARANCE_PRIORITY:
                        self._update(geometry_id, bbox, frame_number, bt_id, appearance)
                        used_ids.add(geometry_id)
                        return geometry_id, geometry_meta

                self._update(appearance_id, bbox, frame_number, bt_id, appearance)
                if owner_hint is not None:
                    self.entries[appearance_id]['owner_id'] = owner_hint
                used_ids.add(appearance_id)
                return appearance_id, {'reason': 'bag-reid', 'score': appearance_score}

        best_id, geometry_meta = self._match_by_geometry(
            bbox, class_id, frame_number, used_ids, centroid_thresh
        )
        if best_id is not None:
            self._update(best_id, bbox, frame_number, bt_id, appearance)
            used_ids.add(best_id)
            return best_id, geometry_meta

        new_id = self._new_id(class_id)
        self.entries[new_id] = {
            'bbox': bbox,
            'class_id': class_id,
            'last_frame': frame_number,
            'last_seen_ts': time.time(),
            'bt_ids': {bt_id} if bt_id is not None else set(),
            'appearance': appearance.copy() if appearance is not None else None,
            'appearance_gallery': deque(
                [appearance.copy()] if appearance is not None else [],
                maxlen=self._gallery_size(class_id),
            ),
            'owner_id': owner_hint if class_id in self.config.BAG_CLASS_IDS else None,
            'persisted_only': False,
        }
        used_ids.add(new_id)
        return new_id, {'reason': 'new', 'score': 0.0}

    def _update(self, stable_id: int, bbox: List[float], frame_number: int,
                bt_id: Optional[int], appearance: Optional[np.ndarray] = None):
        entry = self.entries[stable_id]
        entry['bbox'] = bbox
        entry['last_frame'] = frame_number
        entry['last_seen_ts'] = time.time()
        entry['persisted_only'] = False
        if bt_id is not None:
            entry['bt_ids'].add(bt_id)
        if appearance is not None:
            existing = entry.get('appearance')
            if existing is None:
                entry['appearance'] = appearance.copy()
            else:
                alpha = self.config.REID_APPEARANCE_UPDATE_WEIGHT
                blended = ((1.0 - alpha) * existing) + (alpha * appearance)
                norm = float(np.linalg.norm(blended))
                entry['appearance'] = blended / norm if norm > 0 else appearance.copy()
            gallery = entry.setdefault(
                'appearance_gallery',
                deque(maxlen=self._gallery_size(entry['class_id']))
            )
            gallery.append(appearance.copy())

    def _expire(self, frame_number: int):
        now = time.time()
        self.entries = {
            key: value for key, value in self.entries.items()
            if (
                value.get('persisted_only')
                and now - float(value.get('last_seen_ts', 0.0)) <= self.config.REID_PERSIST_MAX_AGE_SECONDS
            )
            or value.get('class_id') == self.config.PERSON_CLASS_ID
            or frame_number - value['last_frame'] <= self.config.REID_MAX_AGE_FRAMES
        }
        self.used_ids_by_frame = {
            key: value for key, value in self.used_ids_by_frame.items()
            if key >= frame_number - 2
        }

    def _new_id(self, class_id: int) -> int:
        if class_id == self.config.PERSON_CLASS_ID:
            stable_id = self.PERSON_BASE_ID + self.next_person_id
            self.next_person_id += 1
            return stable_id

        stable_id = self.BAG_BASE_ID + self.next_bag_id
        self.next_bag_id += 1
        return stable_id

    @staticmethod
    def _centroid(bbox: List[float]) -> Tuple[float, float]:
        return (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2

    def _gallery_size(self, class_id: int) -> int:
        if class_id == self.config.PERSON_CLASS_ID:
            return self.config.REID_PERSON_GALLERY_SIZE
        return self.config.REID_BAG_GALLERY_SIZE

    def _match_by_geometry(self, bbox: List[float], class_id: int, frame_number: int,
                           used_ids: set, centroid_thresh: float) -> Tuple[Optional[int], Optional[Dict[str, float | str]]]:
        best_id = None
        best_score = (-1.0, float('-inf'))
        best_meta = None
        cx, cy = self._centroid(bbox)

        for stable_id, entry in self.entries.items():
            if entry['class_id'] != class_id or stable_id in used_ids:
                continue

            if entry.get('persisted_only'):
                continue

            age = frame_number - entry['last_frame']
            if age < 0 or age > self.config.REID_MATCH_MAX_AGE_FRAMES:
                continue

            iou = self._iou(bbox, entry['bbox'])
            ex, ey = self._centroid(entry['bbox'])
            centroid_dist = float(np.hypot(cx - ex, cy - ey))

            if centroid_dist > centroid_thresh and iou < self.config.REID_IOU_THRESH:
                continue

            score = (iou, -centroid_dist)
            if score > best_score:
                best_score = score
                best_id = stable_id
                best_meta = {
                    'reason': 'geo',
                    'score': float(iou),
                    'iou': float(iou),
                    'centroid_dist': float(centroid_dist),
                }

        return best_id, best_meta

    def _match_person_by_appearance(self, bbox: List[float], appearance: np.ndarray,
                                    frame_number: int, used_ids: set) -> Tuple[Optional[int], float]:
        best_id = None
        best_similarity = self.config.REID_PERSON_EMBED_THRESHOLD

        for stable_id, entry in self.entries.items():
            if entry['class_id'] != self.config.PERSON_CLASS_ID or stable_id in used_ids:
                continue

            if entry.get('persisted_only'):
                last_seen_ts = float(entry.get('last_seen_ts', 0.0))
                if last_seen_ts <= 0:
                    continue
                if time.time() - last_seen_ts > self.config.REID_PERSIST_MAX_AGE_SECONDS:
                    continue
            else:
                age = frame_number - entry['last_frame']
                if age < 0 or age > self.config.REID_APPEARANCE_MAX_AGE_FRAMES:
                    continue

            stored_appearance = entry.get('appearance')
            gallery = entry.get('appearance_gallery') or []
            if stored_appearance is None and not gallery:
                continue

            if not self._appearance_size_compatible(bbox, entry['bbox']):
                continue

            similarities = [float(np.dot(appearance, gallery_feature)) for gallery_feature in gallery]
            if stored_appearance is not None:
                similarities.append(float(np.dot(appearance, stored_appearance)))
            similarity = max(similarities) if similarities else -1.0
            if similarity > best_similarity:
                best_similarity = similarity
                best_id = stable_id

        return best_id, best_similarity

    def _match_bag_by_appearance(self, bbox: List[float], class_id: int, appearance: np.ndarray,
                                 frame_number: int, used_ids: set,
                                 owner_hint: Optional[int]) -> Tuple[Optional[int], float]:
        best_id = None
        best_similarity = self.config.REID_BAG_EMBED_THRESHOLD

        for stable_id, entry in self.entries.items():
            if entry['class_id'] != class_id or stable_id in used_ids:
                continue

            if entry.get('persisted_only'):
                last_seen_ts = float(entry.get('last_seen_ts', 0.0))
                if last_seen_ts <= 0:
                    continue
                if time.time() - last_seen_ts > self.config.REID_PERSIST_MAX_AGE_SECONDS:
                    continue
            else:
                age = frame_number - entry['last_frame']
                if age < 0 or age > self.config.REID_BAG_APPEARANCE_MAX_AGE_FRAMES:
                    continue

            stored_appearance = entry.get('appearance')
            gallery = entry.get('appearance_gallery') or []
            if stored_appearance is None and not gallery:
                continue

            if not self._bag_size_compatible(bbox, entry['bbox']):
                continue

            similarities = [float(np.dot(appearance, gallery_feature)) for gallery_feature in gallery]
            if stored_appearance is not None:
                similarities.append(float(np.dot(appearance, stored_appearance)))
            similarity = max(similarities) if similarities else -1.0

            if owner_hint is not None and entry.get('owner_id') == owner_hint:
                similarity += self.config.REID_BAG_OWNER_MATCH_BONUS

            if similarity > best_similarity:
                best_similarity = similarity
                best_id = stable_id

        return best_id, best_similarity

    def _appearance_size_compatible(self, bbox_a: List[float], bbox_b: List[float]) -> bool:
        height_a = max(1.0, bbox_a[3] - bbox_a[1])
        height_b = max(1.0, bbox_b[3] - bbox_b[1])
        size_ratio = min(height_a, height_b) / max(height_a, height_b)
        if size_ratio < self.config.REID_PERSON_SIZE_RATIO_MIN:
            return False

        width_a = max(1.0, bbox_a[2] - bbox_a[0])
        width_b = max(1.0, bbox_b[2] - bbox_b[0])
        aspect_a = width_a / height_a
        aspect_b = width_b / height_b
        return abs(aspect_a - aspect_b) <= self.config.REID_PERSON_ASPECT_RATIO_MAX_DIFF

    def _bag_size_compatible(self, bbox_a: List[float], bbox_b: List[float]) -> bool:
        height_a = max(1.0, bbox_a[3] - bbox_a[1])
        height_b = max(1.0, bbox_b[3] - bbox_b[1])
        size_ratio = min(height_a, height_b) / max(height_a, height_b)
        if size_ratio < self.config.REID_BAG_SIZE_RATIO_MIN:
            return False

        width_a = max(1.0, bbox_a[2] - bbox_a[0])
        width_b = max(1.0, bbox_b[2] - bbox_b[0])
        aspect_a = width_a / height_a
        aspect_b = width_b / height_b
        return abs(aspect_a - aspect_b) <= self.config.REID_BAG_ASPECT_RATIO_MAX_DIFF

    def update_bag_owner(self, bag_id: int, owner_id: Optional[int]):
        entry = self.entries.get(bag_id)
        if entry is None or entry.get('class_id') not in self.config.BAG_CLASS_IDS:
            return
        entry['owner_id'] = owner_id

    @staticmethod
    def _iou(bbox_a: List[float], bbox_b: List[float]) -> float:
        ax1, ay1, ax2, ay2 = bbox_a
        bx1, by1, bx2, by2 = bbox_b
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        if inter == 0:
            return 0.0
        union = ((ax2 - ax1) * (ay2 - ay1)) + ((bx2 - bx1) * (by2 - by1)) - inter
        return inter / union if union > 0 else 0.0

    def load_persistent_entries(self):
        if self.persist_path is None or not self.persist_path.exists():
            return

        try:
            with open(self.persist_path, 'rb') as file_handle:
                payload = pickle.load(file_handle)
        except Exception as e:
            print(f"⚠️  Failed to load persistent person registry: {e}")
            return

        loaded_person_count = 0
        loaded_bag_count = 0
        next_person_id = int(payload.get('next_person_id', 1)) if isinstance(payload, dict) else 1
        next_bag_id = int(payload.get('next_bag_id', 1)) if isinstance(payload, dict) else 1
        entries = payload.get('entries', []) if isinstance(payload, dict) else []
        for item in entries:
            try:
                stable_id = int(item['stable_id'])
                class_id = int(item.get('class_id', self.config.PERSON_CLASS_ID))
                if stable_id < self.PERSON_BASE_ID:
                    continue

                appearance_list = item.get('appearance', [])
                appearance = np.asarray(appearance_list, dtype=np.float32) if appearance_list else None
                gallery_arrays = [
                    np.asarray(feature, dtype=np.float32)
                    for feature in item.get('appearance_gallery', [])
                ]
                if appearance is None and not gallery_arrays:
                    continue

                self.entries[stable_id] = {
                    'bbox': item.get('bbox', [0.0, 0.0, 0.0, 0.0]),
                    'class_id': class_id,
                    'last_frame': 0,
                    'last_seen_ts': float(item.get('last_seen_ts', 0.0)),
                    'bt_ids': set(item.get('bt_ids', [])),
                    'appearance': appearance,
                    'appearance_gallery': deque(gallery_arrays, maxlen=self._gallery_size(class_id)),
                    'saved_frame_count': int(item.get('saved_frame_count', 0)),
                    'last_logged_frame': int(item.get('last_logged_frame', -1)),
                    'owner_id': item.get('owner_id'),
                    'persisted_only': True,
                }
                if class_id == self.config.PERSON_CLASS_ID:
                    loaded_person_count += 1
                elif class_id in self.config.BAG_CLASS_IDS:
                    loaded_bag_count += 1
            except Exception:
                continue

        self.next_person_id = max(self.next_person_id, next_person_id)
        self.next_bag_id = max(self.next_bag_id, next_bag_id)
        self.loaded_person_count = loaded_person_count
        self.loaded_bag_count = loaded_bag_count

    def save_persistent_entries(self):
        if self.persist_path is None:
            return

        persisted_entries = []
        for stable_id, entry in self.entries.items():
            appearance = entry.get('appearance')
            gallery = entry.get('appearance_gallery') or []
            if appearance is None and not gallery:
                continue

            persisted_entries.append({
                'stable_id': stable_id,
                'class_id': int(entry.get('class_id', self.config.PERSON_CLASS_ID)),
                'bbox': entry.get('bbox', [0.0, 0.0, 0.0, 0.0]),
                'last_seen_ts': float(entry.get('last_seen_ts', time.time())),
                'bt_ids': sorted(entry.get('bt_ids', set())),
                'appearance': appearance.tolist() if appearance is not None else [],
                'appearance_gallery': [feature.tolist() for feature in gallery],
                'saved_frame_count': int(entry.get('saved_frame_count', 0)),
                'last_logged_frame': int(entry.get('last_logged_frame', -1)),
                'owner_id': entry.get('owner_id'),
            })

        payload = {
            'entries': persisted_entries,
            'next_person_id': self.next_person_id,
            'next_bag_id': self.next_bag_id,
        }

        try:
            self.persist_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.persist_path, 'wb') as file_handle:
                pickle.dump(payload, file_handle)
            self._write_entry_logs(persisted_entries)
        except Exception as e:
            print(f"⚠️  Failed to save persistent person registry: {e}")

    def _write_entry_logs(self, persisted_entries: List[Dict]):
        if self.persist_log_dir is None:
            return

        self.persist_log_dir.mkdir(parents=True, exist_ok=True)
        for item in persisted_entries:
            class_id = int(item.get('class_id', self.config.PERSON_CLASS_ID))
            prefix = self._entry_prefix(class_id)
            summary_path = self.persist_log_dir / f"{prefix}_{item['stable_id']}.json"
            summary = {
                'stable_id': item['stable_id'],
                'class_id': class_id,
                'last_seen_ts': item['last_seen_ts'],
                'gallery_size': len(item.get('appearance_gallery', [])),
                'bt_ids': item.get('bt_ids', []),
                'bbox': item.get('bbox', [0.0, 0.0, 0.0, 0.0]),
                'saved_frame_count': int(item.get('saved_frame_count', 0)),
                'last_logged_frame': int(item.get('last_logged_frame', -1)),
                'owner_id': item.get('owner_id'),
            }
            with open(summary_path, 'w', encoding='utf-8') as file_handle:
                json.dump(summary, file_handle, indent=2)

    def _entry_prefix(self, class_id: int) -> str:
        if class_id == self.config.PERSON_CLASS_ID:
            return 'person'
        return 'bag'

    def log_person_frame(self, stable_id: int, frame_number: int, frame: np.ndarray,
                         bbox: List[float], match_meta: Optional[Dict[str, float | str]] = None,
                         bt_id: Optional[int] = None):
        self._log_entity_frame(
            stable_id,
            self.config.PERSON_CLASS_ID,
            frame_number,
            frame,
            bbox,
            match_meta,
            bt_id,
        )

    def log_bag_frame(self, stable_id: int, frame_number: int, frame: np.ndarray,
                      bbox: List[float], match_meta: Optional[Dict[str, float | str]] = None,
                      bt_id: Optional[int] = None):
        self._log_entity_frame(
            stable_id,
            None,
            frame_number,
            frame,
            bbox,
            match_meta,
            bt_id,
        )

    def _log_entity_frame(self, stable_id: int, expected_class_id: Optional[int],
                          frame_number: int, frame: np.ndarray, bbox: List[float],
                          match_meta: Optional[Dict[str, float | str]] = None,
                          bt_id: Optional[int] = None):
        if self.persist_log_dir is None:
            return

        entry = self.entries.get(stable_id)
        if entry is None:
            return
        if expected_class_id is not None and entry.get('class_id') != expected_class_id:
            return
        if expected_class_id is None and entry.get('class_id') not in self.config.BAG_CLASS_IDS:
            return
        if entry.get('last_logged_frame') == frame_number:
            return

        frame_h, frame_w = frame.shape[:2]
        x1 = max(0, min(frame_w - 1, int(bbox[0])))
        y1 = max(0, min(frame_h - 1, int(bbox[1])))
        x2 = max(0, min(frame_w, int(bbox[2])))
        y2 = max(0, min(frame_h, int(bbox[3])))
        if x2 <= x1 or y2 <= y1:
            return

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return

        prefix = self._entry_prefix(int(entry.get('class_id', self.config.PERSON_CLASS_ID)))
        entity_dir = self.persist_log_dir / f"{prefix}_{stable_id}"
        frames_dir = entity_dir / self.config.REID_FRAME_IMAGE_DIRNAME
        frames_dir.mkdir(parents=True, exist_ok=True)

        filename = f"frame_{frame_number:06d}{self.config.REID_FRAME_IMAGE_EXT}"
        image_path = frames_dir / filename
        if not cv2.imwrite(str(image_path), crop):
            return

        metadata_path = entity_dir / self.config.REID_FRAME_METADATA_NAME
        record = {
            'frame_number': int(frame_number),
            'image_path': str(image_path),
            'class_id': int(entry.get('class_id', self.config.PERSON_CLASS_ID)),
            'bbox': [float(x1), float(y1), float(x2), float(y2)],
            'match_reason': str((match_meta or {}).get('reason', '')),
            'match_score': float((match_meta or {}).get('score', 0.0)),
            'bt_id': int(bt_id) if bt_id is not None else None,
            'owner_id': entry.get('owner_id'),
            'timestamp': time.time(),
        }
        with open(metadata_path, 'a', encoding='utf-8') as file_handle:
            file_handle.write(json.dumps(record) + "\n")

        entry['saved_frame_count'] = int(entry.get('saved_frame_count', 0)) + 1
        entry['last_logged_frame'] = frame_number


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 3: BAG STATE (Section 7.2 - Ownership States)
# ═══════════════════════════════════════════════════════════════════════════
@dataclass
class BagState:
    """
    Persistent state for each bag (Section 7.2)
    
    Maintains:
    - owner_id: Current owner
    - owner_since: When ownership was assigned
    - last_close_time: Last time owner was close
    - candidate_owner: Potential new owner
    - distance_history: Rolling window of distances (trend-based)
    """
    bag_id: int
    owner_id: Optional[int] = None
    owner_since: float = 0.0
    last_close_time: float = 0.0
    candidate_owner: Optional[int] = None
    candidate_since: float = 0.0
    owner_distance_history: Deque[float] = None
    status: str = "OK"  # OK, POTENTIAL, UNATTENDED
    
    def __post_init__(self):
        if self.owner_distance_history is None:
            self.owner_distance_history = deque(maxlen=BGSConfig.DISTANCE_HISTORY_SIZE)


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 4: OWNERSHIP MANAGER (Section 7 - Ownership Persistence Logic)
# ═══════════════════════════════════════════════════════════════════════════
class OwnershipManager:
    """
    Manages ownership association with persistence (Section 7)
    
    Key Features (Section 7.1):
    - Trend-based decisions (NOT frame-by-frame)
    - Rolling distance averages
    - Ownership locking
    - Confirmation time before assignment
    - Smooth status transitions
    """
    
    def __init__(self, distance_estimator: DistanceEstimator):
        self.distance_estimator = distance_estimator
        self.config = BGSConfig
        self.bag_states: Dict[int, BagState] = {}
    
    def update_ownership(self, bags: List[Dict], people: List[Dict], 
                        current_time: float) -> Dict[int, BagState]:
        """
        Update ownership for all bags (Section 7 - Main Logic)
        
        Args:
            bags: List of bag detections with 3D positions
            people: List of person detections with 3D positions
            current_time: Current timestamp
            
        Returns:
            Dictionary of bag_id -> BagState
        """
        # Get active bag IDs
        active_bag_ids = {bag['id'] for bag in bags}
        
        # Remove bags no longer detected
        self.bag_states = {
            k: v for k, v in self.bag_states.items() 
            if k in active_bag_ids
        }
        
        # Initialize new bags
        for bag in bags:
            if bag['id'] not in self.bag_states:
                self.bag_states[bag['id']] = BagState(
                    bag_id=bag['id'],
                    last_close_time=current_time
                )
        
        # Update each bag's ownership
        for bag in bags:
            bag_id = bag['id']
            bag_state = self.bag_states[bag_id]
            bag_pos = bag['position_3d']
            
            # Calculate distances to all people
            person_distances = []
            for person in people:
                person_id = person['id']
                person_pos = person['position_3d']
                distance = self.distance_estimator.calculate_distance(bag_pos, person_pos)
                person_distances.append((person_id, distance))
            
            # Sort by distance (closest first)
            person_distances.sort(key=lambda x: x[1])
            
            owner_distance = None
            if bag_state.owner_id is not None:
                for pid, dist in person_distances:
                    if pid == bag_state.owner_id:
                        owner_distance = dist
                        break

            sample_distance = (
                owner_distance
                if owner_distance is not None
                else (person_distances[0][1] if person_distances else 999.0)
            )
            bag_state.owner_distance_history.append(sample_distance)
            smoothed_distance = float(np.mean(list(bag_state.owner_distance_history)))

            closest_distance = person_distances[0][1] if person_distances else 999.0
            
            # Apply ownership rules (Section 7.3)
            self._apply_ownership_rules(
                bag_state,
                person_distances,
                closest_distance,
                owner_distance,
                current_time,
            )
            
            # Update bag status (Section 8 - Unattended Logic)
            self._update_bag_status(bag_state, current_time)
        
        return self.bag_states
    
    def _apply_ownership_rules(self, bag_state: BagState,
                               person_distances: List[Tuple[int, float]],
                               closest_distance: float,
                               owner_distance: Optional[float],
                               current_time: float):
        """
        Apply ownership assignment rules (Section 7.3)
        
        Rules:
        1. New owner assigned only after confirmation time
        2. Ownership locked for fixed duration
        3. Switching requires significant distance improvement
        4. Trend-based, not single-frame
        """
        if not person_distances:
            # No people detected
            return
        
        closest_person_id, _ = person_distances[0]
        
        # CASE 1: Bag has NO owner
        if bag_state.owner_id is None:
            if closest_distance <= self.config.ASSIGNMENT_DISTANCE:
                # Person is close - check confirmation time
                if bag_state.candidate_owner == closest_person_id:
                    # Same candidate - check if confirmed
                    time_as_candidate = current_time - bag_state.candidate_since
                    if time_as_candidate >= self.config.CONFIRMATION_TIME:
                        # ASSIGN OWNERSHIP
                        bag_state.owner_id = closest_person_id
                        bag_state.owner_since = current_time
                        bag_state.last_close_time = current_time
                        bag_state.candidate_owner = None
                else:
                    # New candidate
                    bag_state.candidate_owner = closest_person_id
                    bag_state.candidate_since = current_time
            else:
                # Too far - reset candidate
                bag_state.candidate_owner = None
        
        # CASE 2: Bag HAS owner
        release_after = self.config.UNATTENDED_THRESHOLD + self.config.OWNERSHIP_RELEASE_GRACE
        if (current_time - bag_state.last_close_time >= release_after) and owner_distance is None:
            bag_state.owner_id = None
            bag_state.owner_since = 0.0
            bag_state.candidate_owner = None
            bag_state.owner_distance_history.clear()
            return

        if owner_distance is not None and owner_distance <= self.config.ASSIGNMENT_DISTANCE:
            bag_state.last_close_time = current_time

        time_locked = current_time - bag_state.owner_since
        if time_locked < self.config.OWNERSHIP_LOCK_TIME:
            return

        owner_close = owner_distance is not None and owner_distance <= self.config.ASSIGNMENT_DISTANCE
        if owner_close or closest_distance > self.config.ASSIGNMENT_DISTANCE:
            return

        if owner_distance is None:
            bag_state.owner_id = closest_person_id
            bag_state.owner_since = current_time
            bag_state.last_close_time = current_time
            bag_state.owner_distance_history.clear()
            return

        if owner_distance - closest_distance >= self.config.SWITCH_DISTANCE_IMPROVEMENT:
            bag_state.owner_id = closest_person_id
            bag_state.owner_since = current_time
            bag_state.last_close_time = current_time
            bag_state.owner_distance_history.clear()
    
    def _update_bag_status(self, bag_state: BagState, current_time: float):
        """
        Update bag status: OK, POTENTIAL, UNATTENDED (Section 8)
        
        Timing Rules (Section 8.2):
        - OK: Owner is close
        - POTENTIAL: Owner far for short period
        - UNATTENDED: Owner absent/far for prolonged period
        """
        if bag_state.owner_distance_history:
            smoothed_owner_distance = float(np.mean(list(bag_state.owner_distance_history)))
            if smoothed_owner_distance <= self.config.ASSIGNMENT_DISTANCE:
                bag_state.last_close_time = current_time

        time_since_close = current_time - bag_state.last_close_time
        
        if time_since_close >= self.config.UNATTENDED_THRESHOLD:
            bag_state.status = "UNATTENDED"
        elif time_since_close >= self.config.POTENTIAL_THRESHOLD:
            bag_state.status = "POTENTIAL"
        else:
            bag_state.status = "OK"


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 5: VISUALIZER (Section 10 - Visualization & Debugging)
# ═══════════════════════════════════════════════════════════════════════════
class Visualizer:
    """Professional visualization (Section 10)"""
    
    def __init__(self):
        self.config = BGSConfig
    
    def draw_detection(self, frame: np.ndarray, detection: Dict, 
                      bag_state: Optional[BagState] = None):
        """Draw detection with appropriate styling"""
        x1, y1, x2, y2 = map(int, detection['bbox'])
        obj_id = detection['id']
        class_name = detection['class']
        
        # Determine color and thickness based on type and status
        if class_name == 'person':
            color = self.config.COLOR_PERSON
            thickness = self.config.LINE_THICKNESS
            label = f"Person #{obj_id}"
            match_reason = detection.get('match_reason')
            match_score = detection.get('match_score')
            if match_reason:
                label += f" [{match_reason}]"
            if match_reason in {'reid', 'geo'}:
                label += f" {match_score:.2f}"
        else:
            # Bag - color based on status
            if bag_state:
                if bag_state.status == "UNATTENDED":
                    color = self.config.COLOR_BAG_UNATTENDED
                    thickness = 5
                elif bag_state.status == "POTENTIAL":
                    color = self.config.COLOR_BAG_POTENTIAL
                    thickness = 4
                else:
                    color = self.config.COLOR_BAG_OK
                    thickness = self.config.LINE_THICKNESS
                
                label = f"{class_name.upper()} #{obj_id}"
                if bag_state.owner_id:
                    label += f" [Owner: #{bag_state.owner_id}]"
                label += f" [{bag_state.status}]"
            else:
                color = self.config.COLOR_BAG_OK
                thickness = self.config.LINE_THICKNESS
                label = f"{class_name.upper()} #{obj_id}"
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        
        # Corner accents
        corner_len = 20
        cv2.line(frame, (x1, y1), (x1+corner_len, y1), color, thickness+1)
        cv2.line(frame, (x1, y1), (x1, y1+corner_len), color, thickness+1)
        
        # Label with shadow
        (lw, lh), _ = cv2.getTextSize(label, self.config.FONT, self.config.FONT_SCALE, 2)
        ly = max(y1 - 15, lh + 8)
        
        # Shadow
        cv2.rectangle(frame, (x1+3, ly-lh-5+3), (x1+lw+13, ly+5+3), (0,0,0), -1)
        # Main
        cv2.rectangle(frame, (x1, ly-lh-5), (x1+lw+10, ly+5), color, -1)
        cv2.putText(frame, label, (x1+5, ly), self.config.FONT, 
                   self.config.FONT_SCALE, self.config.COLOR_TEXT, 2, cv2.LINE_AA)
        
        # Distance label if available
        if 'distance_to_owner' in detection and self.config.SHOW_DISTANCE_LABELS:
            dist_text = f"{detection['distance_to_owner']:.2f}m"
            cv2.putText(frame, dist_text, (x1, y2+20), self.config.FONT,
                       0.5, color, 2, cv2.LINE_AA)
    
    def draw_distance_line(self, frame: np.ndarray, bag_bbox: List[float],
                          person_bbox: List[float], distance: float):
        """Draw line connecting bag to owner with distance label"""
        if not self.config.SHOW_DISTANCE_LINES:
            return
        
        # Calculate centers
        bag_cx = int((bag_bbox[0] + bag_bbox[2]) / 2)
        bag_cy = int((bag_bbox[1] + bag_bbox[3]) / 2)
        person_cx = int((person_bbox[0] + person_bbox[2]) / 2)
        person_cy = int((person_bbox[1] + person_bbox[3]) / 2)
        
        # Draw line
        cv2.line(frame, (bag_cx, bag_cy), (person_cx, person_cy),
                self.config.COLOR_DISTANCE_LINE, 2, cv2.LINE_AA)
        
        # Draw distance label at midpoint
        mid_x = (bag_cx + person_cx) // 2
        mid_y = (bag_cy + person_cy) // 2
        
        dist_text = f"{distance:.2f}m"
        (tw, th), _ = cv2.getTextSize(dist_text, self.config.FONT, 0.6, 2)
        
        # Background
        cv2.rectangle(frame, (mid_x-5, mid_y-th-5), (mid_x+tw+5, mid_y+5),
                     (0, 0, 0), -1)
        cv2.putText(frame, dist_text, (mid_x, mid_y), self.config.FONT,
                   0.6, self.config.COLOR_DISTANCE_LINE, 2, cv2.LINE_AA)
    
    def draw_debug_overlay(self, frame: np.ndarray, stats: Dict):
        """Draw comprehensive debug overlay (Section 10 - Optional Debug Overlay)"""
        if not self.config.SHOW_DEBUG_OVERLAY:
            return
        
        panel_h = 200
        panel_w = 550
        
        # Gradient background
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (panel_w, panel_h), (20, 20, 20), -1)
        cv2.rectangle(overlay, (5, 5), (panel_w-5, panel_h-5), (40, 40, 40), -1)
        frame[:] = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
        
        # Border
        cv2.rectangle(frame, (0, 0), (panel_w, panel_h), self.config.COLOR_HIGHLIGHT, 2)
        
        y = 30
        spacing = 25
        
        # Title
        cv2.putText(frame, "BAG GUARD SYSTEM", (15, y),
                   self.config.FONT, 0.8, self.config.COLOR_HIGHLIGHT, 2, cv2.LINE_AA)
        y += 10
        cv2.line(frame, (15, y), (panel_w-15, y), self.config.COLOR_HIGHLIGHT, 2)
        
        # Stats
        y += spacing
        cv2.putText(frame, f"FPS: {stats.get('fps', 0)}", (15, y),
                   self.config.FONT, 0.6, self.config.COLOR_TEXT, 2, cv2.LINE_AA)
        
        y += spacing - 3
        cv2.putText(frame, f"People: {stats.get('people_count', 0)} | "
                          f"Bags: {stats.get('bags_count', 0)}", (15, y),
                   self.config.FONT, 0.6, self.config.COLOR_PERSON, 2, cv2.LINE_AA)
        
        y += spacing - 3
        ok_count = stats.get('bags_ok', 0)
        pot_count = stats.get('bags_potential', 0)
        un_count = stats.get('bags_unattended', 0)
        cv2.putText(frame, f"Status: OK:{ok_count} POT:{pot_count} UN:{un_count}", (15, y),
                   self.config.FONT, 0.6, self.config.COLOR_BAG_OK, 2, cv2.LINE_AA)
        
        y += spacing - 3
        if un_count > 0:
            cv2.putText(frame, f"ALERT: UNATTENDED BAGS: {un_count}", (15, y),
                       self.config.FONT, 0.65, self.config.COLOR_BAG_UNATTENDED, 3, cv2.LINE_AA)
        else:
            cv2.putText(frame, f"All Bags Monitored", (15, y),
                       self.config.FONT, 0.6, self.config.COLOR_TEXT, 2, cv2.LINE_AA)
        
        y += spacing - 3
        cv2.putText(frame, f"Tracker: {self.config.TRACKER.replace('.yaml', '')}", (15, y),
                   self.config.FONT, 0.5, (180, 180, 180), 1, cv2.LINE_AA)
        
        y += spacing - 5
        cv2.putText(frame, f"Frame: {stats.get('frame_number', 0)}", (15, y),
                   self.config.FONT, 0.5, (180, 180, 180), 1, cv2.LINE_AA)
        
        # Parameters (right side)
        y2 = 30 + spacing + 10
        x2 = 280
        cv2.putText(frame, "PARAMETERS:", (x2, y2),
                   self.config.FONT, 0.5, self.config.COLOR_HIGHLIGHT, 1, cv2.LINE_AA)
        y2 += 18
        cv2.putText(frame, f"Assign Dist: {self.config.ASSIGNMENT_DISTANCE}m", (x2, y2),
                   self.config.FONT, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        y2 += 16
        cv2.putText(frame, f"Potential: {self.config.POTENTIAL_THRESHOLD}s", (x2, y2),
                   self.config.FONT, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        y2 += 16
        cv2.putText(frame, f"Unattended: {self.config.UNATTENDED_THRESHOLD}s", (x2, y2),
                   self.config.FONT, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        y2 += 16
        cv2.putText(frame, f"Lock Time: {self.config.OWNERSHIP_LOCK_TIME}s", (x2, y2),
                   self.config.FONT, 0.45, (200, 200, 200), 1, cv2.LINE_AA)


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 6: MAIN SYSTEM (Integration)
# ═══════════════════════════════════════════════════════════════════════════
class BagGuardSystem:
    """
    Main BGS System - Full Specification Implementation
    
    Integrates:
    - Detection (Section 4)
    - Tracking (Section 5)
    - Distance Estimation (Section 6)
    - Ownership Management (Section 7)
    - Unattended Detection (Section 8)
    - Visualization (Section 10)
    """
    
    def __init__(self, model_path: str, video_path: str, output_path: str,
                 imgsz: int, max_fps: float, skip: int, half: bool,
                 tracker_profile: str = "main", show: bool = False):
        self.model_path = model_path
        self.video_path = video_path
        self.output_path = output_path
        self.imgsz = int(imgsz)
        self.max_fps = float(max_fps)
        self.skip = max(0, int(skip))
        self.half = bool(half)
        self.show = bool(show)
        self.tracker_profile = str(tracker_profile).strip().lower()
        self.config = BGSConfig
        resolved_model_path = os.path.abspath(model_path)
        
        print("\n" + "="*80)
        print("BAG GUARD SYSTEM - FULL SPECIFICATION IMPLEMENTATION")
        print("="*80)
        print("📋 Specification Compliance:")
        print("  ✓ Section 4: Object Detection (YOLOv8)")
        print("  ✓ Section 5: Stable ID Tracking (ByteTrack)")
        print("  ✓ Section 6: Distance Estimation (Monocular Trigonometry)")
        print("  ✓ Section 7: Ownership Persistence (Trend-Based Logic)")
        print("  ✓ Section 8: Unattended Detection (3-State: OK/POTENTIAL/UNATTENDED)")
        print("  ✓ Section 9: Frozen Parameters (Reproducibility)")
        print("  ✓ Section 10: Professional Visualization")
        print("="*80)
        print(f"Model: {model_path}")
        print(f"Resolved model path: {resolved_model_path}")
        print(f"Video: {video_path}")
        print(f"Output: {output_path}")
        print("="*80)

        if not os.path.exists(resolved_model_path):
            print(f"❌ Weights file not found: {resolved_model_path}")
            sys.exit(1)
        
        # Load model
        try:
            self.model = YOLO(resolved_model_path)
            print("✓ YOLO model loaded successfully")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            sys.exit(1)

        self.cuda_available = torch.cuda.is_available()
        self.device = "cuda:0" if self.cuda_available else "cpu"
        if not self.cuda_available:
            self.half = False
        self.model.to(self.device)

        self._validate_class_config()
        self.tracker_path = self._resolve_tracker_path()
        self._log_device()
        self._run_startup_inference()
        
        # Initialize subsystems
        self.distance_estimator = DistanceEstimator()
        self.ownership_manager = OwnershipManager(self.distance_estimator)
        self.visualizer = Visualizer()
        self.person_reid = PersonReIDEmbedder(
            self.device,
            self.config.REID_PERSON_MODEL_NAME,
            self._resolve_reid_model_path(),
        )
        self.bag_reid = BagReIDEmbedder()
        reid_registry_path, reid_log_dir = self._resolve_reid_store_paths()
        self.id_registry = BGSRegistry(reid_registry_path, reid_log_dir)
        
        print("✓ Distance Estimator initialized")
        print("✓ Ownership Manager initialized")
        print("✓ Visualizer initialized")
        if self.person_reid.enabled:
            print("✓ Person re-ID initialized")
        if self.bag_reid.enabled:
            print("✓ Bag re-ID initialized")
        if self.id_registry.loaded_person_count:
            print(f"✓ Loaded persistent person logs: {self.id_registry.loaded_person_count}")
        if self.id_registry.loaded_bag_count:
            print(f"✓ Loaded persistent bag logs: {self.id_registry.loaded_bag_count}")
        
        # Statistics
        self.frame_count = 0
        self.inference_frame_count = 0
        self.person_ids_seen = set()
        self.bag_ids_seen = set()
        self.start_time = None
        self.logged_person_frame_count = 0
        self.logged_bag_frame_count = 0

    def _validate_class_config(self):
        expected_ids = self.config.EXPECTED_CLASS_IDS
        expected_map = self.config.EXPECTED_TARGET_CLASSES
        actual_ids = set(self.config.CLASS_IDS)
        actual_map = self.config.TARGET_CLASSES

        if actual_ids != expected_ids:
            print("❌ Class ID set mismatch")
            print(f"Expected: {sorted(expected_ids)}")
            print(f"Actual: {sorted(actual_ids)}")
            sys.exit(1)

        if actual_map != expected_map:
            print("❌ Class label mapping mismatch")
            print(f"Expected: {expected_map}")
            print(f"Actual: {actual_map}")
            sys.exit(1)

        if set(actual_map.keys()) != actual_ids:
            print("❌ Class IDs and label mapping keys do not align")
            print(f"Class IDs: {sorted(actual_ids)}")
            print(f"Label keys: {sorted(actual_map.keys())}")
            sys.exit(1)

        print("✓ Class IDs and label mapping verified")

    def _log_device(self):
        print(f"✓ Device: {self.device}")
        print(
            f"✓ Runtime: imgsz={self.imgsz} max_fps={self.max_fps} "
            f"skip={self.skip} half={self.half} tracker_profile={self.tracker_profile}"
        )

    def _resolve_tracker_path(self) -> str:
        tracker_value = str(self.config.TRACKER).strip()
        root = Path(__file__).resolve().parents[1]

        if tracker_value.lower() in {"bytetrack", "bytetrack.yaml"}:
            profile_to_file = {
                "main": root / "trackers" / "bytetrack_bgs.yaml",
                "stable": root / "trackers" / "bytetrack_bgs_stable.yaml",
            }
            if self.tracker_profile not in profile_to_file:
                print(f"❌ Invalid tracker profile: {self.tracker_profile}")
                print("Allowed profiles: main, stable")
                sys.exit(1)

            tracker_file = profile_to_file[self.tracker_profile]
        else:
            tracker_path = Path(tracker_value)
            tracker_file = tracker_path if tracker_path.is_absolute() else root / tracker_path

        if not tracker_file.exists():
            print(f"❌ Tracker config not found: {tracker_file}")
            sys.exit(1)

        try:
            with open(tracker_file, "r", encoding="utf-8") as file_handle:
                tracker_text = file_handle.read()
            if "tracker_type" not in tracker_text or "bytetrack" not in tracker_text.lower():
                print(f"❌ Invalid tracker config (expected ByteTrack): {tracker_file}")
                sys.exit(1)
        except Exception as e:
            print(f"❌ Failed to load tracker config: {e}")
            sys.exit(1)

        print(f"✓ Tracker config loaded: {tracker_file}")
        return str(tracker_file)

    def _resolve_reid_model_path(self) -> str:
        model_value = str(self.config.REID_PERSON_MODEL_PATH).strip()
        if not model_value:
            return ""

        root = Path(__file__).resolve().parents[1]
        model_path = Path(model_value)
        resolved = model_path if model_path.is_absolute() else root / model_path
        if resolved.exists():
            return str(resolved)

        try:
            from torchreid.reid_model_factory import get_model_url
            import gdown

            model_url = get_model_url(SimpleNamespace(name=resolved.name))
            if model_url:
                resolved.parent.mkdir(parents=True, exist_ok=True)
                print(f"⚠️  Downloading person re-ID weights to: {resolved}")
                gdown.download(model_url, str(resolved), quiet=False)
                if resolved.exists():
                    return str(resolved)
        except Exception as e:
            print(f"⚠️  Person re-ID weight download failed: {e}")

        print(f"⚠️  Person re-ID weights not found: {resolved}")
        print(f"⚠️  Falling back to built-in pretrained {self.config.REID_PERSON_MODEL_NAME}")
        return ""

    def _resolve_reid_store_paths(self) -> Tuple[Path, Path]:
        root = Path(__file__).resolve().parents[1]
        persist_path = root / self.config.REID_PERSIST_PATH
        log_dir = root / self.config.REID_PERSIST_LOG_DIR
        return persist_path, log_dir

    def _run_startup_inference(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print("⚠️  Startup inference skipped: cannot open video")
            return

        ret, frame = cap.read()
        cap.release()
        if not ret:
            print("⚠️  Startup inference skipped: no frame read")
            return

        frame = cv2.resize(frame, (self.config.IMAGE_WIDTH, self.config.IMAGE_HEIGHT))
        results = self.model.predict(
            frame,
            conf=self.config.DETECTION_CONFIDENCE,
            iou=self.config.IOU_THRESHOLD,
            classes=self.config.CLASS_IDS,
            imgsz=self.imgsz,
            device=self.device,
            half=self.half,
            verbose=False
        )

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            print("⚠️  Startup inference: 0 detections for configured classes")
            return

        class_counts = {}
        for cls_id in boxes.cls.cpu().numpy().astype(int).tolist():
            class_counts[cls_id] = class_counts.get(cls_id, 0) + 1

        person_count = class_counts.get(self.config.PERSON_CLASS_ID, 0)
        bag_count = sum(class_counts.get(cid, 0) for cid in self.config.BAG_CLASS_IDS)

        print("✓ Startup inference detections:")
        print(f"  Person: {person_count}")
        for cid in self.config.BAG_CLASS_IDS:
            label = self.config.TARGET_CLASSES.get(cid, f"class_{cid}")
            print(f"  {label}: {class_counts.get(cid, 0)}")
        print(f"  Bags total: {bag_count}")
    
    def extract_detections(self, results, frame: np.ndarray) -> Tuple[List[Dict], List[Dict]]:
        """Extract detections with 3D position estimation"""
        people = []
        bags = []
        
        if results[0].boxes is None or len(results[0].boxes) == 0:
            return people, bags
        
        boxes = results[0].boxes
        person_appearances: Dict[int, np.ndarray] = {}
        bag_appearances: Dict[int, np.ndarray] = {}

        if self.person_reid.enabled:
            person_indices = []
            person_crops = []
            for i in range(len(boxes)):
                cls_id = int(boxes[i].cls[0])
                if cls_id != self.config.PERSON_CLASS_ID:
                    continue

                conf = float(boxes[i].conf[0])
                if conf < self.config.PERSON_CONF:
                    continue

                bbox = boxes[i].xyxy[0].cpu().numpy().tolist()
                crop = self.person_reid.crop_person(frame, bbox)
                if crop is None:
                    continue

                person_indices.append(i)
                person_crops.append(crop)

            embeddings = self.person_reid.extract(person_crops)
            for idx, embedding in zip(person_indices, embeddings):
                if embedding is not None:
                    person_appearances[idx] = embedding

        if self.bag_reid.enabled:
            for i in range(len(boxes)):
                cls_id = int(boxes[i].cls[0])
                if cls_id not in self.config.BAG_CLASS_IDS:
                    continue

                conf = float(boxes[i].conf[0])
                if conf < self.config.BAG_CONF:
                    continue

                bbox = boxes[i].xyxy[0].cpu().numpy().tolist()
                crop = self.bag_reid.crop_bag(frame, bbox)
                embedding = self.bag_reid.extract_one(crop)
                if embedding is not None:
                    bag_appearances[i] = embedding

        for i in range(len(boxes)):
            cls_id = int(boxes[i].cls[0])

            if cls_id != self.config.PERSON_CLASS_ID:
                continue

            bbox = boxes[i].xyxy[0].cpu().numpy().tolist()
            conf = float(boxes[i].conf[0])
            class_name = self.config.TARGET_CLASSES[cls_id]
            if conf < self.config.PERSON_CONF:
                continue

            appearance = person_appearances.get(i)
            bt_id = int(boxes[i].id[0]) if boxes[i].id is not None else None
            stable_id, match_meta = self.id_registry.resolve(
                bbox,
                cls_id,
                self.frame_count,
                bt_id,
                appearance=appearance,
            )

            position_3d = self.distance_estimator.estimate_position_3d(bbox, is_person=True)
            self.id_registry.log_person_frame(
                stable_id,
                self.frame_count,
                frame,
                bbox,
                match_meta=match_meta,
                bt_id=bt_id,
            )
            self.logged_person_frame_count += 1

            detection = {
                'id': stable_id,
                'class': class_name,
                'class_id': cls_id,
                'bbox': bbox,
                'conf': conf,
                'position_3d': position_3d
            }
            detection['match_reason'] = str(match_meta.get('reason', ''))
            detection['match_score'] = float(match_meta.get('score', 0.0))
            people.append(detection)
            self.person_ids_seen.add(stable_id)
        
        for i in range(len(boxes)):
            cls_id = int(boxes[i].cls[0])
            
            if cls_id not in self.config.CLASS_IDS:
                continue
            
            bbox = boxes[i].xyxy[0].cpu().numpy().tolist()
            conf = float(boxes[i].conf[0])
            class_name = self.config.TARGET_CLASSES[cls_id]

            if cls_id == self.config.PERSON_CLASS_ID:
                if conf < self.config.PERSON_CONF:
                    continue
            elif cls_id in self.config.BAG_CLASS_IDS:
                if conf < self.config.BAG_CONF:
                    continue
            
            if cls_id == self.config.PERSON_CLASS_ID:
                continue

            appearance = bag_appearances.get(i)
            owner_hint = self._estimate_bag_owner_hint(bbox, people)

            bt_id = int(boxes[i].id[0]) if boxes[i].id is not None else None
            stable_id, match_meta = self.id_registry.resolve(
                bbox,
                cls_id,
                self.frame_count,
                bt_id,
                appearance=appearance,
                owner_hint=owner_hint,
            )

            position_3d = self.distance_estimator.estimate_position_3d(bbox, is_person=False)

            detection = {
                'id': stable_id,
                'class': class_name,
                'class_id': cls_id,
                'bbox': bbox,
                'conf': conf,
                'position_3d': position_3d
            }
            detection['match_reason'] = str(match_meta.get('reason', ''))
            detection['match_score'] = float(match_meta.get('score', 0.0))
            detection['owner_hint'] = owner_hint
            self.id_registry.log_bag_frame(
                stable_id,
                self.frame_count,
                frame,
                bbox,
                match_meta=match_meta,
                bt_id=bt_id,
            )
            self.logged_bag_frame_count += 1
            bags.append(detection)
            self.bag_ids_seen.add(stable_id)
        
        return people, bags

    def _estimate_bag_owner_hint(self, bag_bbox: List[float], people: List[Dict]) -> Optional[int]:
        if not people:
            return None

        bx = (bag_bbox[0] + bag_bbox[2]) / 2
        by = (bag_bbox[1] + bag_bbox[3]) / 2
        best_person_id = None
        best_dist = float(self.config.BAG_MATCH_THRESHOLD_PX)

        for person in people:
            px1, py1, px2, py2 = person['bbox']
            px = (px1 + px2) / 2
            py = (py1 + py2) / 2
            dist = float(np.hypot(bx - px, by - py))
            if dist < best_dist:
                best_dist = dist
                best_person_id = int(person['id'])

        return best_person_id

    def _refine_bag_depths(self, bags: List[Dict], people: List[Dict]):
        if not people:
            return

        for bag in bags:
            bx1, by1, bx2, by2 = bag['bbox']
            bcx = (bx1 + bx2) / 2
            bcy = (by1 + by2) / 2
            best_depth = None
            best_dist = 200.0

            for person in people:
                px1, py1, px2, py2 = person['bbox']
                pcx = (px1 + px2) / 2
                pcy = (py1 + py2) / 2
                dist = float(np.hypot(bcx - pcx, bcy - pcy))
                if dist < best_dist:
                    best_dist = dist
                    best_depth = person['position_3d'][2]

            if best_depth is not None:
                bag['position_3d'] = self.distance_estimator.estimate_position_3d(
                    bag['bbox'], is_person=False, reference_depth=best_depth
                )
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Process single frame through complete pipeline"""
        current_time = time.time()
        
        # Detection + Tracking
        results = self.model.track(
            frame,
            persist=self.config.PERSIST,
            conf=min(self.config.PERSON_CONF, self.config.BAG_CONF),
            iou=self.config.IOU_THRESHOLD,
            classes=self.config.CLASS_IDS,
            tracker=self.tracker_path,
            imgsz=self.imgsz,
            device=self.device,
            half=self.half,
            verbose=False
        )

        # Extract with 3D positions
        people, bags = self.extract_detections(results, frame)
        self._refine_bag_depths(bags, people)
        
        # Update ownership
        bag_states = self.ownership_manager.update_ownership(bags, people, current_time)
        for bag_id, bag_state in bag_states.items():
            self.id_registry.update_bag_owner(bag_id, bag_state.owner_id)
        
        # Count statuses
        status_counts = {'OK': 0, 'POTENTIAL': 0, 'UNATTENDED': 0}
        for state in bag_states.values():
            status_counts[state.status] += 1
        
        # Visualize
        for person in people:
            self.visualizer.draw_detection(frame, person)
        
        for bag in bags:
            bag_state = bag_states.get(bag['id'])
            self.visualizer.draw_detection(frame, bag, bag_state)
            
            # Draw distance line if has owner
            if bag_state and bag_state.owner_id:
                owner = next((p for p in people if p['id'] == bag_state.owner_id), None)
                if owner:
                    distance = self.distance_estimator.calculate_distance(
                        bag['position_3d'], owner['position_3d']
                    )
                    self.visualizer.draw_distance_line(
                        frame, bag['bbox'], owner['bbox'], distance
                    )
        
        # Stats
        stats = {
            'fps': 0,
            'people_count': len(people),
            'bags_count': len(bags),
            'bags_ok': status_counts['OK'],
            'bags_potential': status_counts['POTENTIAL'],
            'bags_unattended': status_counts['UNATTENDED'],
            'frame_number': self.frame_count
        }
        
        return frame, stats
    
    def run(self) -> bool:
        """Main processing loop"""
        cap = cv2.VideoCapture(self.video_path)

        cv2.setUseOptimized(True)
        
        if not cap.isOpened():
            print("❌ Cannot open video")
            return False
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"\n✓ Video: {total_frames} frames @ {fps} FPS")
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.output_path, fourcc, fps, 
                             (self.config.IMAGE_WIDTH, self.config.IMAGE_HEIGHT))
        
        print("✓ Starting processing...\n")
        
        self.start_time = time.time()
        prev_time = time.time()
        last_annotated_frame = None
        last_stats = None
        read_count = 0
        
        try:
            while cap.isOpened():
                loop_start = time.time()
                ret, frame = cap.read()
                if not ret:
                    break

                self.frame_count += 1
                read_count += 1
                frame = cv2.resize(frame, (self.config.IMAGE_WIDTH, self.config.IMAGE_HEIGHT))

                do_infer = (self.skip == 0) or (read_count % (self.skip + 1) == 1)
                if do_infer:
                    self.inference_frame_count += 1
                    current_time = time.time()

                    annotated_frame, stats = self.process_frame(frame)

                    fps_display = int(1 / (current_time - prev_time)) if current_time > prev_time else 0
                    prev_time = current_time
                    stats['fps'] = fps_display

                    self.visualizer.draw_debug_overlay(annotated_frame, stats)
                    last_annotated_frame = annotated_frame
                    last_stats = stats
                else:
                    annotated_frame = last_annotated_frame if last_annotated_frame is not None else frame
                    stats = last_stats if last_stats is not None else {
                        'fps': 0,
                        'people_count': 0,
                        'bags_count': 0,
                        'bags_ok': 0,
                        'bags_potential': 0,
                        'bags_unattended': 0,
                        'frame_number': self.frame_count
                    }

                out.write(annotated_frame)
                if self.show:
                    cv2.imshow('BGS - Full Specification', annotated_frame)

                if do_infer and self.frame_count % 30 == 0:
                    if self.frame_count % self.config.REID_PERSIST_INTERVAL_FRAMES == 0:
                        self.id_registry.save_persistent_entries()
                    if total_frames > 0:
                        progress = (self.frame_count / total_frames) * 100
                        prefix = f"Progress: {progress:5.1f}% | Frame: {self.frame_count:5d}/{total_frames}"
                    else:
                        prefix = f"LIVE | Frame: {self.frame_count:5d}"

                    print(f"{prefix} | FPS: {stats['fps']:3d} | People: {stats['people_count']} | "
                          f"Bags: {stats['bags_count']} | Unattended: {stats['bags_unattended']}")

                if self.show and cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                if self.max_fps > 0:
                    elapsed = time.time() - loop_start
                    sleep_time = max(0.0, (1.0 / self.max_fps) - elapsed)
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                    
        except Exception as e:
            print(f"\n❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            self.id_registry.save_persistent_entries()
            cap.release()
            out.release()
            cv2.destroyAllWindows()
        
        elapsed = time.time() - self.start_time
        print(f"\n{'='*80}")
        print("PROCESSING COMPLETE")
        print(f"{'='*80}")
        print(f"Frames: {self.frame_count} | Time: {elapsed:.2f}s | Avg FPS: {self.frame_count/elapsed:.2f}")
        print(f"Person IDs: {len(self.person_ids_seen)} | Bag IDs: {len(self.bag_ids_seen)}")
        print(f"Output: {self.output_path}")
        print(f"{'='*80}\n")
        
        return True


# ═══════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════
def main():
    """Main function"""
    root = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(description="Bag Guard System")
    parser.add_argument("--source", default="0", help="0 for webcam or path to a video file")
    parser.add_argument(
        "--model",
        default=str(root / "models" / "yolo26x.pt"),
        help="Path to model weights",
    )
    parser.add_argument(
        "--out",
        "--output",
        dest="output",
        default=str(root / "outputs" / "detection_output.mp4"),
        help="Output video path",
    )
    parser.add_argument("--show", action="store_true", help="Show live window")
    parser.add_argument("--imgsz", type=int, default=416, help="Inference image size")
    parser.add_argument("--max_fps", type=float, default=12, help="Max processing FPS (0 to disable)")
    parser.add_argument("--skip", type=int, default=0, help="Process every (skip+1)th frame")
    parser.add_argument("--half", action="store_true", help="Use FP16 if CUDA is available")
    parser.add_argument(
        "--tracker-profile",
        choices=["main", "stable"],
        default="main",
        help="ByteTrack preset: main (default) or stable (longer persistence)",
    )

    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.is_absolute():
        candidate = root / model_path
        if candidate.exists():
            model_path = candidate
        elif model_path.parent == Path("."):
            model_path = root / "models" / model_path
        else:
            model_path = root / model_path

    source_arg = str(args.source)
    if source_arg.isdigit():
        video_path = int(source_arg)
    else:
        video_path = Path(source_arg)
        if not video_path.is_absolute():
            if video_path.parent == Path("."):
                video_path = root / "data" / video_path
            else:
                video_path = root / video_path
        video_path = str(video_path)

    output_path = Path(args.output)
    if not output_path.is_absolute():
        if output_path.parent == Path("."):
            output_path = root / "outputs" / output_path
        else:
            output_path = root / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("\nResolved paths:")
    print(f"Resolved model path: {model_path}")
    print(f"Resolved source: {video_path}")
    print(f"Resolved output path: {output_path}")

    system = BagGuardSystem(
        str(model_path),
        video_path,
        str(output_path),
        imgsz=args.imgsz,
        max_fps=args.max_fps,
        skip=args.skip,
        half=args.half,
        tracker_profile=args.tracker_profile,
        show=args.show,
    )
    success = system.run()
    
    if success:
        print("🎉 SUCCESS! Full BGS Specification Implemented!")
        print("\n📋 IMPLEMENTED FEATURES:")
        print("  ✓ Monocular distance estimation (meters)")
        print("  ✓ Ownership persistence (trend-based)")
        print("  ✓ 3-state bag status (OK/POTENTIAL/UNATTENDED)")
        print("  ✓ Ownership locking & confirmation")
        print("  ✓ Distance lines & labels")
        print("  ✓ Frozen parameters (reproducible)")
        print("  ✓ Professional debug overlay")


if __name__ == "__main__":
    main()