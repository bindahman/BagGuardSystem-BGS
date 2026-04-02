import json
import pickle
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .config import BGSConfig


class PersonReIDEmbedder:
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
    def __init__(self, device: str):
        self.config = BGSConfig
        self.device = "cuda" if str(device).startswith("cuda") else "cpu"
        self.extractor = None
        self._fallback_notice_shown = False

        try:
            from deep_sort_realtime.embedder.embedder_pytorch import MobileNetv2_Embedder

            self.extractor = MobileNetv2_Embedder(
                half=self.device == "cuda",
                bgr=True,
                gpu=self.device == "cuda",
            )
            print("✓ Bag re-ID extractor loaded: DeepSORT MobileNetV2")
        except Exception as e:
            print(f"⚠️  Bag re-ID model unavailable, using fallback appearance features: {e}")

    @property
    def enabled(self) -> bool:
        return True

    def crop_bag(self, frame: np.ndarray, bbox: List[float]) -> Optional[np.ndarray]:
        frame_h, frame_w = frame.shape[:2]
        width = max(1.0, float(bbox[2] - bbox[0]))
        height = max(1.0, float(bbox[3] - bbox[1]))
        pad_x = int(width * 0.06)
        pad_y = int(height * 0.06)

        x1 = max(0, min(frame_w - 1, int(bbox[0]) - pad_x))
        y1 = max(0, min(frame_h - 1, int(bbox[1]) - pad_y))
        x2 = max(0, min(frame_w, int(bbox[2]) + pad_x))
        y2 = max(0, min(frame_h, int(bbox[3]) + pad_y))

        if x2 <= x1 or y2 <= y1:
            return None

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        crop_h, crop_w = crop.shape[:2]
        if crop_h < self.config.REID_BAG_MIN_CROP_SIZE or crop_w < self.config.REID_BAG_MIN_CROP_SIZE:
            return None

        return crop

    def extract(self, crops: List[np.ndarray]) -> List[Optional[np.ndarray]]:
        if not crops:
            return []

        if self.extractor is not None:
            try:
                features = self.extractor.predict(crops)
                embeddings = []
                for feature in features:
                    vector = np.asarray(feature, dtype=np.float32)
                    norm = float(np.linalg.norm(vector))
                    embeddings.append(vector / norm if norm > 0 else None)
                return embeddings
            except Exception as e:
                if not self._fallback_notice_shown:
                    print(f"⚠️  Bag re-ID model inference failed, using fallback appearance features: {e}")
                    self._fallback_notice_shown = True

        return [self._extract_fallback(crop) for crop in crops]

    def extract_one(self, crop: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if crop is None or crop.size == 0:
            return None
        return self.extract([crop])[0]

    def _extract_fallback(self, crop: Optional[np.ndarray]) -> Optional[np.ndarray]:
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
    PERSON_BASE_ID = 1_000_000
    BAG_BASE_ID = 2_000_000

    def __init__(
        self,
        person_persist_path: Optional[Path] = None,
        person_log_dir: Optional[Path] = None,
        bag_persist_path: Optional[Path] = None,
        bag_log_dir: Optional[Path] = None,
    ):
        self.config = BGSConfig
        self.entries: Dict[int, Dict] = {}
        self.next_person_id = 1
        self.next_bag_id = 1
        self.used_ids_by_frame: Dict[int, set] = {}
        self.person_persist_path = Path(person_persist_path) if person_persist_path is not None else None
        self.person_log_dir = Path(person_log_dir) if person_log_dir is not None else None
        self.bag_persist_path = Path(bag_persist_path) if bag_persist_path is not None else None
        self.bag_log_dir = Path(bag_log_dir) if bag_log_dir is not None else None
        self.loaded_person_count = 0
        self.loaded_bag_count = 0
        self.load_persistent_entries()

    def _persist_label(self, expected_class_id: Optional[int]) -> str:
        return 'person' if expected_class_id == self.config.PERSON_CLASS_ID else 'bag'

    def _matches_expected_class(self, class_id: int, expected_class_id: Optional[int]) -> bool:
        if expected_class_id == self.config.PERSON_CLASS_ID:
            return class_id == self.config.PERSON_CLASS_ID
        return class_id in self.config.BAG_CLASS_IDS

    def _is_persisted_entry_active(self, entry: Dict) -> bool:
        last_seen_ts = float(entry.get('last_seen_ts', 0.0))
        return last_seen_ts > 0 and time.time() - last_seen_ts <= self.config.REID_PERSIST_MAX_AGE_SECONDS

    @staticmethod
    def _same_feature_shape(feature_a: Optional[np.ndarray], feature_b: Optional[np.ndarray]) -> bool:
        if feature_a is None or feature_b is None:
            return False
        return np.asarray(feature_a).shape == np.asarray(feature_b).shape

    @classmethod
    def _max_similarity(cls, appearance: np.ndarray, entry: Dict) -> float:
        similarities = [
            float(np.dot(appearance, gallery_feature))
            for gallery_feature in (entry.get('appearance_gallery') or [])
            if cls._same_feature_shape(appearance, gallery_feature)
        ]
        stored_appearance = entry.get('appearance')
        if cls._same_feature_shape(appearance, stored_appearance):
            similarities.append(float(np.dot(appearance, stored_appearance)))
        return max(similarities) if similarities else -1.0

    def _serialize_entry(self, stable_id: int, entry: Dict) -> Optional[Dict]:
        appearance = entry.get('appearance')
        gallery = entry.get('appearance_gallery') or []
        if appearance is None and not gallery:
            return None

        return {
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
        }

    def _resolve_track_match(
        self,
        class_id: int,
        frame_number: int,
        used_ids: set,
        bbox: List[float],
        bt_id: Optional[int],
        appearance: Optional[np.ndarray],
    ) -> Optional[Tuple[int, Dict[str, float | str]]]:
        if bt_id is None:
            return None

        for stable_id, entry in self.entries.items():
            if stable_id in used_ids:
                continue
            if self._class_match(entry['class_id'], class_id) and bt_id in entry['bt_ids']:
                self._update(stable_id, bbox, frame_number, bt_id, appearance, class_id)
                used_ids.add(stable_id)
                return stable_id, {'reason': 'track', 'score': 1.0}
        return None

    def _geometry_override_match(
        self,
        appearance_id: int,
        appearance_score: float,
        bbox: List[float],
        class_id: int,
        frame_number: int,
        used_ids: set,
        centroid_thresh: float,
        appearance_priority: float,
        embed_threshold: float,
    ) -> Optional[Tuple[int, Dict[str, float | str]]]:
        geometry_id, geometry_meta = self._match_by_geometry(bbox, class_id, frame_number, used_ids, centroid_thresh)
        if geometry_id is None or geometry_id == appearance_id:
            return None

        geometry_iou = float(geometry_meta.get('iou', 0.0))
        appearance_margin = appearance_score - embed_threshold
        if geometry_iou >= self.config.REID_IOU_THRESH and appearance_margin < appearance_priority:
            return geometry_id, geometry_meta
        return None

    def _resolve_person_appearance_match(
        self,
        bbox: List[float],
        frame_number: int,
        used_ids: set,
        centroid_thresh: float,
        bt_id: Optional[int],
        appearance: np.ndarray,
        class_id: int,
    ) -> Optional[Tuple[int, Dict[str, float | str]]]:
        appearance_id, appearance_score = self._match_person_by_appearance(bbox, appearance, frame_number, used_ids)
        if appearance_id is None:
            return None

        override = self._geometry_override_match(
            appearance_id,
            appearance_score,
            bbox,
            class_id,
            frame_number,
            used_ids,
            centroid_thresh,
            self.config.REID_PERSON_APPEARANCE_PRIORITY,
            self.config.REID_PERSON_EMBED_THRESHOLD,
        )
        stable_id, match_meta = override or (appearance_id, {'reason': 'reid', 'score': appearance_score})
        self._update(stable_id, bbox, frame_number, bt_id, appearance, class_id)
        used_ids.add(stable_id)
        return stable_id, match_meta

    def _resolve_bag_appearance_match(
        self,
        bbox: List[float],
        frame_number: int,
        used_ids: set,
        centroid_thresh: float,
        bt_id: Optional[int],
        appearance: np.ndarray,
        class_id: int,
        owner_hint: Optional[int],
    ) -> Optional[Tuple[int, Dict[str, float | str]]]:
        appearance_id, appearance_score = self._match_bag_by_appearance(bbox, class_id, appearance, frame_number, used_ids, owner_hint)
        if appearance_id is None:
            return None

        override = self._geometry_override_match(
            appearance_id,
            appearance_score,
            bbox,
            class_id,
            frame_number,
            used_ids,
            centroid_thresh,
            self.config.REID_BAG_APPEARANCE_PRIORITY,
            self.config.REID_BAG_EMBED_THRESHOLD,
        )
        stable_id, match_meta = override or (appearance_id, {'reason': 'reid', 'score': appearance_score})
        self._update(stable_id, bbox, frame_number, bt_id, appearance, class_id)
        if owner_hint is not None:
            self.entries[stable_id]['owner_id'] = owner_hint
        used_ids.add(stable_id)
        return stable_id, match_meta

    def _create_entry(
        self,
        bbox: List[float],
        class_id: int,
        frame_number: int,
        bt_id: Optional[int],
        appearance: Optional[np.ndarray],
        owner_hint: Optional[int],
    ) -> Tuple[int, Dict[str, float | str]]:
        new_id = self._new_id(class_id)
        self.entries[new_id] = {
            'bbox': bbox,
            'class_id': class_id,
            'last_frame': frame_number,
            'last_seen_ts': time.time(),
            'bt_ids': {bt_id} if bt_id is not None else set(),
            'appearance': appearance.copy() if appearance is not None else None,
            'appearance_gallery': deque([appearance.copy()] if appearance is not None else [], maxlen=self._gallery_size(class_id)),
            'owner_id': owner_hint if class_id in self.config.BAG_CLASS_IDS else None,
            'persisted_only': False,
        }
        return new_id, {'reason': 'new', 'score': 0.0}

    def _frame_log_allowed(self, entry: Dict, expected_class_id: Optional[int]) -> bool:
        class_id = int(entry.get('class_id', self.config.PERSON_CLASS_ID))
        return self._matches_expected_class(class_id, expected_class_id)

    @staticmethod
    def _crop_from_frame(frame: np.ndarray, bbox: List[float]) -> Tuple[Optional[np.ndarray], Optional[List[float]]]:
        frame_h, frame_w = frame.shape[:2]
        x1 = max(0, min(frame_w - 1, int(bbox[0])))
        y1 = max(0, min(frame_h - 1, int(bbox[1])))
        x2 = max(0, min(frame_w, int(bbox[2])))
        y2 = max(0, min(frame_h, int(bbox[3])))
        if x2 <= x1 or y2 <= y1:
            return None, None

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None, None
        return crop, [float(x1), float(y1), float(x2), float(y2)]

    def _build_frame_record(
        self,
        entry: Dict,
        frame_number: int,
        image_path: Path,
        clipped_bbox: List[float],
        match_meta: Optional[Dict[str, float | str]],
        bt_id: Optional[int],
    ) -> Dict:
        return {
            'frame_number': int(frame_number),
            'image_path': str(image_path),
            'class_id': int(entry.get('class_id', self.config.PERSON_CLASS_ID)),
            'bbox': clipped_bbox,
            'match_reason': str((match_meta or {}).get('reason', '')),
            'match_score': float((match_meta or {}).get('score', 0.0)),
            'bt_id': int(bt_id) if bt_id is not None else None,
            'owner_id': entry.get('owner_id'),
            'timestamp': time.time(),
        }

    def resolve(
        self,
        bbox: List[float],
        class_id: int,
        frame_number: int,
        bt_id: Optional[int] = None,
        appearance: Optional[np.ndarray] = None,
        owner_hint: Optional[int] = None,
    ) -> Tuple[int, Dict[str, float | str]]:
        self._expire(frame_number)
        used_ids = self.used_ids_by_frame.setdefault(frame_number, set())
        centroid_thresh = self.config.REID_CENTROID_THRESH if class_id == self.config.PERSON_CLASS_ID else self.config.REID_BAG_CENTROID_THRESH

        track_match = self._resolve_track_match(class_id, frame_number, used_ids, bbox, bt_id, appearance)
        if track_match is not None:
            return track_match

        if class_id == self.config.PERSON_CLASS_ID and appearance is not None:
            person_match = self._resolve_person_appearance_match(
                bbox,
                frame_number,
                used_ids,
                centroid_thresh,
                bt_id,
                appearance,
                class_id,
            )
            if person_match is not None:
                return person_match

        if class_id in self.config.BAG_CLASS_IDS and appearance is not None:
            bag_match = self._resolve_bag_appearance_match(
                bbox,
                frame_number,
                used_ids,
                centroid_thresh,
                bt_id,
                appearance,
                class_id,
                owner_hint,
            )
            if bag_match is not None:
                return bag_match

        best_id, geometry_meta = self._match_by_geometry(bbox, class_id, frame_number, used_ids, centroid_thresh)
        if best_id is not None:
            self._update(best_id, bbox, frame_number, bt_id, appearance, class_id)
            used_ids.add(best_id)
            return best_id, geometry_meta

        new_id, match_meta = self._create_entry(bbox, class_id, frame_number, bt_id, appearance, owner_hint)
        used_ids.add(new_id)
        return new_id, match_meta

    def _update(
        self,
        stable_id: int,
        bbox: List[float],
        frame_number: int,
        bt_id: Optional[int],
        appearance: Optional[np.ndarray] = None,
        class_id: Optional[int] = None,
    ):
        entry = self.entries[stable_id]
        entry['bbox'] = bbox
        entry['last_frame'] = frame_number
        entry['last_seen_ts'] = time.time()
        entry['persisted_only'] = False
        if class_id is not None and class_id in self.config.BAG_CLASS_IDS:
            entry['class_id'] = class_id
        if bt_id is not None:
            entry['bt_ids'].add(bt_id)
        if appearance is not None:
            existing = entry.get('appearance')
            gallery_size = self._gallery_size(entry['class_id'])
            if existing is None or not self._same_feature_shape(appearance, existing):
                entry['appearance'] = appearance.copy()
                gallery = deque(maxlen=gallery_size)
                entry['appearance_gallery'] = gallery
            else:
                alpha = self.config.REID_APPEARANCE_UPDATE_WEIGHT
                blended = ((1.0 - alpha) * existing) + (alpha * appearance)
                norm = float(np.linalg.norm(blended))
                entry['appearance'] = blended / norm if norm > 0 else appearance.copy()
                gallery = deque(
                    [
                        feature for feature in (entry.get('appearance_gallery') or [])
                        if self._same_feature_shape(appearance, feature)
                    ],
                    maxlen=gallery_size,
                )
                entry['appearance_gallery'] = gallery
            gallery = entry.setdefault('appearance_gallery', deque(maxlen=gallery_size))
            gallery.append(appearance.copy())

    def _expire(self, frame_number: int):
        now = time.time()
        self.entries = {
            key: value for key, value in self.entries.items()
            if (
                value.get('persisted_only') and now - float(value.get('last_seen_ts', 0.0)) <= self.config.REID_PERSIST_MAX_AGE_SECONDS
            ) or value.get('class_id') == self.config.PERSON_CLASS_ID or frame_number - value['last_frame'] <= self.config.REID_MAX_AGE_FRAMES
        }
        self.used_ids_by_frame = {key: value for key, value in self.used_ids_by_frame.items() if key >= frame_number - 2}

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

    def _class_match(self, entry_class_id: int, observed_class_id: int) -> bool:
        if observed_class_id == self.config.PERSON_CLASS_ID:
            return entry_class_id == observed_class_id
        if observed_class_id in self.config.BAG_CLASS_IDS:
            return entry_class_id == observed_class_id
        return entry_class_id == observed_class_id

    def _match_by_geometry(
        self,
        bbox: List[float],
        class_id: int,
        frame_number: int,
        used_ids: set,
        centroid_thresh: float,
    ) -> Tuple[Optional[int], Optional[Dict[str, float | str]]]:
        best_id = None
        best_score = (-1.0, float('-inf'))
        best_meta = None
        cx, cy = self._centroid(bbox)

        for stable_id, entry in self.entries.items():
            if not self._class_match(entry['class_id'], class_id) or stable_id in used_ids:
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

    def _match_person_by_appearance(
        self,
        bbox: List[float],
        appearance: np.ndarray,
        frame_number: int,
        used_ids: set,
    ) -> Tuple[Optional[int], float]:
        best_id = None
        best_similarity = self.config.REID_PERSON_EMBED_THRESHOLD

        for stable_id, entry in self.entries.items():
            if entry['class_id'] != self.config.PERSON_CLASS_ID or stable_id in used_ids:
                continue

            if entry.get('persisted_only'):
                if not self._is_persisted_entry_active(entry):
                    continue
            else:
                age = frame_number - entry['last_frame']
                if age < 0 or age > self.config.REID_APPEARANCE_MAX_AGE_FRAMES:
                    continue

            if entry.get('appearance') is None and not entry.get('appearance_gallery'):
                continue
            if not self._appearance_size_compatible(bbox, entry['bbox']):
                continue

            similarity = self._max_similarity(appearance, entry)
            if similarity > best_similarity:
                best_similarity = similarity
                best_id = stable_id

        return best_id, best_similarity

    def _match_bag_by_appearance(
        self,
        bbox: List[float],
        class_id: int,
        appearance: np.ndarray,
        frame_number: int,
        used_ids: set,
        owner_hint: Optional[int],
    ) -> Tuple[Optional[int], float]:
        best_id = None
        best_similarity = self.config.REID_BAG_EMBED_THRESHOLD

        for stable_id, entry in self.entries.items():
            if entry.get('class_id') not in self.config.BAG_CLASS_IDS or stable_id in used_ids:
                continue

            if entry.get('persisted_only'):
                if not self._is_persisted_entry_active(entry):
                    continue
            else:
                age = frame_number - entry['last_frame']
                if age < 0 or age > self.config.REID_BAG_APPEARANCE_MAX_AGE_FRAMES:
                    continue

            if entry.get('appearance') is None and not entry.get('appearance_gallery'):
                continue
            if not self._bag_size_compatible(bbox, entry['bbox']):
                continue

            similarity = self._max_similarity(appearance, entry)

            if entry.get('class_id') != class_id:
                similarity -= self.config.REID_BAG_CLASS_MISMATCH_PENALTY
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
        loaded_person_count = self._load_entries_from_path(self.person_persist_path, self.config.PERSON_CLASS_ID)
        loaded_bag_count = self._load_entries_from_path(self.bag_persist_path, None)
        self.loaded_person_count = loaded_person_count
        self.loaded_bag_count = loaded_bag_count

    def _load_entries_from_path(self, persist_path: Optional[Path], expected_class_id: Optional[int]) -> int:
        if persist_path is None or not persist_path.exists():
            return 0
        try:
            with open(persist_path, 'rb') as file_handle:
                payload = pickle.load(file_handle)
        except Exception as e:
            label = self._persist_label(expected_class_id)
            print(f"⚠️  Failed to load persistent {label} registry: {e}")
            return 0

        loaded_count = 0
        next_person_id = int(payload.get('next_person_id', 1)) if isinstance(payload, dict) else 1
        next_bag_id = int(payload.get('next_bag_id', 1)) if isinstance(payload, dict) else 1
        entries = payload.get('entries', []) if isinstance(payload, dict) else []
        for item in entries:
            try:
                stable_id = int(item['stable_id'])
                class_id = int(item.get('class_id', self.config.PERSON_CLASS_ID))
                if stable_id < self.PERSON_BASE_ID:
                    continue
                if not self._matches_expected_class(class_id, expected_class_id):
                    continue

                appearance_list = item.get('appearance', [])
                appearance = np.asarray(appearance_list, dtype=np.float32) if appearance_list else None
                gallery_arrays = [np.asarray(feature, dtype=np.float32) for feature in item.get('appearance_gallery', [])]
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
                loaded_count += 1
            except Exception:
                continue

        self.next_person_id = max(self.next_person_id, next_person_id)
        self.next_bag_id = max(self.next_bag_id, next_bag_id)
        return loaded_count

    def save_persistent_entries(self):
        self._save_entries_to_path(self.person_persist_path, self.config.PERSON_CLASS_ID)
        self._save_entries_to_path(self.bag_persist_path, None)

    def _save_entries_to_path(self, persist_path: Optional[Path], expected_class_id: Optional[int]):
        if persist_path is None:
            return

        persisted_entries = []
        for stable_id, entry in self.entries.items():
            class_id = int(entry.get('class_id', self.config.PERSON_CLASS_ID))
            if not self._matches_expected_class(class_id, expected_class_id):
                continue
            serialized = self._serialize_entry(stable_id, entry)
            if serialized is not None:
                persisted_entries.append(serialized)

        payload = {
            'entries': persisted_entries,
            'next_person_id': self.next_person_id,
            'next_bag_id': self.next_bag_id,
        }

        try:
            persist_path.parent.mkdir(parents=True, exist_ok=True)
            with open(persist_path, 'wb') as file_handle:
                pickle.dump(payload, file_handle)
            self._write_entry_logs(persisted_entries)
        except Exception as e:
            label = self._persist_label(expected_class_id)
            print(f"⚠️  Failed to save persistent {label} registry: {e}")

    def _write_entry_logs(self, persisted_entries: List[Dict]):
        if not persisted_entries:
            return

        for item in persisted_entries:
            class_id = int(item.get('class_id', self.config.PERSON_CLASS_ID))
            prefix = self._entry_prefix(class_id)
            log_dir = self._entry_log_dir(class_id)
            if log_dir is None:
                continue
            log_dir.mkdir(parents=True, exist_ok=True)
            summary_path = log_dir / f"{prefix}_{item['stable_id']}.json"
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
        return 'person' if class_id == self.config.PERSON_CLASS_ID else 'bag'

    def _entry_log_dir(self, class_id: int) -> Optional[Path]:
        if class_id == self.config.PERSON_CLASS_ID:
            return self.person_log_dir
        if class_id in self.config.BAG_CLASS_IDS:
            return self.bag_log_dir
        return None

    def log_person_frame(
        self,
        stable_id: int,
        frame_number: int,
        frame: np.ndarray,
        bbox: List[float],
        match_meta: Optional[Dict[str, float | str]] = None,
        bt_id: Optional[int] = None,
    ):
        self._log_entity_frame(stable_id, self.config.PERSON_CLASS_ID, frame_number, frame, bbox, match_meta, bt_id)

    def log_bag_frame(
        self,
        stable_id: int,
        frame_number: int,
        frame: np.ndarray,
        bbox: List[float],
        match_meta: Optional[Dict[str, float | str]] = None,
        bt_id: Optional[int] = None,
    ):
        self._log_entity_frame(stable_id, None, frame_number, frame, bbox, match_meta, bt_id)

    def _log_entity_frame(
        self,
        stable_id: int,
        expected_class_id: Optional[int],
        frame_number: int,
        frame: np.ndarray,
        bbox: List[float],
        match_meta: Optional[Dict[str, float | str]] = None,
        bt_id: Optional[int] = None,
    ):
        entry = self.entries.get(stable_id)
        if entry is None:
            return
        if not self._frame_log_allowed(entry, expected_class_id):
            return

        log_dir = self._entry_log_dir(int(entry.get('class_id', self.config.PERSON_CLASS_ID)))
        if log_dir is None or entry.get('last_logged_frame') == frame_number:
            return

        crop, clipped_bbox = self._crop_from_frame(frame, bbox)
        if crop is None or clipped_bbox is None:
            return

        prefix = self._entry_prefix(int(entry.get('class_id', self.config.PERSON_CLASS_ID)))
        entity_dir = log_dir / f"{prefix}_{stable_id}"
        frames_dir = entity_dir / self.config.REID_FRAME_IMAGE_DIRNAME
        frames_dir.mkdir(parents=True, exist_ok=True)

        filename = f"frame_{frame_number:06d}{self.config.REID_FRAME_IMAGE_EXT}"
        image_path = frames_dir / filename
        if not cv2.imwrite(str(image_path), crop):
            return

        metadata_path = entity_dir / self.config.REID_FRAME_METADATA_NAME
        record = self._build_frame_record(entry, frame_number, image_path, clipped_bbox, match_meta, bt_id)
        with open(metadata_path, 'a', encoding='utf-8') as file_handle:
            file_handle.write(json.dumps(record) + "\n")

        entry['saved_frame_count'] = int(entry.get('saved_frame_count', 0)) + 1
        entry['last_logged_frame'] = frame_number