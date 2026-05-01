import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from .config import BGSConfig
from .distance import DistanceEstimator
from .ownership import OwnershipManager
from .reid import BGSRegistry, BagReIDEmbedder, PersonReIDEmbedder
from .visualization import Visualizer


class BagGuardSystem:
    def __init__(
        self,
        model_path: str,
        video_path: str,
        output_path: str,
        imgsz: int,
        max_fps: float,
        skip: int,
        half: bool,
        tracker_backend: str = "bytetrack",
        tracker_profile: str = "stable",
        show: bool = False,
        display_width: int = 1280,
        display_height: int = 720,
    ):
        self.model_path = model_path
        self.video_path = video_path
        self.output_path = output_path
        self.imgsz = int(imgsz)
        self.max_fps = float(max_fps)
        self.skip = max(0, int(skip))
        self.half = bool(half)
        self.show = bool(show)
        self.display_width = max(0, int(display_width))
        self.display_height = max(0, int(display_height))
        self.tracker_backend = str(tracker_backend).strip().lower()
        self.tracker_profile = str(tracker_profile).strip().lower()
        self.config = BGSConfig
        self.window_name = 'BGS - Full Specification'
        resolved_model_path = os.path.abspath(model_path)

        print("\n" + "=" * 80)
        print("BAG GUARD SYSTEM - FULL SPECIFICATION IMPLEMENTATION")
        print("=" * 80)
        print("📋 Specification Compliance:")
        print("  ✓ Section 4: Object Detection (YOLOv8)")
        print("  ✓ Section 5: Stable ID Tracking (DeepSORT / ByteTrack)")
        print("  ✓ Section 6: Distance Estimation (Monocular Trigonometry)")
        print("  ✓ Section 7: Ownership Persistence (Trend-Based Logic)")
        print("  ✓ Section 8: Unattended Detection (3-State: OK/POTENTIAL/UNATTENDED)")
        print("  ✓ Section 9: Frozen Parameters (Reproducibility)")
        print("  ✓ Section 10: Professional Visualization")
        print("=" * 80)
        print(f"Model: {model_path}")
        print(f"Resolved model path: {resolved_model_path}")
        print(f"Video: {video_path}")
        print(f"Output: {output_path}")
        print("=" * 80)

        if not os.path.exists(resolved_model_path):
            print(f"❌ Weights file not found: {resolved_model_path}")
            sys.exit(1)

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

        self.frame_count = 0
        self.inference_frame_count = 0
        self.person_ids_seen = set()
        self.bag_ids_seen = set()
        self.start_time = None
        self.logged_person_frame_count = 0
        self.logged_bag_frame_count = 0
        self.runtime_frame_size: Optional[Tuple[int, int]] = None
        self.last_people_frame = -10_000
        self.last_bags_frame = -10_000
        self.last_people_count = 0
        self.last_bags_count = 0
        self.tracker_path = None
        self.person_tracker = None
        self.bag_trackers: Dict[int, object] = {}
        self._deepsort_track_memory: Dict[int, Dict] = {}
        self._display_window_initialized = False
        self._display_window_size: Optional[Tuple[int, int]] = None

        self._validate_class_config()
        self._configure_trackers()
        self._log_device()
        self._run_startup_inference()

        self.distance_estimator = DistanceEstimator()
        self.ownership_manager = OwnershipManager(self.distance_estimator)
        self.visualizer = Visualizer()
        self.person_reid = PersonReIDEmbedder(self.device, self.config.REID_PERSON_MODEL_NAME, self._resolve_reid_model_path())
        self.bag_reid = BagReIDEmbedder(self.device)
        person_registry_path, person_log_dir, bag_registry_path, bag_log_dir = self._resolve_reid_store_paths()
        self.id_registry = BGSRegistry(person_registry_path, person_log_dir, bag_registry_path, bag_log_dir)

        print("✓ Distance Estimator initialized")
        print("✓ Ownership Manager initialized")
        print("✓ Visualizer initialized")
        if self.person_reid.enabled:
            print("✓ Person re-ID initialized")
        if self.bag_reid.enabled:
            print("✓ Bag re-ID initialized")
        self._log_loaded_registry_counts()

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
        tracker_profile = self.tracker_profile if self.tracker_backend == "bytetrack" else "-"
        print(f"✓ Device: {self.device}")
        print(
            f"✓ Runtime: imgsz={self.imgsz} max_fps={self.max_fps} skip={self.skip} half={self.half} "
            f"tracker_backend={self.tracker_backend} tracker_profile={tracker_profile} "
            f"display={self.display_width}x{self.display_height}"
        )

    def _log_loaded_registry_counts(self):
        if self.id_registry.loaded_person_count:
            print(f"✓ Loaded persistent person logs: {self.id_registry.loaded_person_count}")
        if self.id_registry.loaded_bag_count:
            print(f"✓ Loaded persistent bag logs: {self.id_registry.loaded_bag_count}")

    def _passes_conf_threshold(self, class_id: int, conf: float) -> bool:
        if class_id == self.config.PERSON_CLASS_ID:
            return conf >= self.config.PERSON_CONF
        if class_id in self.config.BAG_CLASS_IDS:
            return conf >= self.config.BAG_CONF
        return False

    @staticmethod
    def _track_id(box) -> Optional[int]:
        return int(box.id[0]) if box.id is not None else None

    @staticmethod
    def _bbox(box) -> List[float]:
        return box.xyxy[0].cpu().numpy().tolist()

    def _display_frame_size(self, frame_w: int, frame_h: int) -> Tuple[int, int]:
        if self.display_width <= 0 and self.display_height <= 0:
            return frame_w, frame_h

        if self.display_width > 0 and self.display_height > 0:
            scale = min(self.display_width / frame_w, self.display_height / frame_h)
        elif self.display_width > 0:
            scale = self.display_width / frame_w
        else:
            scale = self.display_height / frame_h

        display_w = max(1, int(round(frame_w * scale)))
        display_h = max(1, int(round(frame_h * scale)))
        return display_w, display_h

    def _prepare_display_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
        frame_h, frame_w = frame.shape[:2]
        display_w, display_h = self._display_frame_size(frame_w, frame_h)
        if (display_w, display_h) == (frame_w, frame_h):
            return frame, (display_w, display_h)

        interpolation = cv2.INTER_LINEAR if display_w >= frame_w and display_h >= frame_h else cv2.INTER_AREA
        display_frame = cv2.resize(frame, (display_w, display_h), interpolation=interpolation)
        return display_frame, (display_w, display_h)

    def _show_frame(self, frame: np.ndarray):
        display_frame, display_size = self._prepare_display_frame(frame)

        if not self._display_window_initialized:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            self._display_window_initialized = True

        if self._display_window_size != display_size:
            cv2.resizeWindow(self.window_name, display_size[0], display_size[1])
            self._display_window_size = display_size

        cv2.imshow(self.window_name, display_frame)

    @staticmethod
    def _iou(bbox_a: List[float], bbox_b: List[float]) -> float:
        ax1, ay1, ax2, ay2 = bbox_a
        bx1, by1, bx2, by2 = bbox_b
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        intersection = iw * ih
        if intersection <= 0.0:
            return 0.0
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union = area_a + area_b - intersection
        return intersection / union if union > 0.0 else 0.0

    @staticmethod
    def _overlap_coverage(bbox_a: List[float], bbox_b: List[float]) -> float:
        ax1, ay1, ax2, ay2 = bbox_a
        bx1, by1, bx2, by2 = bbox_b
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        intersection = iw * ih
        if intersection <= 0.0:
            return 0.0
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        smaller_area = min(area_a, area_b)
        return intersection / smaller_area if smaller_area > 0.0 else 0.0

    @staticmethod
    def _bbox_center(bbox: List[float]) -> Tuple[float, float]:
        return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)

    @staticmethod
    def _bbox_size_metrics(bbox: List[float]) -> Tuple[float, float, float]:
        width = max(1.0, float(bbox[2] - bbox[0]))
        height = max(1.0, float(bbox[3] - bbox[1]))
        area = width * height
        return width, height, area

    def _bag_duplicate_match_priority(self, detection: Dict) -> int:
        priorities = {
            'track': 4,
            'reid': 3,
            'geo': 2,
            'new': 1,
            'pending': 0,
        }
        return priorities.get(str(detection.get('match_reason', '')), 0)

    def _bag_sort_key(self, detection: Dict):
        return (
            bool(detection.get('pending', False)),
            detection.get('source_index', 0) < 0,
            detection.get('tracker_id') is None,
            -self._bag_duplicate_match_priority(detection),
            -float(detection.get('conf', 0.0)),
        )

    def _is_duplicate_bag_detection(self, detection: Dict, kept: Dict) -> bool:
        if int(detection.get('class_id', -1)) != int(kept.get('class_id', -2)):
            return False

        iou = self._iou(detection['bbox'], kept['bbox'])
        coverage = self._overlap_coverage(detection['bbox'], kept['bbox'])
        if iou >= self.config.BAG_DUPLICATE_IOU_THRESH or coverage >= self.config.BAG_DUPLICATE_COVERAGE_THRESH:
            return True

        dx, dy = self._bbox_center(detection['bbox'])
        kx, ky = self._bbox_center(kept['bbox'])
        centroid_dist = float(np.hypot(dx - kx, dy - ky))
        det_w, det_h, det_area = self._bbox_size_metrics(detection['bbox'])
        kept_w, kept_h, kept_area = self._bbox_size_metrics(kept['bbox'])
        center_limit = self.config.BAG_DUPLICATE_CENTER_FACTOR * min(max(det_w, det_h), max(kept_w, kept_h))
        area_ratio = min(det_area, kept_area) / max(det_area, kept_area)

        return (
            coverage >= self.config.BAG_DUPLICATE_SOFT_COVERAGE_THRESH
            and area_ratio >= self.config.BAG_DUPLICATE_AREA_RATIO_THRESH
            and centroid_dist <= center_limit
        )

    def _deduplicate_bag_detections(self, detections: List[Dict]) -> List[Dict]:
        if not detections:
            return detections

        ordered = sorted(detections, key=self._bag_sort_key)
        kept: List[Dict] = []
        for detection in ordered:
            if any(self._is_duplicate_bag_detection(detection, existing) for existing in kept):
                continue
            kept.append(detection)
        return kept

    @staticmethod
    def _normalize_embedding(embedding: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if embedding is None:
            return None

        vector = np.asarray(embedding, dtype=np.float32)
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm > 0 else None

    @staticmethod
    def _coerce_track_id(track_id) -> Optional[int]:
        if track_id is None:
            return None
        try:
            return int(track_id)
        except (TypeError, ValueError):
            try:
                return int(str(track_id).rsplit("_", 1)[-1])
            except (TypeError, ValueError):
                return None

    @staticmethod
    def _namespace_track_id(namespace: int, track_id: Optional[int]) -> Optional[int]:
        if track_id is None:
            return None
        return (int(namespace) * 1_000_000) + int(track_id)

    def _build_detection_input(
        self,
        source_index: int,
        class_id: int,
        bbox: List[float],
        conf: float,
        appearance: Optional[np.ndarray],
        tracker_id: Optional[int] = None,
    ) -> Dict:
        return {
            'source_index': int(source_index),
            'class_id': int(class_id),
            'bbox': [float(value) for value in bbox],
            'conf': float(conf),
            'appearance': self._normalize_embedding(appearance),
            'tracker_id': tracker_id,
        }

    def _build_detection_inputs(self, boxes, frame: np.ndarray) -> List[Dict]:
        detections: List[Dict] = []
        if boxes is None or len(boxes) == 0:
            return detections

        person_appearances = self._collect_person_appearances(boxes, frame)
        bag_appearances = self._collect_bag_appearances(boxes, frame)

        for i in range(len(boxes)):
            class_id = int(boxes[i].cls[0])
            conf = float(boxes[i].conf[0])
            if class_id not in self.config.CLASS_IDS or not self._passes_conf_threshold(class_id, conf):
                continue

            appearance = person_appearances.get(i) if class_id == self.config.PERSON_CLASS_ID else bag_appearances.get(i)
            detections.append(
                self._build_detection_input(
                    source_index=i,
                    class_id=class_id,
                    bbox=self._bbox(boxes[i]),
                    conf=conf,
                    appearance=appearance,
                    tracker_id=self._track_id(boxes[i]),
                )
            )

        return self._suppress_duplicate_bag_detections(detections)

    def _suppress_duplicate_bag_detections(self, detections: List[Dict]) -> List[Dict]:
        if not detections:
            return detections

        people = [detection for detection in detections if detection['class_id'] == self.config.PERSON_CLASS_ID]
        bag_candidates = [detection for detection in detections if detection['class_id'] in self.config.BAG_CLASS_IDS]
        kept_bags = self._deduplicate_bag_detections(bag_candidates)
        return people + kept_bags

    def _deepsort_kwargs(self, max_age: int, max_cosine_distance: float) -> Dict:
        return {
            'max_iou_distance': self.config.DEEPSORT_MAX_IOU_DISTANCE,
            'max_age': max_age,
            'n_init': self.config.DEEPSORT_N_INIT,
            'nms_max_overlap': self.config.DEEPSORT_NMS_MAX_OVERLAP,
            'max_cosine_distance': max_cosine_distance,
            'nn_budget': self.config.DEEPSORT_NN_BUDGET,
            'embedder': None,
            'gating_only_position': False,
        }

    def _deepsort_output_max_age(self, class_id: int) -> int:
        if class_id == self.config.PERSON_CLASS_ID:
            return self.config.DEEPSORT_PERSON_OUTPUT_MAX_AGE
        return self.config.DEEPSORT_BAG_OUTPUT_MAX_AGE

    def _prune_deepsort_track_memory(self):
        max_age = max(self.config.DEEPSORT_PERSON_MAX_AGE, self.config.DEEPSORT_BAG_MAX_AGE)
        min_frame = self.frame_count - max_age - 2
        self._deepsort_track_memory = {
            track_id: entry
            for track_id, entry in self._deepsort_track_memory.items()
            if int(entry.get('last_frame', min_frame)) >= min_frame
        }

    def _configure_trackers(self):
        if self.tracker_backend == "bytetrack":
            self.tracker_path = self._resolve_tracker_path()
            print(f"✓ Tracking backend loaded: ByteTrack ({self.tracker_profile})")
            return

        if self.tracker_backend == "deepsort":
            self._init_deepsort_trackers()
            print("✓ Tracking backend loaded: DeepSORT")
            return

        print(f"❌ Invalid tracker backend: {self.tracker_backend}")
        print("Allowed backends: deepsort, bytetrack")
        sys.exit(1)

    def _init_deepsort_trackers(self):
        try:
            from deep_sort_realtime.deepsort_tracker import DeepSort
        except Exception as e:
            print(f"❌ DeepSORT import failed: {e}")
            print("Install dependency: deep-sort-realtime")
            sys.exit(1)

        self.person_tracker = DeepSort(
            **self._deepsort_kwargs(
                self.config.DEEPSORT_PERSON_MAX_AGE,
                self.config.DEEPSORT_PERSON_MAX_COSINE_DISTANCE,
            )
        )
        self.bag_trackers = {
            class_id: DeepSort(
                **self._deepsort_kwargs(
                    self.config.DEEPSORT_BAG_MAX_AGE,
                    self.config.DEEPSORT_BAG_MAX_COSINE_DISTANCE,
                )
            )
            for class_id in self.config.BAG_CLASS_IDS
        }

    def _run_deepsort_tracker(self, tracker, detections: List[Dict], namespace: int, output_max_age: int) -> Tuple[Dict[int, Dict], List[Dict]]:
        detection_map = {int(det['source_index']): det for det in detections}
        raw_detections = []
        embeddings = []
        others = []

        for detection in detections:
            appearance = detection.get('appearance')
            if appearance is None:
                continue

            x1, y1, x2, y2 = detection['bbox']
            raw_detections.append((
                [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                float(detection['conf']),
                self.config.TARGET_CLASSES[detection['class_id']],
            ))
            embeddings.append(appearance)
            others.append({'source_index': int(detection['source_index'])})

        tracks = tracker.update_tracks(raw_detections, embeds=embeddings, others=others)
        tracked_by_source: Dict[int, Dict] = {}
        predicted_tracks: List[Dict] = []

        for track in tracks:
            if not track.is_confirmed():
                continue

            local_track_id = self._coerce_track_id(track.track_id)
            if local_track_id is None:
                continue

            namespaced_track_id = self._namespace_track_id(namespace, local_track_id)
            if namespaced_track_id is None:
                continue

            bbox = track.to_ltrb(orig=track.time_since_update == 0, orig_strict=False)
            if bbox is None:
                continue

            if track.time_since_update != 0:
                if track.time_since_update > output_max_age:
                    continue

                cached_track = self._deepsort_track_memory.get(namespaced_track_id)
                if cached_track is None:
                    continue

                predicted_tracks.append(
                    self._build_detection_input(
                        source_index=-(namespaced_track_id + track.time_since_update),
                        class_id=int(cached_track['class_id']),
                        bbox=np.asarray(bbox, dtype=np.float32).tolist(),
                        conf=float(cached_track.get('conf', 0.0)),
                        appearance=cached_track.get('appearance'),
                        tracker_id=namespaced_track_id,
                    )
                )
                continue

            supplementary = track.get_det_supplementary() or {}
            source_index = supplementary.get('source_index')
            if source_index is None:
                continue

            source_index = int(source_index)
            base_detection = detection_map.get(source_index)
            if base_detection is None:
                continue

            appearance = self._normalize_embedding(track.get_feature())
            if appearance is None:
                appearance = base_detection.get('appearance')

            conf = float(track.get_det_conf() or base_detection['conf'])
            tracked_by_source[source_index] = self._build_detection_input(
                source_index=source_index,
                class_id=base_detection['class_id'],
                bbox=np.asarray(bbox, dtype=np.float32).tolist(),
                conf=conf,
                appearance=appearance,
                tracker_id=namespaced_track_id,
            )
            self._deepsort_track_memory[namespaced_track_id] = {
                'class_id': int(base_detection['class_id']),
                'conf': conf,
                'appearance': appearance.copy() if appearance is not None else None,
                'last_frame': self.frame_count,
            }

        return tracked_by_source, predicted_tracks

    def _apply_deepsort_tracking(self, detections: List[Dict]) -> List[Dict]:
        tracked_by_source: Dict[int, Dict] = {}
        predicted_tracks: List[Dict] = []

        person_detections = [
            detection for detection in detections
            if detection['class_id'] == self.config.PERSON_CLASS_ID and detection.get('appearance') is not None
        ]
        person_tracked, person_predicted = self._run_deepsort_tracker(
            self.person_tracker,
            person_detections,
            self.config.PERSON_CLASS_ID + 1,
            self._deepsort_output_max_age(self.config.PERSON_CLASS_ID),
        )
        tracked_by_source.update(person_tracked)
        predicted_tracks.extend(person_predicted)

        for class_id in self.config.BAG_CLASS_IDS:
            class_detections = [
                detection for detection in detections
                if detection['class_id'] == class_id and detection.get('appearance') is not None
            ]
            class_tracked, class_predicted = self._run_deepsort_tracker(
                self.bag_trackers[class_id],
                class_detections,
                class_id + 100,
                self._deepsort_output_max_age(class_id),
            )
            tracked_by_source.update(class_tracked)
            predicted_tracks.extend(class_predicted)

        self._prune_deepsort_track_memory()
        current_tracks = [tracked_by_source.get(int(detection['source_index']), detection) for detection in detections]
        return current_tracks + predicted_tracks

    def _detect_and_track(self, frame: np.ndarray) -> List[Dict]:
        common_kwargs = {
            'conf': min(self.config.PERSON_CONF, self.config.BAG_CONF),
            'iou': self.config.IOU_THRESHOLD,
            'classes': self.config.CLASS_IDS,
            'imgsz': self.imgsz,
            'device': self.device,
            'half': self.half,
            'verbose': False,
        }

        if self.tracker_backend == "bytetrack":
            results = self.model.track(
                frame,
                persist=self.config.PERSIST,
                tracker=self.tracker_path,
                **common_kwargs,
            )
            return self._build_detection_inputs(results[0].boxes, frame)

        results = self.model.predict(frame, **common_kwargs)
        detections = self._build_detection_inputs(results[0].boxes, frame)
        return self._apply_deepsort_tracking(detections)

    def _build_detection(
        self,
        stable_id: int,
        class_id: int,
        bbox: List[float],
        conf: float,
        position_3d: Tuple[float, float, float],
        match_meta: Dict[str, float | str],
        **extra,
    ) -> Dict:
        detection = {
            'id': stable_id,
            'class': self.config.TARGET_CLASSES[class_id],
            'class_id': class_id,
            'bbox': bbox,
            'conf': conf,
            'position_3d': position_3d,
            'match_reason': str(match_meta.get('reason', '')),
            'match_score': float(match_meta.get('score', 0.0)),
        }
        detection.update(extra)
        return detection

    @staticmethod
    def _empty_stats(frame_number: int) -> Dict:
        return {
            'fps': 0,
            'people_count': 0,
            'bags_count': 0,
            'bags_ok': 0,
            'bags_potential': 0,
            'bags_unattended': 0,
            'frame_number': frame_number,
            'tracker_backend': '',
        }

    def _set_runtime_frame_geometry(self, frame: np.ndarray):
        frame_h, frame_w = frame.shape[:2]
        if frame_w <= 0 or frame_h <= 0:
            return

        runtime_size = (int(frame_w), int(frame_h))
        if self.runtime_frame_size == runtime_size:
            return

        self.runtime_frame_size = runtime_size
        self.config.IMAGE_WIDTH = runtime_size[0]
        self.config.IMAGE_HEIGHT = runtime_size[1]
        self.config.FOCAL_LENGTH = (self.config.IMAGE_WIDTH / 2) / np.tan(np.radians(self.config.CAMERA_HFOV / 2))
        print(f"✓ Frame geometry: {self.config.IMAGE_WIDTH}x{self.config.IMAGE_HEIGHT}")

    def _passes_rescue_threshold(self, class_id: int, conf: float) -> bool:
        if class_id == self.config.PERSON_CLASS_ID:
            return conf >= self.config.DETECTION_RESCUE_PERSON_CONF
        if class_id in self.config.BAG_CLASS_IDS:
            return conf >= self.config.DETECTION_RESCUE_BAG_CONF
        return False

    def _recently_saw_people(self) -> bool:
        return self.frame_count - self.last_people_frame <= self.config.DETECTION_RESCUE_RECENT_FRAMES

    def _recently_saw_bags(self) -> bool:
        return self.frame_count - self.last_bags_frame <= self.config.DETECTION_RESCUE_RECENT_FRAMES

    def _rescue_missing_detections(self, frame: np.ndarray, people: List[Dict], bags: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        if not self.config.DETECTION_RESCUE_ENABLED or self.tracker_backend != "bytetrack":
            return people, bags

        rescue_people = self._recently_saw_people() and len(people) < self.last_people_count
        rescue_bags = self._recently_saw_bags() and len(bags) < self.last_bags_count
        if not rescue_people and not rescue_bags:
            return people, bags

        rescue_results = self.model.predict(
            frame,
            conf=min(self.config.DETECTION_RESCUE_PERSON_CONF, self.config.DETECTION_RESCUE_BAG_CONF),
            iou=self.config.IOU_THRESHOLD,
            classes=self.config.CLASS_IDS,
            imgsz=self.imgsz,
            device=self.device,
            half=self.half,
            verbose=False,
        )

        rescue_boxes = rescue_results[0].boxes
        if rescue_boxes is None or len(rescue_boxes) == 0:
            return people, bags

        person_appearances = self._collect_person_appearances(rescue_boxes, frame) if rescue_people else {}
        bag_appearances = self._collect_bag_appearances(rescue_boxes, frame) if rescue_bags else {}

        if rescue_people:
            for i in range(len(rescue_boxes)):
                cls_id = int(rescue_boxes[i].cls[0])
                conf = float(rescue_boxes[i].conf[0])
                if cls_id != self.config.PERSON_CLASS_ID or not self._passes_rescue_threshold(cls_id, conf):
                    continue
                people.append(
                    self._resolve_person_detection(
                        self._build_detection_input(
                            source_index=i,
                            class_id=cls_id,
                            bbox=self._bbox(rescue_boxes[i]),
                            conf=conf,
                            appearance=person_appearances.get(i),
                        ),
                        frame,
                    )
                )

        if rescue_bags:
            for i in range(len(rescue_boxes)):
                cls_id = int(rescue_boxes[i].cls[0])
                conf = float(rescue_boxes[i].conf[0])
                if cls_id not in self.config.BAG_CLASS_IDS or not self._passes_rescue_threshold(cls_id, conf):
                    continue
                bags.append(
                    self._resolve_bag_detection(
                        self._build_detection_input(
                            source_index=i,
                            class_id=cls_id,
                            bbox=self._bbox(rescue_boxes[i]),
                            conf=conf,
                            appearance=bag_appearances.get(i),
                        ),
                        frame,
                        people,
                    )
                )

        return people, bags

    def _collect_person_appearances(self, boxes, frame: np.ndarray) -> Dict[int, np.ndarray]:
        person_appearances: Dict[int, np.ndarray] = {}
        if not self.person_reid.enabled:
            return person_appearances

        person_indices = []
        person_crops = []
        for i in range(len(boxes)):
            cls_id = int(boxes[i].cls[0])
            conf = float(boxes[i].conf[0])
            if cls_id != self.config.PERSON_CLASS_ID or not self._passes_conf_threshold(cls_id, conf):
                continue
            crop = self.person_reid.crop_person(frame, self._bbox(boxes[i]))
            if crop is None:
                continue
            person_indices.append(i)
            person_crops.append(crop)

        for idx, embedding in zip(person_indices, self.person_reid.extract(person_crops)):
            if embedding is not None:
                person_appearances[idx] = embedding
        return person_appearances

    def _collect_bag_appearances(self, boxes, frame: np.ndarray) -> Dict[int, np.ndarray]:
        bag_appearances: Dict[int, np.ndarray] = {}
        if not self.bag_reid.enabled:
            return bag_appearances

        for i in range(len(boxes)):
            cls_id = int(boxes[i].cls[0])
            conf = float(boxes[i].conf[0])
            if cls_id not in self.config.BAG_CLASS_IDS or not self._passes_conf_threshold(cls_id, conf):
                continue
            crop = self.bag_reid.crop_bag(frame, self._bbox(boxes[i]))
            embedding = self.bag_reid.extract_one(crop)
            if embedding is not None:
                bag_appearances[i] = embedding
        return bag_appearances

    def _resolve_person_detection(self, detection: Dict, frame: np.ndarray) -> Dict:
        cls_id = int(detection['class_id'])
        bbox = detection['bbox']
        conf = float(detection['conf'])
        bt_id = detection.get('tracker_id')
        appearance = detection.get('appearance')
        stable_id, match_meta = self.id_registry.resolve(bbox, cls_id, self.frame_count, bt_id, appearance=appearance)
        position_3d = self.distance_estimator.estimate_position_3d(bbox, is_person=True)
        self.id_registry.log_person_frame(stable_id, self.frame_count, frame, bbox, match_meta=match_meta, bt_id=bt_id)
        self.logged_person_frame_count += 1
        self.person_ids_seen.add(stable_id)
        return self._build_detection(stable_id, cls_id, bbox, conf, position_3d, match_meta)

    def _resolve_bag_detection(
        self,
        detection: Dict,
        frame: np.ndarray,
        people: List[Dict],
    ) -> Dict:
        cls_id = int(detection['class_id'])
        bbox = detection['bbox']
        conf = float(detection['conf'])
        appearance = detection.get('appearance')
        owner_hint = self._estimate_bag_owner_hint(bbox, people)
        bt_id = detection.get('tracker_id')
        stable_id, match_meta = self.id_registry.resolve(
            bbox,
            cls_id,
            self.frame_count,
            bt_id,
            appearance=appearance,
            owner_hint=owner_hint,
        )
        position_3d = self.distance_estimator.estimate_position_3d(bbox, is_person=False)
        pending = stable_id < 0 or str(match_meta.get('reason', '')) == 'pending'
        if not pending:
            self.id_registry.log_bag_frame(stable_id, self.frame_count, frame, bbox, match_meta=match_meta, bt_id=bt_id)
            self.logged_bag_frame_count += 1
            self.bag_ids_seen.add(stable_id)
        return self._build_detection(
            stable_id,
            cls_id,
            bbox,
            conf,
            position_3d,
            match_meta,
            owner_hint=owner_hint,
            pending=pending,
            pending_frames_left=int(match_meta.get('frames_left', 0)),
        )

    @staticmethod
    def _count_bag_states(bag_states) -> Dict[str, int]:
        status_counts = {'OK': 0, 'POTENTIAL': 0, 'UNATTENDED': 0}
        for state in bag_states.values():
            status_counts[state.status] += 1
        return status_counts

    def _draw_scene(self, frame: np.ndarray, people: List[Dict], bags: List[Dict], bag_states: Dict):
        people_by_id = {person['id']: person for person in people}
        for person in people:
            self.visualizer.draw_detection(frame, person)
        for bag in bags:
            bag_state = bag_states.get(bag['id'])
            self.visualizer.draw_detection(frame, bag, bag_state)
            owner = people_by_id.get(bag_state.owner_id) if bag_state and bag_state.owner_id else None
            if owner is None:
                continue
            distance = self.distance_estimator.calculate_distance(bag['position_3d'], owner['position_3d'])
            self.visualizer.draw_distance_line(frame, bag['bbox'], owner['bbox'], distance)

    def _handle_persistence_checkpoint(self, stats: Dict, total_frames: int):
        if self.frame_count % self.config.REID_PERSIST_INTERVAL_FRAMES == 0:
            self.id_registry.save_persistent_entries()
        if total_frames > 0:
            progress = (self.frame_count / total_frames) * 100
            prefix = f"Progress: {progress:5.1f}% | Frame: {self.frame_count:5d}/{total_frames}"
        else:
            prefix = f"LIVE | Frame: {self.frame_count:5d}"

        print(f"{prefix} | FPS: {stats['fps']:3d} | People: {stats['people_count']} | Bags: {stats['bags_count']} | Unattended: {stats['bags_unattended']}")

    def _process_current_frame(self, frame: np.ndarray, prev_time: float) -> Tuple[np.ndarray, Dict, float]:
        self.inference_frame_count += 1
        current_time = time.time()
        annotated_frame, stats = self.process_frame(frame)
        stats['fps'] = int(1 / (current_time - prev_time)) if current_time > prev_time else 0
        self.visualizer.draw_debug_overlay(annotated_frame, stats)
        return annotated_frame, stats, current_time

    def _resolve_tracker_path(self) -> str:
        tracker_value = str(self.config.TRACKER).strip()
        root = Path(__file__).resolve().parents[2]

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

        root = Path(__file__).resolve().parents[2]
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

    def _resolve_reid_store_paths(self) -> Tuple[Path, Path, Path, Path]:
        root = Path(__file__).resolve().parents[2]
        person_persist_path = root / self.config.REID_PERSON_PERSIST_PATH
        person_log_dir = root / self.config.REID_PERSON_PERSIST_LOG_DIR
        bag_persist_path = root / self.config.REID_BAG_PERSIST_PATH
        bag_log_dir = root / self.config.REID_BAG_PERSIST_LOG_DIR
        return person_persist_path, person_log_dir, bag_persist_path, bag_log_dir

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

        self._set_runtime_frame_geometry(frame)
        results = self.model.predict(
            frame,
            conf=self.config.DETECTION_CONFIDENCE,
            iou=self.config.IOU_THRESHOLD,
            classes=self.config.CLASS_IDS,
            imgsz=self.imgsz,
            device=self.device,
            half=self.half,
            verbose=False,
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

    def extract_detections(self, detections: List[Dict], frame: np.ndarray) -> Tuple[List[Dict], List[Dict]]:
        people = []
        bags = []
        if not detections:
            return people, bags

        for detection in detections:
            cls_id = int(detection['class_id'])
            if cls_id != self.config.PERSON_CLASS_ID:
                continue

            conf = float(detection['conf'])
            if not self._passes_conf_threshold(cls_id, conf):
                continue

            people.append(self._resolve_person_detection(detection, frame))

        for detection in detections:
            cls_id = int(detection['class_id'])
            if cls_id not in self.config.CLASS_IDS or cls_id == self.config.PERSON_CLASS_ID:
                continue

            conf = float(detection['conf'])
            if not self._passes_conf_threshold(cls_id, conf):
                continue

            bags.append(self._resolve_bag_detection(detection, frame, people))

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
                bag['position_3d'] = self.distance_estimator.estimate_position_3d(bag['bbox'], is_person=False, reference_depth=best_depth)

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict]:
        current_time = time.time()
        detections = self._detect_and_track(frame)
        people, bags = self.extract_detections(detections, frame)
        people, bags = self._rescue_missing_detections(frame, people, bags)
        bags = self._deduplicate_bag_detections(bags)
        if people:
            self.last_people_frame = self.frame_count
            self.last_people_count = len(people)
        else:
            self.last_people_count = 0
        if bags:
            self.last_bags_frame = self.frame_count
            self.last_bags_count = len(bags)
        else:
            self.last_bags_count = 0
        confirmed_bags = [bag for bag in bags if not bag.get('pending')]
        self._refine_bag_depths(confirmed_bags, people)

        bag_states = self.ownership_manager.update_ownership(confirmed_bags, people, current_time)
        for bag_id, bag_state in bag_states.items():
            self.id_registry.update_bag_owner(bag_id, bag_state.owner_id)

        status_counts = self._count_bag_states(bag_states)
        self._draw_scene(frame, people, bags, bag_states)

        stats = {
            'fps': 0,
            'people_count': len(people),
            'bags_count': len(bags),
            'bags_ok': status_counts['OK'],
            'bags_potential': status_counts['POTENTIAL'],
            'bags_unattended': status_counts['UNATTENDED'],
            'frame_number': self.frame_count,
            'tracker_backend': self.tracker_backend,
        }
        return frame, stats

    def run(self) -> bool:
        cap = cv2.VideoCapture(self.video_path)
        cv2.setUseOptimized(True)
        if not cap.isOpened():
            print("❌ Cannot open video")
            return False

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"\n✓ Video: {total_frames} frames @ {fps} FPS")

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = None

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

                if out is None:
                    self._set_runtime_frame_geometry(frame)
                    output_fps = fps if fps > 0 else 30
                    out = cv2.VideoWriter(self.output_path, fourcc, output_fps, (self.config.IMAGE_WIDTH, self.config.IMAGE_HEIGHT))
                    print("✓ Starting processing...\n")

                do_infer = (self.skip == 0) or (read_count % (self.skip + 1) == 1)
                if do_infer:
                    annotated_frame, stats, prev_time = self._process_current_frame(frame, prev_time)
                    last_annotated_frame = annotated_frame
                    last_stats = stats
                else:
                    annotated_frame = last_annotated_frame if last_annotated_frame is not None else frame
                    stats = last_stats if last_stats is not None else self._empty_stats(self.frame_count)

                out.write(annotated_frame)
                if self.show:
                    self._show_frame(annotated_frame)

                if do_infer and self.frame_count % 30 == 0:
                    self._handle_persistence_checkpoint(stats, total_frames)

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
            if out is not None:
                out.release()
            cv2.destroyAllWindows()

        elapsed = time.time() - self.start_time
        print(f"\n{'=' * 80}")
        print("PROCESSING COMPLETE")
        print(f"{'=' * 80}")
        print(f"Frames: {self.frame_count} | Time: {elapsed:.2f}s | Avg FPS: {self.frame_count / elapsed:.2f}")
        print(f"Person IDs: {len(self.person_ids_seen)} | Bag IDs: {len(self.bag_ids_seen)}")
        print(f"Output: {self.output_path}")
        print(f"{'=' * 80}\n")
        return True