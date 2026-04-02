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
        tracker_backend: str = "deepsort",
        tracker_profile: str = "main",
        show: bool = False,
    ):
        self.model_path = model_path
        self.video_path = video_path
        self.output_path = output_path
        self.imgsz = int(imgsz)
        self.max_fps = float(max_fps)
        self.skip = max(0, int(skip))
        self.half = bool(half)
        self.show = bool(show)
        self.tracker_backend = str(tracker_backend).strip().lower()
        self.tracker_profile = str(tracker_profile).strip().lower()
        self.config = BGSConfig
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

        self._validate_class_config()
        self.tracker_path = None
        self.person_tracker = None
        self.bag_trackers = {}
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
        tracker_profile = self.tracker_profile if self.tracker_backend == "bytetrack" else "-"
        print(f"✓ Device: {self.device}")
        print(f"✓ Runtime: imgsz={self.imgsz} max_fps={self.max_fps} skip={self.skip} half={self.half} tracker_backend={self.tracker_backend} tracker_profile={tracker_profile}")

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

        return detections

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

    def _run_deepsort_tracker(self, tracker, detections: List[Dict], namespace: int) -> Dict[int, Dict]:
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

        for track in tracks:
            if track.time_since_update != 0:
                continue

            supplementary = track.get_det_supplementary() or {}
            source_index = supplementary.get('source_index')
            if source_index is None:
                continue

            source_index = int(source_index)
            base_detection = detection_map.get(source_index)
            if base_detection is None:
                continue

            bbox = track.to_ltrb(orig=True, orig_strict=True)
            local_track_id = self._coerce_track_id(track.track_id)
            if bbox is None or local_track_id is None:
                continue

            appearance = self._normalize_embedding(track.get_feature())
            if appearance is None:
                appearance = base_detection.get('appearance')

            tracked_by_source[source_index] = self._build_detection_input(
                source_index=source_index,
                class_id=base_detection['class_id'],
                bbox=np.asarray(bbox, dtype=np.float32).tolist(),
                conf=float(track.get_det_conf() or base_detection['conf']),
                appearance=appearance,
                tracker_id=self._namespace_track_id(namespace, local_track_id),
            )

        return tracked_by_source

    def _apply_deepsort_tracking(self, detections: List[Dict]) -> List[Dict]:
        tracked_by_source: Dict[int, Dict] = {}

        person_detections = [
            detection for detection in detections
            if detection['class_id'] == self.config.PERSON_CLASS_ID and detection.get('appearance') is not None
        ]
        tracked_by_source.update(self._run_deepsort_tracker(self.person_tracker, person_detections, self.config.PERSON_CLASS_ID + 1))

        for class_id in self.config.BAG_CLASS_IDS:
            class_detections = [
                detection for detection in detections
                if detection['class_id'] == class_id and detection.get('appearance') is not None
            ]
            tracked_by_source.update(self._run_deepsort_tracker(self.bag_trackers[class_id], class_detections, class_id + 100))

        return [tracked_by_source.get(int(detection['source_index']), detection) for detection in detections]

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
        }

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

        bag_indices = []
        bag_crops = []
        for i in range(len(boxes)):
            cls_id = int(boxes[i].cls[0])
            conf = float(boxes[i].conf[0])
            if cls_id not in self.config.BAG_CLASS_IDS or not self._passes_conf_threshold(cls_id, conf):
                continue
            crop = self.bag_reid.crop_bag(frame, self._bbox(boxes[i]))
            if crop is None:
                continue
            bag_indices.append(i)
            bag_crops.append(crop)

        for idx, embedding in zip(bag_indices, self.bag_reid.extract(bag_crops)):
            if embedding is not None:
                bag_appearances[idx] = embedding
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

        frame = cv2.resize(frame, (self.config.IMAGE_WIDTH, self.config.IMAGE_HEIGHT))
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
        self._refine_bag_depths(bags, people)

        bag_states = self.ownership_manager.update_ownership(bags, people, current_time)
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
        out = cv2.VideoWriter(self.output_path, fourcc, fps, (self.config.IMAGE_WIDTH, self.config.IMAGE_HEIGHT))
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
                    annotated_frame, stats, prev_time = self._process_current_frame(frame, prev_time)
                    last_annotated_frame = annotated_frame
                    last_stats = stats
                else:
                    annotated_frame = last_annotated_frame if last_annotated_frame is not None else frame
                    stats = last_stats if last_stats is not None else self._empty_stats(self.frame_count)

                out.write(annotated_frame)
                if self.show:
                    cv2.imshow('BGS - Full Specification', annotated_frame)

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