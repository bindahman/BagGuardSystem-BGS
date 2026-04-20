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
        self.tracker_profile = str(tracker_profile).strip().lower()
        self.config = BGSConfig
        resolved_model_path = os.path.abspath(model_path)

        print("\n" + "=" * 80)
        print("BAG GUARD SYSTEM - FULL SPECIFICATION IMPLEMENTATION")
        print("=" * 80)
        print("📋 Specification Compliance:")
        print("  ✓ Section 4: Object Detection (YOLOv8)")
        print("  ✓ Section 5: Stable ID Tracking (ByteTrack)")
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
        self.tracker_path = self._resolve_tracker_path()
        self._log_device()
        self._run_startup_inference()

        self.distance_estimator = DistanceEstimator()
        self.ownership_manager = OwnershipManager(self.distance_estimator)
        self.visualizer = Visualizer()
        self.person_reid = PersonReIDEmbedder(self.device, self.config.REID_PERSON_MODEL_NAME, self._resolve_reid_model_path())
        self.bag_reid = BagReIDEmbedder()
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
        print(f"✓ Device: {self.device}")
        print(f"✓ Runtime: imgsz={self.imgsz} max_fps={self.max_fps} skip={self.skip} half={self.half} tracker_profile={self.tracker_profile}")

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

    def _resolve_person_detection(self, box, frame: np.ndarray, appearance: Optional[np.ndarray]) -> Dict:
        cls_id = int(box.cls[0])
        bbox = self._bbox(box)
        conf = float(box.conf[0])
        bt_id = self._track_id(box)
        stable_id, match_meta = self.id_registry.resolve(bbox, cls_id, self.frame_count, bt_id, appearance=appearance)
        position_3d = self.distance_estimator.estimate_position_3d(bbox, is_person=True)
        self.id_registry.log_person_frame(stable_id, self.frame_count, frame, bbox, match_meta=match_meta, bt_id=bt_id)
        self.logged_person_frame_count += 1
        self.person_ids_seen.add(stable_id)
        return self._build_detection(stable_id, cls_id, bbox, conf, position_3d, match_meta)

    def _resolve_bag_detection(
        self,
        box,
        frame: np.ndarray,
        people: List[Dict],
        appearance: Optional[np.ndarray],
    ) -> Dict:
        cls_id = int(box.cls[0])
        bbox = self._bbox(box)
        conf = float(box.conf[0])
        owner_hint = self._estimate_bag_owner_hint(bbox, people)
        bt_id = self._track_id(box)
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

    def extract_detections(self, results, frame: np.ndarray) -> Tuple[List[Dict], List[Dict]]:
        people = []
        bags = []
        if results[0].boxes is None or len(results[0].boxes) == 0:
            return people, bags

        boxes = results[0].boxes
        person_appearances = self._collect_person_appearances(boxes, frame)
        bag_appearances = self._collect_bag_appearances(boxes, frame)

        for i in range(len(boxes)):
            cls_id = int(boxes[i].cls[0])
            if cls_id != self.config.PERSON_CLASS_ID:
                continue

            conf = float(boxes[i].conf[0])
            if not self._passes_conf_threshold(cls_id, conf):
                continue

            people.append(self._resolve_person_detection(boxes[i], frame, person_appearances.get(i)))

        for i in range(len(boxes)):
            cls_id = int(boxes[i].cls[0])
            if cls_id not in self.config.CLASS_IDS or cls_id == self.config.PERSON_CLASS_ID:
                continue

            conf = float(boxes[i].conf[0])
            if not self._passes_conf_threshold(cls_id, conf):
                continue

            bags.append(self._resolve_bag_detection(boxes[i], frame, people, bag_appearances.get(i)))

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
            verbose=False,
        )

        people, bags = self.extract_detections(results, frame)
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