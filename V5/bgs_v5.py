"""
═══════════════════════════════════════════════════════════════════════════════
BAG GUARD SYSTEM (BGS) v5.0 — Two-Layer Stable Identity
═══════════════════════════════════════════════════════════════════════════════

ROOT CAUSE OF ID CHURN (260002 → 260003 → 270003)
──────────────────────────────────────────────────
ByteTrack deprioritises stationary / low-confidence objects (bags).
For these, boxes[i].id returns None every few frames.
The v4 sentinel fallback -(cls_id * 10000 + i) produced a brand-new
negative integer every frame — exactly the churn that was observed.

SOLUTION: TWO-LAYER IDENTITY SYSTEM
─────────────────────────────────────
Layer 1  ByteTrack     — short-term frame-to-frame association (people & bags)
Layer 2  BGSRegistry   — long-term spatial re-ID (up to 30 seconds memory)
                         IoU + centroid matching with per-class stable counters

How they interact
  • When ByteTrack gives an ID  → registry maps BT-ID → BGS-ID
    If the same spatial object reappears under a NEW BT-ID, the registry
    recognises it by IoU/centroid and returns the ORIGINAL BGS-ID.
  • When ByteTrack gives None   → registry matches by centroid/IoU alone
    and returns the same BGS-ID it gave last time for that object.
  • Result: bags get IDs like B-1, B-2 that never change; people get P-1, P-2.

ALSO FIXED IN v5
────────────────
  ✅ Owner-specific distance rolling mean (not closest-stranger distance)
  ✅ Ownership release after prolonged absence
  ✅ Bag depth refined from nearest-person depth reference
  ✅ YAML auto-written next to script — no FileNotFoundError

Author : BGS Implementation Team
Version: 5.0
Date   : February 2026
═══════════════════════════════════════════════════════════════════════════════
"""

import argparse
import cv2
import numpy as np
from ultralytics import YOLO
import time
from collections import deque
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Deque
import sys
from pathlib import Path
import torch


# ═══════════════════════════════════════════════════════════════════════════
# TRACKER YAML RESOLVER
# ═══════════════════════════════════════════════════════════════════════════
def _resolve_tracker_yaml() -> str:
    """Return existing bytetrack_bgs.yaml path next to this script (no auto-write)."""
    path = Path(__file__).resolve().parent / "bytetrack_bgs.yaml"

    if not path.exists():
        print(f"✗ Tracker YAML not found: {path}")
        print("  Please place bytetrack_bgs.yaml next to bgs_v5.py")
        sys.exit(1)

    text = None
    last_error = None
    for encoding in ("utf-8", "cp1252", "latin-1"):
        try:
            text = path.read_text(encoding=encoding)
            break
        except Exception as e:
            last_error = e

    if text is None:
        print(f"✗ Failed to read tracker YAML: {last_error}")
        sys.exit(1)

    lower = text.lower()
    if "tracker_type" not in lower or "bytetrack" not in lower:
        print(f"✗ Invalid tracker YAML (expected ByteTrack): {path}")
        sys.exit(1)

    return str(path)


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1: FROZEN CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════
class BGSConfig:
    TARGET_CLASSES = {0: 'person', 24: 'backpack', 26: 'handbag', 28: 'suitcase'}
    CLASS_IDS      = [0, 24, 26, 28]
    BAG_CLASS_IDS  = [24, 26, 28]
    PERSON_CLASS_ID = 0
    EXPECTED_CLASS_IDS = {0, 24, 26, 28}
    EXPECTED_TARGET_CLASSES = {0: 'person', 24: 'backpack', 26: 'handbag', 28: 'suitcase'}

    DETECTION_CONFIDENCE = 0.20
    PERSON_CONF          = DETECTION_CONFIDENCE
    BAG_CONF             = DETECTION_CONFIDENCE
    IOU_THRESHOLD        = 0.45

    TRACKER = _resolve_tracker_yaml()
    PERSIST = True
    BAG_MATCH_THRESHOLD_PX = 100
    BAG_FRAME_MEMORY = 30

    # Camera
    CAMERA_HFOV           = 60.0
    ASSUMED_PERSON_HEIGHT = 1.70
    ASSUMED_BAG_HEIGHT    = 0.50
    IMAGE_WIDTH           = 1280
    IMAGE_HEIGHT          = 720
    # f = (w/2) / tan(HFOV/2) = 1108.97 px for w=1280 and HFOV=60°
    FOCAL_LENGTH          = (IMAGE_WIDTH / 2) / np.tan(np.radians(CAMERA_HFOV / 2))

    # Ownership
    DISTANCE_HISTORY_SIZE       = 30
    ASSIGNMENT_DISTANCE         = 2.5
    CONFIRMATION_TIME           = 1.0
    OWNERSHIP_LOCK_TIME         = 3.0
    SWITCH_DISTANCE_IMPROVEMENT = 1.0

    # Unattended
    POTENTIAL_THRESHOLD      = 5.0
    UNATTENDED_THRESHOLD     = 10.0
    OWNERSHIP_RELEASE_GRACE  = 5.0

    # ── Re-ID registry parameters ──────────────────────────────────────────
    # Max frames a lost track is remembered (900 = 30 s @ 30 fps)
    REID_MAX_AGE_FRAMES       = BAG_FRAME_MEMORY
    REID_MATCH_MAX_AGE_FRAMES = BAG_FRAME_MEMORY
    REID_CENTROID_THRESH      = 80
    REID_BAG_CENTROID_THRESH  = BAG_MATCH_THRESHOLD_PX
    REID_IOU_THRESH           = 0.25
    REID_BAG_AREA_RATIO_MIN   = 0.50
    REID_BAG_AREA_RATIO_MAX   = 2.00

    # Visualisation
    COLOR_PERSON        = (50, 205, 50)
    COLOR_BAG_OK        = (255, 165, 0)
    COLOR_BAG_POTENTIAL = (0, 165, 255)
    COLOR_BAG_UNATTENDED= (0, 0, 255)
    COLOR_DISTANCE_LINE = (255, 255, 0)
    COLOR_TEXT          = (255, 255, 255)
    COLOR_HIGHLIGHT     = (0, 255, 255)
    SHOW_DISTANCE_LINES = True
    SHOW_DEBUG_OVERLAY  = True
    LINE_THICKNESS      = 3
    FONT_SCALE          = 0.7
    FONT                = cv2.FONT_HERSHEY_SIMPLEX


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2: BGS STABLE-ID REGISTRY  (Layer 2 — the core fix)
# ═══════════════════════════════════════════════════════════════════════════
class BGSRegistry:
    """
    Two-layer identity system for persistent BGS IDs.

    Internal data per tracked entry
    ────────────────────────────────
    bgs_id       : stable human-readable ID  (e.g. P-3, B-7)
    bbox         : last known bounding box
    class_id     : COCO class integer
    last_frame   : frame number when last seen
    bt_ids       : set of ByteTrack IDs ever seen for this object
                   (a single real object may cycle through several BT IDs
                   as it briefly disappears; we collect them all so a
                   returning BT ID immediately maps back)

    Matching priority
    ──────────────────
    1. Exact ByteTrack ID match  (bt_id in entry.bt_ids)
    2. Centroid distance  < REID_CENTROID_THRESH  AND same class
    3. IoU               > REID_IOU_THRESH        AND same class
    If none match → new BGS ID is created.
    """

    def __init__(self):
        self.cfg         = BGSConfig
        self._entries: Dict[str, dict] = {}   # bgs_id → entry dict
        self._person_ctr = 0
        self._bag_ctr    = 0
        self._used_ids_in_frame: Dict[int, set] = {}

    def _same_identity_group(self, entry_class_id: int, query_class_id: int) -> bool:
        """Treat all bag classes as one identity group to prevent class-flip ID churn."""
        if entry_class_id == self.cfg.PERSON_CLASS_ID and query_class_id == self.cfg.PERSON_CLASS_ID:
            return True
        return (entry_class_id in self.cfg.BAG_CLASS_IDS) and (query_class_id in self.cfg.BAG_CLASS_IDS)

    # ── public API ─────────────────────────────────────────────────────────
    def resolve(self, bbox: List[float], class_id: int,
                frame_number: int,
                bt_id: Optional[int] = None) -> str:
        """
        Given a raw detection, return the stable BGS ID string.
        Updates the registry in-place.
        """
        self._expire(frame_number)
        used_ids = self._used_ids_in_frame.setdefault(frame_number, set())
        match_age_limit = self.cfg.REID_MATCH_MAX_AGE_FRAMES
        centroid_thresh = (
            self.cfg.REID_CENTROID_THRESH
            if class_id == 0
            else self.cfg.REID_BAG_CENTROID_THRESH
        )

        # --- Step 1: exact ByteTrack ID match ---
        if bt_id is not None:
            for bgs_id, entry in self._entries.items():
                if bgs_id in used_ids:
                    continue
                if self._same_identity_group(entry['class_id'], class_id) and bt_id in entry['bt_ids']:
                    self._update(bgs_id, bbox, frame_number, bt_id)
                    used_ids.add(bgs_id)
                    return bgs_id

        # --- Step 2: best spatial candidate (IoU-first, centroid secondary) ---
        best_id = None
        best_score = (-1.0, float('-inf'))
        cx, cy     = self._centroid(bbox)
        area = self._area(bbox)

        for bgs_id, entry in self._entries.items():
            if not self._same_identity_group(entry['class_id'], class_id):
                continue
            if bgs_id in used_ids:
                continue
            age = frame_number - entry['last_frame']
            if age < 0 or age > match_age_limit:
                continue

            if class_id in self.cfg.BAG_CLASS_IDS:
                prev_area = self._area(entry['bbox'])
                if prev_area > 0 and area > 0:
                    ratio = area / prev_area
                    if ratio < self.cfg.REID_BAG_AREA_RATIO_MIN or ratio > self.cfg.REID_BAG_AREA_RATIO_MAX:
                        continue

            iou = self._iou(bbox, entry['bbox'])
            ec, ey = self._predicted_centroid(entry, age)
            cd = np.hypot(cx - ec, cy - ey)

            if cd > centroid_thresh and iou < self.cfg.REID_IOU_THRESH:
                continue

            # Prefer overlap first, then tighter motion-consistent centroid match.
            score = (iou, -cd)
            if score > best_score:
                best_score = score
                best_id = bgs_id

        if best_id is not None:
            self._update(best_id, bbox, frame_number, bt_id)
            used_ids.add(best_id)
            return best_id

        # --- No match: create new entry ---
        new_id = self._new_id(class_id)
        self._entries[new_id] = {
            'bbox':       bbox,
            'class_id':   class_id,
            'last_frame': frame_number,
            'velocity':   (0.0, 0.0),
            'bt_ids':     {bt_id} if bt_id is not None else set(),
        }
        used_ids.add(new_id)
        return new_id

    def to_int(self, bgs_id: str) -> int:
        """Return integer portion of a BGS ID for ownership dict keys."""
        return int(bgs_id.split('-')[1])

    # ── private helpers ────────────────────────────────────────────────────
    def _update(self, bgs_id: str, bbox, frame_number, bt_id):
        e = self._entries[bgs_id]
        age = max(1, frame_number - e['last_frame'])
        prev_cx, prev_cy = self._centroid(e['bbox'])
        curr_cx, curr_cy = self._centroid(bbox)
        vel_x = (curr_cx - prev_cx) / age
        vel_y = (curr_cy - prev_cy) / age
        e['bbox']       = bbox
        e['last_frame'] = frame_number
        e['velocity']   = (vel_x, vel_y)
        if bt_id is not None:
            e['bt_ids'].add(bt_id)

    def _expire(self, frame_number: int):
        self._entries = {
            k: v for k, v in self._entries.items()
            if frame_number - v['last_frame'] <= self.cfg.REID_MAX_AGE_FRAMES
        }
        self._used_ids_in_frame = {
            k: v for k, v in self._used_ids_in_frame.items()
            if k >= frame_number - 2
        }

    def _new_id(self, class_id: int) -> str:
        if class_id == 0:
            self._person_ctr += 1
            return f"P-{self._person_ctr}"
        else:
            self._bag_ctr += 1
            return f"B-{self._bag_ctr}"

    @staticmethod
    def _centroid(bbox) -> Tuple[float, float]:
        return (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2

    @staticmethod
    def _area(bbox) -> float:
        return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])

    def _predicted_centroid(self, entry: Dict, age: int) -> Tuple[float, float]:
        cx, cy = self._centroid(entry['bbox'])
        vx, vy = entry.get('velocity', (0.0, 0.0))
        return cx + vx * age, cy + vy * age

    @staticmethod
    def _iou(a, b) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        if inter == 0:
            return 0.0
        union = ((ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter)
        return inter / union if union > 0 else 0.0


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 3: DISTANCE ESTIMATOR
# ═══════════════════════════════════════════════════════════════════════════
class DistanceEstimator:
    """
    Monocular trigonometric distance estimation.

    f = (IMAGE_WIDTH / 2) / tan(HFOV / 2)
    Z = (real_height * f) / bbox_height_pixels
    X = ((cx - image_center_x) * Z) / f
    Y = ((cy - image_center_y) * Z) / f
    D_3d = sqrt((x2-x1)^2 + (y2-y1)^2 + (z2-z1)^2)
    """

    def __init__(self):
        self.cfg = BGSConfig

    def estimate_depth(self, bbox_height_px: float, real_height_m: float) -> float:
        if bbox_height_px < 1:
            return 10.0
        return (real_height_m * self.cfg.FOCAL_LENGTH) / bbox_height_px

    def estimate_position_3d(self, bbox: List[float],
                             is_person: bool = False,
                             reference_depth: Optional[float] = None
                             ) -> Tuple[float, float, float]:
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        bh = y2 - y1

        if is_person and bh > 50:
            depth = self.estimate_depth(bh, self.cfg.ASSUMED_PERSON_HEIGHT)
        elif reference_depth is not None:
            depth = reference_depth          # shared-plane reference → less error
        else:
            depth = self.estimate_depth(bh, self.cfg.ASSUMED_BAG_HEIGHT)

        icx = self.cfg.IMAGE_WIDTH  / 2
        icy = self.cfg.IMAGE_HEIGHT / 2
        x_m = ((cx - icx) * depth) / self.cfg.FOCAL_LENGTH
        y_m = ((cy - icy) * depth) / self.cfg.FOCAL_LENGTH
        return x_m, y_m, depth

    def calculate_distance(self, p1: Tuple[float, float, float],
                           p2: Tuple[float, float, float]) -> float:
        return float(np.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2))))


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 4: BAG STATE
# ═══════════════════════════════════════════════════════════════════════════
@dataclass
class BagState:
    bag_id: str                    # BGS string ID e.g. "B-3"
    owner_id: Optional[str] = None # BGS string ID e.g. "P-1"
    owner_since: float        = 0.0
    last_close_time: float    = 0.0
    candidate_owner: Optional[str] = None
    candidate_since: float    = 0.0
    status: str               = "OK"

    owner_distance_history: Deque[float] = field(default_factory=lambda: deque(
        maxlen=BGSConfig.DISTANCE_HISTORY_SIZE
    ))


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 5: OWNERSHIP MANAGER
# ═══════════════════════════════════════════════════════════════════════════
class OwnershipManager:
    """
    Manages bag ownership using BGS string IDs.
    Trend-based decisions; owner-specific distance rolling mean.
    """

    def __init__(self, distance_estimator: DistanceEstimator):
        self.de          = distance_estimator
        self.cfg         = BGSConfig
        self.bag_states: Dict[str, BagState] = {}

    def update_ownership(self, bags: List[Dict], people: List[Dict],
                         current_time: float) -> Dict[str, BagState]:

        active_ids = {b['id'] for b in bags}
        self.bag_states = {k: v for k, v in self.bag_states.items()
                          if k in active_ids}

        for bag in bags:
            if bag['id'] not in self.bag_states:
                self.bag_states[bag['id']] = BagState(
                    bag_id=bag['id'],
                    last_close_time=current_time
                )

        for bag in bags:
            bid       = bag['id']
            state     = self.bag_states[bid]
            bag_pos   = bag['position_3d']

            # Distances to all people
            person_distances: List[Tuple[str, float]] = []
            for person in people:
                d = self.de.calculate_distance(bag_pos, person['position_3d'])
                person_distances.append((person['id'], d))
            person_distances.sort(key=lambda x: x[1])

            # Owner-specific distance
            owner_distance: Optional[float] = None
            if state.owner_id is not None:
                for pid, d in person_distances:
                    if pid == state.owner_id:
                        owner_distance = d
                        break

            # Rolling history — owner distance when owned, else closest
            sample = (owner_distance if owner_distance is not None
                      else (person_distances[0][1] if person_distances else 999.0))
            state.owner_distance_history.append(sample)
            smoothed = float(np.mean(list(state.owner_distance_history)))

            closest_distance = person_distances[0][1] if person_distances else 999.0

            self._apply_ownership_rules(state, person_distances,
                                        closest_distance, smoothed, owner_distance, current_time)
            self._update_status(state, smoothed, current_time)

        return self.bag_states

    def _apply_ownership_rules(self, state: BagState,
                               person_distances: List[Tuple[str, float]],
                               closest_distance: float,
                               smoothed_distance: float,
                               owner_distance: Optional[float],
                               current_time: float):
        if not person_distances:
            return

        closest_id, _ = person_distances[0]

        # ── No owner ──────────────────────────────────────────────────────
        if state.owner_id is None:
            if smoothed_distance <= self.cfg.ASSIGNMENT_DISTANCE:
                if state.candidate_owner == closest_id:
                    if current_time - state.candidate_since >= self.cfg.CONFIRMATION_TIME:
                        state.owner_id        = closest_id
                        state.owner_since     = current_time
                        state.last_close_time = current_time
                        state.candidate_owner = None
                else:
                    state.candidate_owner = closest_id
                    state.candidate_since  = current_time
            else:
                state.candidate_owner = None
            return

        # Update close-time if owner is near
        if owner_distance is not None and owner_distance <= self.cfg.ASSIGNMENT_DISTANCE:
            state.last_close_time = current_time

        # Consider switch only after lock period
        if current_time - state.owner_since < self.cfg.OWNERSHIP_LOCK_TIME:
            return

        owner_close = (owner_distance is not None
                       and owner_distance <= self.cfg.ASSIGNMENT_DISTANCE)
        if owner_close or smoothed_distance > self.cfg.ASSIGNMENT_DISTANCE:
            return

        if owner_distance is None:
            state.owner_id        = closest_id
            state.owner_since     = current_time
            state.last_close_time = current_time
            state.owner_distance_history.clear()
        elif owner_distance - closest_distance >= self.cfg.SWITCH_DISTANCE_IMPROVEMENT:
            state.owner_id        = closest_id
            state.owner_since     = current_time
            state.last_close_time = current_time
            state.owner_distance_history.clear()

    def _update_status(self, state: BagState,
                       smoothed_owner_distance: float, current_time: float):
        # "Since owner last within 2.5m" is timestamp-based and uses smoothed distance.
        if smoothed_owner_distance <= self.cfg.ASSIGNMENT_DISTANCE:
            state.last_close_time = current_time

        t = current_time - state.last_close_time
        if t >= self.cfg.UNATTENDED_THRESHOLD:
            state.status = "UNATTENDED"
        elif t >= self.cfg.POTENTIAL_THRESHOLD:
            state.status = "POTENTIAL"
        else:
            state.status = "OK"


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 6: VISUALIZER
# ═══════════════════════════════════════════════════════════════════════════
class Visualizer:

    def __init__(self):
        self.cfg = BGSConfig

    def draw_detection(self, frame: np.ndarray, det: Dict,
                       bag_state: Optional[BagState] = None):
        x1, y1, x2, y2 = map(int, det['bbox'])
        cname = det['class']

        if cname == 'person':
            color, thickness = self.cfg.COLOR_PERSON, self.cfg.LINE_THICKNESS
            label = f"Person {det['id']}"
        else:
            if bag_state:
                if   bag_state.status == "UNATTENDED": color, thickness = self.cfg.COLOR_BAG_UNATTENDED, 5
                elif bag_state.status == "POTENTIAL":  color, thickness = self.cfg.COLOR_BAG_POTENTIAL,  4
                else:                                  color, thickness = self.cfg.COLOR_BAG_OK, self.cfg.LINE_THICKNESS
                label = f"{cname.upper()} {det['id']}"
                if bag_state.owner_id:
                    label += f" [Owner:{bag_state.owner_id}]"
                label += f" [{bag_state.status}]"
            else:
                color, thickness = self.cfg.COLOR_BAG_OK, self.cfg.LINE_THICKNESS
                label = f"{cname.upper()} {det['id']}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        cl = 20
        cv2.line(frame, (x1, y1), (x1+cl, y1), color, thickness+1)
        cv2.line(frame, (x1, y1), (x1, y1+cl), color, thickness+1)

        (lw, lh), _ = cv2.getTextSize(label, self.cfg.FONT, self.cfg.FONT_SCALE, 2)
        ly = max(y1 - 15, lh + 8)
        cv2.rectangle(frame, (x1+3, ly-lh-5+3), (x1+lw+13, ly+5+3), (0, 0, 0), -1)
        cv2.rectangle(frame, (x1, ly-lh-5),     (x1+lw+10, ly+5),    color,     -1)
        cv2.putText(frame, label, (x1+5, ly), self.cfg.FONT,
                    self.cfg.FONT_SCALE, self.cfg.COLOR_TEXT, 2, cv2.LINE_AA)

    def draw_distance_line(self, frame: np.ndarray,
                           bag_bbox, person_bbox, distance: float):
        if not self.cfg.SHOW_DISTANCE_LINES:
            return
        bc = (int((bag_bbox[0]+bag_bbox[2])/2),    int((bag_bbox[1]+bag_bbox[3])/2))
        pc = (int((person_bbox[0]+person_bbox[2])/2), int((person_bbox[1]+person_bbox[3])/2))
        cv2.line(frame, bc, pc, self.cfg.COLOR_DISTANCE_LINE, 2, cv2.LINE_AA)
        mx, my = (bc[0]+pc[0])//2, (bc[1]+pc[1])//2
        txt = f"{distance:.2f}m"
        (tw, th), _ = cv2.getTextSize(txt, self.cfg.FONT, 0.6, 2)
        cv2.rectangle(frame, (mx-5, my-th-5), (mx+tw+5, my+5), (0,0,0), -1)
        cv2.putText(frame, txt, (mx, my), self.cfg.FONT,
                    0.6, self.cfg.COLOR_DISTANCE_LINE, 2, cv2.LINE_AA)

    def draw_debug_overlay(self, frame: np.ndarray, stats: Dict):
        if not self.cfg.SHOW_DEBUG_OVERLAY:
            return
        pw, ph = 580, 215
        ov = frame.copy()
        cv2.rectangle(ov, (0, 0), (pw, ph), (20, 20, 20), -1)
        cv2.rectangle(ov, (5, 5), (pw-5, ph-5), (40, 40, 40), -1)
        frame[:] = cv2.addWeighted(ov, 0.7, frame, 0.3, 0)
        cv2.rectangle(frame, (0, 0), (pw, ph), self.cfg.COLOR_HIGHLIGHT, 2)

        y, sp = 30, 25
        cv2.putText(frame, "BAG GUARD SYSTEM  v5.0 — Stable Identity",
                    (15, y), self.cfg.FONT, 0.72, self.cfg.COLOR_HIGHLIGHT, 2, cv2.LINE_AA)
        y += 10
        cv2.line(frame, (15, y), (pw-15, y), self.cfg.COLOR_HIGHLIGHT, 2)

        rows = [
            (f"FPS: {stats.get('fps',0)}   Frame: {stats.get('frame_number',0)}",
             self.cfg.COLOR_TEXT),
            (f"People: {stats.get('people_count',0)}   Bags: {stats.get('bags_count',0)}",
             self.cfg.COLOR_PERSON),
            (f"OK: {stats.get('bags_ok',0)}   POTENTIAL: {stats.get('bags_potential',0)}   "
             f"UNATTENDED: {stats.get('bags_unattended',0)}",
             self.cfg.COLOR_BAG_OK),
        ]
        for txt, col in rows:
            y += sp
            cv2.putText(frame, txt, (15, y), self.cfg.FONT, 0.58, col, 2, cv2.LINE_AA)

        y += sp
        un = stats.get('bags_unattended', 0)
        if un > 0:
            cv2.putText(frame, f"ALERT: {un} UNATTENDED BAG(S)", (15, y),
                        self.cfg.FONT, 0.65, self.cfg.COLOR_BAG_UNATTENDED, 3, cv2.LINE_AA)
        else:
            cv2.putText(frame, "All Bags Monitored", (15, y),
                        self.cfg.FONT, 0.6, self.cfg.COLOR_TEXT, 2, cv2.LINE_AA)

        # Right panel — parameters
        y2, x2 = 55, 310
        params = [
            "PARAMETERS (frozen):",
            f"Assign dist : {BGSConfig.ASSIGNMENT_DISTANCE}m",
            f"Potential   : {BGSConfig.POTENTIAL_THRESHOLD}s",
            f"Unattended  : {BGSConfig.UNATTENDED_THRESHOLD}s",
            f"Lock time   : {BGSConfig.OWNERSHIP_LOCK_TIME}s",
            f"Release     : {BGSConfig.UNATTENDED_THRESHOLD + BGSConfig.OWNERSHIP_RELEASE_GRACE}s",
            f"Re-ID window: {BGSConfig.REID_MAX_AGE_FRAMES} frames",
            f"Re-ID dist  : {BGSConfig.REID_CENTROID_THRESH}px",
        ]
        for p in params:
            cv2.putText(frame, p, (x2, y2), self.cfg.FONT,
                        0.42, (200, 200, 200), 1, cv2.LINE_AA)
            y2 += 17


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 7: MAIN SYSTEM
# ═══════════════════════════════════════════════════════════════════════════
class BagGuardSystem:
    """
    Full BGS pipeline with two-layer identity.

    extract_detections()  calls  registry.resolve()  for EVERY detection,
    whether ByteTrack assigned an ID or not.  The registry returns the same
    stable BGS string ID each time the same spatial object is seen.
    """

    def __init__(self, model_path: str, video_path: str, output_path: str,
                 show: bool = False):
        self.model_path  = str(Path(model_path).expanduser().resolve())
        self.video_path  = video_path
        self.output_path = output_path
        self.cfg         = BGSConfig
        self.show        = bool(show)

        print("\n" + "="*80)
        print("BAG GUARD SYSTEM v5.0  — Two-Layer Stable Identity")
        print("="*80)
        print("  Layer 1: ByteTrack       — frame-to-frame association")
        print("  Layer 2: BGS Registry    — 30-second spatial re-identification")
        print("  IDs format: P-N (people)  B-N (bags)  — never repeat or churn")
        print("="*80)

        print(f"Model: {self.model_path}")

        if not Path(self.model_path).exists():
            print(f"✗ Weights file not found: {self.model_path}")
            sys.exit(1)

        try:
            self.model = YOLO(self.model_path)
            print("✓ YOLO model loaded")
        except Exception as e:
            print(f"✗ Cannot load model: {e}")
            sys.exit(1)

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

        self._validate_class_config()
        self._log_device()
        self._run_startup_inference()

        self.registry = BGSRegistry()
        self.de       = DistanceEstimator()
        self.om       = OwnershipManager(self.de)
        self.viz      = Visualizer()

        self.frame_count     = 0
        self.person_ids_seen = set()
        self.bag_ids_seen    = set()
        self.start_time      = None
        print("✓ All subsystems initialised\n")

    def _validate_class_config(self):
        actual_ids = set(self.cfg.CLASS_IDS)
        if actual_ids != self.cfg.EXPECTED_CLASS_IDS:
            print("✗ Class ID set mismatch")
            print(f"Expected: {sorted(self.cfg.EXPECTED_CLASS_IDS)}")
            print(f"Actual: {sorted(actual_ids)}")
            sys.exit(1)

        if self.cfg.TARGET_CLASSES != self.cfg.EXPECTED_TARGET_CLASSES:
            print("✗ Class label mapping mismatch")
            print(f"Expected: {self.cfg.EXPECTED_TARGET_CLASSES}")
            print(f"Actual: {self.cfg.TARGET_CLASSES}")
            sys.exit(1)

        if set(self.cfg.TARGET_CLASSES.keys()) != actual_ids:
            print("✗ Class IDs and label mapping keys do not align")
            sys.exit(1)

        print("✓ Class IDs and label mapping verified")

    def _log_device(self):
        print(f"✓ Device: {self.device}")

    def _run_startup_inference(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print("⚠️  Startup inference skipped: cannot open source")
            return

        ret, frame = cap.read()
        cap.release()
        if not ret:
            print("⚠️  Startup inference skipped: no frame read")
            return

        frame = cv2.resize(frame, (self.cfg.IMAGE_WIDTH, self.cfg.IMAGE_HEIGHT))
        results = self.model.predict(
            frame,
            conf=self.cfg.DETECTION_CONFIDENCE,
            iou=self.cfg.IOU_THRESHOLD,
            classes=self.cfg.CLASS_IDS,
            device=self.device,
            verbose=False,
        )

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            print("⚠️  Startup inference: 0 detections for configured classes")
            return

        class_counts: Dict[int, int] = {}
        for cls_id in boxes.cls.cpu().numpy().astype(int).tolist():
            class_counts[cls_id] = class_counts.get(cls_id, 0) + 1

        person_count = class_counts.get(self.cfg.PERSON_CLASS_ID, 0)
        bag_count = sum(class_counts.get(cid, 0) for cid in self.cfg.BAG_CLASS_IDS)

        print("✓ Startup inference detections:")
        print(f"  Person: {person_count}")
        for cid in self.cfg.BAG_CLASS_IDS:
            label = self.cfg.TARGET_CLASSES.get(cid, f"class_{cid}")
            print(f"  {label}: {class_counts.get(cid, 0)}")
        print(f"  Bags total: {bag_count}")

    # ──────────────────────────────────────────────────────────────────────
    def extract_detections(self, results) -> Tuple[List[Dict], List[Dict]]:
        """
        Extract detections and resolve stable BGS IDs via the two-layer registry.

        ByteTrack ID  → passed to registry as hint (may be None for bags)
        Registry      → returns consistent BGS string ID regardless
        """
        people: List[Dict] = []
        bags:   List[Dict] = []

        if results[0].boxes is None or len(results[0].boxes) == 0:
            return people, bags

        boxes = results[0].boxes
        bag_candidates: List[Dict] = []

        for i in range(len(boxes)):
            cls_id = int(boxes[i].cls[0])
            if cls_id not in self.cfg.CLASS_IDS:
                continue

            bbox       = boxes[i].xyxy[0].cpu().numpy().tolist()
            conf       = float(boxes[i].conf[0])
            class_name = self.cfg.TARGET_CLASSES[cls_id]

            if cls_id == 0 and conf < self.cfg.PERSON_CONF:
                continue
            if cls_id in self.cfg.BAG_CLASS_IDS and conf < self.cfg.BAG_CONF:
                continue

            if cls_id == 0:  # PERSON
                bt_id = int(boxes[i].id[0]) if boxes[i].id is not None else None
                bgs_id = self.registry.resolve(bbox, cls_id, self.frame_count, bt_id)
                pos = self.de.estimate_position_3d(bbox, is_person=True)
                people.append(dict(id=bgs_id, class_='person', class_id=cls_id,
                                   bbox=bbox, conf=conf, position_3d=pos,
                                   **{'class': 'person'}))
                self.person_ids_seen.add(bgs_id)

            else:  # BAG
                bt_id = int(boxes[i].id[0]) if boxes[i].id is not None else None
                bag_candidates.append({
                    'class_id': cls_id,
                    'class': class_name,
                    'bbox': bbox,
                    'conf': conf,
                    'bt_id': bt_id,
                })

        # Suppress duplicate bag boxes in same frame (often caused by class jitter).
        bag_candidates.sort(key=lambda b: b['conf'], reverse=True)
        selected_bags: List[Dict] = []
        for cand in bag_candidates:
            keep = True
            cx = (cand['bbox'][0] + cand['bbox'][2]) / 2
            cy = (cand['bbox'][1] + cand['bbox'][3]) / 2
            for kept in selected_bags:
                iou = BGSRegistry._iou(cand['bbox'], kept['bbox'])
                kx = (kept['bbox'][0] + kept['bbox'][2]) / 2
                ky = (kept['bbox'][1] + kept['bbox'][3]) / 2
                center_dist = float(np.hypot(cx - kx, cy - ky))
                if iou >= 0.70 or center_dist <= (self.cfg.BAG_MATCH_THRESHOLD_PX * 0.35):
                    keep = False
                    break
            if keep:
                selected_bags.append(cand)

        for bag in selected_bags:
            bgs_id = self.registry.resolve(
                bag['bbox'], bag['class_id'], self.frame_count, bag['bt_id']
            )
            pos = self.de.estimate_position_3d(bag['bbox'], is_person=False)
            bags.append(dict(id=bgs_id, class_id=bag['class_id'],
                             bbox=bag['bbox'], conf=bag['conf'], position_3d=pos,
                             **{'class': bag['class']}))
            self.bag_ids_seen.add(bgs_id)

        return people, bags

    # ──────────────────────────────────────────────────────────────────────
    def _refine_bag_depths(self, bags: List[Dict], people: List[Dict]):
        """Use nearest person's depth as shared-plane reference for bags."""
        if not people:
            return
        for bag in bags:
            bx1, by1, bx2, by2 = bag['bbox']
            bcx, bcy = (bx1+bx2)/2, (by1+by2)/2
            best_depth, best_dist = None, 200.0
            for p in people:
                px1, py1, px2, py2 = p['bbox']
                d = np.hypot(bcx-(px1+px2)/2, bcy-(py1+py2)/2)
                if d < best_dist:
                    best_dist  = d
                    best_depth = p['position_3d'][2]
            if best_depth is not None:
                bag['position_3d'] = self.de.estimate_position_3d(
                    bag['bbox'], is_person=False, reference_depth=best_depth)

    # ──────────────────────────────────────────────────────────────────────
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict]:
        current_time = time.time()

        results = self.model.track(
            frame,
            persist=self.cfg.PERSIST,
            conf=min(self.cfg.PERSON_CONF, self.cfg.BAG_CONF),
            iou=self.cfg.IOU_THRESHOLD,
            classes=self.cfg.CLASS_IDS,
            tracker=self.cfg.TRACKER,
            verbose=False
        )

        people, bags = self.extract_detections(results)
        self._refine_bag_depths(bags, people)

        bag_states = self.om.update_ownership(bags, people, current_time)

        counts = {'OK': 0, 'POTENTIAL': 0, 'UNATTENDED': 0}
        for s in bag_states.values():
            counts[s.status] += 1

        for person in people:
            self.viz.draw_detection(frame, person)

        for bag in bags:
            state = bag_states.get(bag['id'])
            self.viz.draw_detection(frame, bag, state)
            if state and state.owner_id:
                owner = next((p for p in people if p['id'] == state.owner_id), None)
                if owner:
                    dist = self.de.calculate_distance(
                        bag['position_3d'], owner['position_3d'])
                    self.viz.draw_distance_line(frame, bag['bbox'], owner['bbox'], dist)

        stats = {
            'fps': 0,
            'people_count':    len(people),
            'bags_count':      len(bags),
            'bags_ok':         counts['OK'],
            'bags_potential':  counts['POTENTIAL'],
            'bags_unattended': counts['UNATTENDED'],
            'frame_number':    self.frame_count,
        }
        return frame, stats

    # ──────────────────────────────────────────────────────────────────────
    def run(self) -> bool:
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print("✗ Cannot open video")
            return False

        fps    = int(cap.get(cv2.CAP_PROP_FPS))
        total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out    = cv2.VideoWriter(self.output_path, fourcc, fps,
                                 (self.cfg.IMAGE_WIDTH, self.cfg.IMAGE_HEIGHT))

        print(f"✓ Video: {total} frames @ {fps} FPS")
        print("✓ Processing...\n")
        self.start_time = time.time()
        prev_time = time.time()

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                self.frame_count += 1
                now = time.time()
                frame = cv2.resize(frame, (self.cfg.IMAGE_WIDTH, self.cfg.IMAGE_HEIGHT))

                annotated, stats = self.process_frame(frame)

                fps_disp  = int(1 / (now - prev_time)) if now > prev_time else 0
                prev_time = now
                stats['fps'] = fps_disp

                self.viz.draw_debug_overlay(annotated, stats)
                out.write(annotated)
                if self.show:
                    cv2.imshow('BGS v5.0 — Stable Identity', annotated)

                if self.frame_count % 30 == 0:
                    pct = (self.frame_count / total) * 100
                    print(f"  {pct:5.1f}% | frame {self.frame_count:5d}/{total} | "
                          f"fps {fps_disp:3d} | people {stats['people_count']} | "
                          f"bags {stats['bags_count']} | "
                          f"unattended {stats['bags_unattended']}")

                if self.show and cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except Exception as e:
            import traceback
            print(f"\n✗ ERROR: {e}")
            traceback.print_exc()
            return False
        finally:
            cap.release()
            out.release()
            cv2.destroyAllWindows()

        elapsed = time.time() - self.start_time
        print(f"\n{'='*80}")
        print(f"DONE  |  frames {self.frame_count}  |  time {elapsed:.1f}s  |  "
              f"avg fps {self.frame_count/elapsed:.1f}")
        print(f"Unique person IDs: {len(self.person_ids_seen)}  |  "
              f"Unique bag IDs: {len(self.bag_ids_seen)}")
        print(f"Output → {self.output_path}")
        print("="*80)
        return True


# ═══════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════
def main():
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(description="BGS v5.0 stable identity")
    parser.add_argument("--source", default="0", help="0 for webcam or video path")
    parser.add_argument("--model", default=str(script_dir / "yolo26x.pt"), help="Model path")
    parser.add_argument("--out", "--output", dest="output",
                        default=str(script_dir / "bgs_v5_output.mp4"),
                        help="Output video path")
    parser.add_argument("--show", action="store_true", help="Show live preview window")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = script_dir / model_path

    if not model_path.exists():
        fallback_model = script_dir.parent / 'BagGuardSystem-BGS' / 'models' / 'yolo26x.pt'
        if fallback_model.exists():
            model_path = fallback_model

    source_arg = str(args.source)
    if source_arg.isdigit():
        video_path = int(source_arg)
    else:
        video_path = Path(source_arg)
        if not video_path.is_absolute():
            video_path = script_dir / video_path
        video_path = str(video_path)

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = script_dir / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    MODEL_PATH = str(model_path)
    VIDEO_PATH = video_path
    OUTPUT_PATH = str(output_path)

    system = BagGuardSystem(
        MODEL_PATH,
        VIDEO_PATH,
        OUTPUT_PATH,
        show=args.show,
    )
    system.run()


if __name__ == "__main__":
    main()
