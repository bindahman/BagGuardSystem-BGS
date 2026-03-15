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
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
import time
from collections import deque
from dataclasses import dataclass
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
    # Section 7.1: Trend-Based Decision (NOT frame-by-frame)
    DISTANCE_HISTORY_SIZE = 30      # frames (rolling window)
    
    # Section 7.3: Assignment Rules
    ASSIGNMENT_DISTANCE = 2.5       # meters - close enough to be owner
    CONFIRMATION_TIME = 2.0         # seconds - time to confirm ownership
    OWNERSHIP_LOCK_TIME = 10.0      # seconds - ownership locked after assignment
    SWITCH_DISTANCE_IMPROVEMENT = 1.5  # meters - required to switch owner
    
    # ═══════════════════════════════════════════════════════════════════════
    # SECTION 8: UNATTENDED BAG LOGIC
    # ═══════════════════════════════════════════════════════════════════════
    # Section 8.1: Three-state status (OK, POTENTIAL, UNATTENDED)
    POTENTIAL_THRESHOLD = 5.0       # seconds - owner far, becoming potential
    UNATTENDED_THRESHOLD = 10.0     # seconds - prolonged absence = unattended
    OWNERSHIP_RELEASE_GRACE = 5.0   # seconds - release owner after prolonged absence

    # Two-layer identity persistence (registry)
    REID_MAX_AGE_FRAMES = 900
    REID_MATCH_MAX_AGE_FRAMES = 240
    REID_CENTROID_THRESH = 80
    REID_BAG_CENTROID_THRESH = 60
    REID_IOU_THRESH = 0.25
    
    # ═══════════════════════════════════════════════════════════════════════
    # SECTION 10: VISUALIZATION
    # ═══════════════════════════════════════════════════════════════════════
    # Professional color scheme (different colors per status)
    COLOR_PERSON = (50, 205, 50)           # Lime Green
    COLOR_BAG_OK = (255, 165, 0)           # Orange (owner close)
    COLOR_BAG_POTENTIAL = (0, 165, 255)    # Orange-Yellow (potentially unattended)
    COLOR_BAG_UNATTENDED = (0, 0, 255)     # Red (unattended)
    COLOR_DISTANCE_LINE = (255, 255, 0)    # Cyan (distance line)
    COLOR_TEXT = (255, 255, 255)           # White
    COLOR_BG = (30, 30, 30)                # Dark gray
    COLOR_HIGHLIGHT = (0, 255, 255)        # Cyan
    
    # Visual settings
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
    """
    Monocular distance estimation using trigonometry (Section 6.2)
    
    Method:
    1. Estimate depth (Z) using person height and bbox height
    2. Calculate lateral position (X, Y) using camera geometry
    3. Compute 3D Euclidean distance between objects
    
    Formula:
    Z = (real_height × focal_length) / bbox_height_pixels
    """
    
    def __init__(self):
        self.config = BGSConfig
    
    def estimate_depth(self, bbox_height_pixels: float, 
                      real_height_meters: float = None) -> float:
        """
        Estimate depth (Z distance) from camera using person height
        
        Args:
            bbox_height_pixels: Height of bounding box in pixels
            real_height_meters: Real-world height (default: assumed person height)
            
        Returns:
            Depth in meters
        """
        if real_height_meters is None:
            real_height_meters = self.config.ASSUMED_PERSON_HEIGHT
        
        if bbox_height_pixels < 1:
            return 10.0  # Far away default
        
        # Z = (real_height × focal_length) / bbox_height
        depth = (real_height_meters * self.config.FOCAL_LENGTH) / bbox_height_pixels
        
        return depth
    
    def estimate_position_3d(self, bbox: List[float],
                            is_person: bool = False,
                            reference_depth: Optional[float] = None) -> Tuple[float, float, float]:
        """
        Estimate 3D position (X, Y, Z) in meters
        
        Args:
            bbox: [x1, y1, x2, y2] bounding box
            is_person: Whether this is a person (for height estimation)
            
        Returns:
            (x_meters, y_meters, depth_meters)
        """
        x1, y1, x2, y2 = bbox
        
        # Center of bounding box (pixels)
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        bbox_height = y2 - y1
        
        # Estimate depth
        if is_person and bbox_height > 50:
            # Use person height for accurate depth
            depth = self.estimate_depth(bbox_height, self.config.ASSUMED_PERSON_HEIGHT)
        elif reference_depth is not None:
            depth = reference_depth
        else:
            depth = self.estimate_depth(bbox_height, self.config.ASSUMED_BAG_HEIGHT)
        
        # Calculate lateral position using camera geometry
        # X = (center_x - image_center_x) * depth / focal_length
        image_center_x = self.config.IMAGE_WIDTH / 2
        image_center_y = self.config.IMAGE_HEIGHT / 2
        
        x_meters = ((center_x - image_center_x) * depth) / self.config.FOCAL_LENGTH
        y_meters = ((center_y - image_center_y) * depth) / self.config.FOCAL_LENGTH
        
        return x_meters, y_meters, depth
    
    def calculate_distance(self, pos1: Tuple[float, float, float],
                          pos2: Tuple[float, float, float]) -> float:
        """
        Calculate 3D Euclidean distance between two positions
        
        Args:
            pos1: (x, y, z) in meters
            pos2: (x, y, z) in meters
            
        Returns:
            Distance in meters
        """
        x1, y1, z1 = pos1
        x2, y2, z2 = pos2
        
        distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
        
        return distance


class BGSRegistry:
    """Two-layer stable-ID registry for person and bag tracks."""

    PERSON_BASE_ID = 1_000_000
    BAG_BASE_ID = 2_000_000

    def __init__(self):
        self.config = BGSConfig
        self.entries: Dict[int, Dict] = {}
        self.next_person_id = 1
        self.next_bag_id = 1
        self.used_ids_by_frame: Dict[int, set] = {}

    def resolve(self, bbox: List[float], class_id: int, frame_number: int,
                bt_id: Optional[int] = None) -> int:
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
                    self._update(stable_id, bbox, frame_number, bt_id)
                    used_ids.add(stable_id)
                    return stable_id

        best_id = None
        best_score = (-1.0, float('-inf'))
        cx, cy = self._centroid(bbox)

        for stable_id, entry in self.entries.items():
            if entry['class_id'] != class_id or stable_id in used_ids:
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

        if best_id is not None:
            self._update(best_id, bbox, frame_number, bt_id)
            used_ids.add(best_id)
            return best_id

        new_id = self._new_id(class_id)
        self.entries[new_id] = {
            'bbox': bbox,
            'class_id': class_id,
            'last_frame': frame_number,
            'bt_ids': {bt_id} if bt_id is not None else set(),
        }
        used_ids.add(new_id)
        return new_id

    def _update(self, stable_id: int, bbox: List[float], frame_number: int,
                bt_id: Optional[int]):
        entry = self.entries[stable_id]
        entry['bbox'] = bbox
        entry['last_frame'] = frame_number
        if bt_id is not None:
            entry['bt_ids'].add(bt_id)

    def _expire(self, frame_number: int):
        self.entries = {
            k: v for k, v in self.entries.items()
            if frame_number - v['last_frame'] <= self.config.REID_MAX_AGE_FRAMES
        }
        self.used_ids_by_frame = {
            k: v for k, v in self.used_ids_by_frame.items()
            if k >= frame_number - 2
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
        self.id_registry = BGSRegistry()
        
        print("✓ Distance Estimator initialized")
        print("✓ Ownership Manager initialized")
        print("✓ Visualizer initialized")
        
        # Statistics
        self.frame_count = 0
        self.inference_frame_count = 0
        self.person_ids_seen = set()
        self.bag_ids_seen = set()
        self.start_time = None

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
    
    def extract_detections(self, results) -> Tuple[List[Dict], List[Dict]]:
        """Extract detections with 3D position estimation"""
        people = []
        bags = []
        
        if results[0].boxes is None or len(results[0].boxes) == 0:
            return people, bags
        
        boxes = results[0].boxes
        
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
            
            bt_id = int(boxes[i].id[0]) if boxes[i].id is not None else None
            stable_id = self.id_registry.resolve(bbox, cls_id, self.frame_count, bt_id)

            if cls_id == 0:  # PERSON
                position_3d = self.distance_estimator.estimate_position_3d(bbox, is_person=True)

                detection = {
                    'id': stable_id,
                    'class': class_name,
                    'class_id': cls_id,
                    'bbox': bbox,
                    'conf': conf,
                    'position_3d': position_3d
                }
                people.append(detection)
                self.person_ids_seen.add(stable_id)
                
            else:  # BAGS
                position_3d = self.distance_estimator.estimate_position_3d(bbox, is_person=False)
                
                detection = {
                    'id': stable_id,
                    'class': class_name,
                    'class_id': cls_id,
                    'bbox': bbox,
                    'conf': conf,
                    'position_3d': position_3d
                }
                bags.append(detection)
                self.bag_ids_seen.add(stable_id)
        
        return people, bags

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
        people, bags = self.extract_detections(results)
        self._refine_bag_depths(bags, people)
        
        # Update ownership
        bag_states = self.ownership_manager.update_ownership(bags, people, current_time)
        
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