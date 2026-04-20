from typing import Dict, List, Optional

import cv2
import numpy as np

from .config import BGSConfig
from .ownership import BagState


class Visualizer:
    def __init__(self):
        self.config = BGSConfig

    def draw_detection(self, frame: np.ndarray, detection: Dict, bag_state: Optional[BagState] = None):
        x1, y1, x2, y2 = map(int, detection['bbox'])
        obj_id = detection['id']
        class_name = detection['class']

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
            match_reason = detection.get('match_reason')
            match_score = detection.get('match_score')
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

            if match_reason:
                label += f" [{match_reason}]"
            if match_reason in {'reid', 'geo'}:
                label += f" {match_score:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        corner_len = 20
        cv2.line(frame, (x1, y1), (x1 + corner_len, y1), color, thickness + 1)
        cv2.line(frame, (x1, y1), (x1, y1 + corner_len), color, thickness + 1)

        (lw, lh), _ = cv2.getTextSize(label, self.config.FONT, self.config.FONT_SCALE, 2)
        ly = max(y1 - 15, lh + 8)
        cv2.rectangle(frame, (x1 + 3, ly - lh - 5 + 3), (x1 + lw + 13, ly + 5 + 3), (0, 0, 0), -1)
        cv2.rectangle(frame, (x1, ly - lh - 5), (x1 + lw + 10, ly + 5), color, -1)
        cv2.putText(frame, label, (x1 + 5, ly), self.config.FONT, self.config.FONT_SCALE, self.config.COLOR_TEXT, 2, cv2.LINE_AA)

        if 'distance_to_owner' in detection and self.config.SHOW_DISTANCE_LABELS:
            dist_text = f"{detection['distance_to_owner']:.2f}m"
            cv2.putText(frame, dist_text, (x1, y2 + 20), self.config.FONT, 0.5, color, 2, cv2.LINE_AA)

    def draw_distance_line(self, frame: np.ndarray, bag_bbox: List[float], person_bbox: List[float], distance: float):
        if not self.config.SHOW_DISTANCE_LINES:
            return

        bag_cx = int((bag_bbox[0] + bag_bbox[2]) / 2)
        bag_cy = int((bag_bbox[1] + bag_bbox[3]) / 2)
        person_cx = int((person_bbox[0] + person_bbox[2]) / 2)
        person_cy = int((person_bbox[1] + person_bbox[3]) / 2)

        cv2.line(frame, (bag_cx, bag_cy), (person_cx, person_cy), self.config.COLOR_DISTANCE_LINE, 2, cv2.LINE_AA)

        mid_x = (bag_cx + person_cx) // 2
        mid_y = (bag_cy + person_cy) // 2
        dist_text = f"{distance:.2f}m"
        (tw, th), _ = cv2.getTextSize(dist_text, self.config.FONT, 0.6, 2)

        cv2.rectangle(frame, (mid_x - 5, mid_y - th - 5), (mid_x + tw + 5, mid_y + 5), (0, 0, 0), -1)
        cv2.putText(frame, dist_text, (mid_x, mid_y), self.config.FONT, 0.6, self.config.COLOR_DISTANCE_LINE, 2, cv2.LINE_AA)

    def draw_debug_overlay(self, frame: np.ndarray, stats: Dict):
        if not self.config.SHOW_DEBUG_OVERLAY:
            return

        panel_h = 225
        panel_w = 550

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (panel_w, panel_h), (20, 20, 20), -1)
        cv2.rectangle(overlay, (5, 5), (panel_w - 5, panel_h - 5), (40, 40, 40), -1)
        frame[:] = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)

        cv2.rectangle(frame, (0, 0), (panel_w, panel_h), self.config.COLOR_HIGHLIGHT, 2)

        y = 30
        spacing = 25
        cv2.putText(frame, "BAG GUARD SYSTEM", (15, y), self.config.FONT, 0.8, self.config.COLOR_HIGHLIGHT, 2, cv2.LINE_AA)
        y += 10
        cv2.line(frame, (15, y), (panel_w - 15, y), self.config.COLOR_HIGHLIGHT, 2)

        y += spacing
        cv2.putText(frame, f"FPS: {stats.get('fps', 0)}", (15, y), self.config.FONT, 0.6, self.config.COLOR_TEXT, 2, cv2.LINE_AA)

        y += spacing - 3
        cv2.putText(frame, f"People: {stats.get('people_count', 0)} | Bags: {stats.get('bags_count', 0)}", (15, y), self.config.FONT, 0.6, self.config.COLOR_PERSON, 2, cv2.LINE_AA)

        y += spacing - 3
        ok_count = stats.get('bags_ok', 0)
        pot_count = stats.get('bags_potential', 0)
        un_count = stats.get('bags_unattended', 0)
        cv2.putText(frame, f"Status: OK:{ok_count} POT:{pot_count} UN:{un_count}", (15, y), self.config.FONT, 0.6, self.config.COLOR_BAG_OK, 2, cv2.LINE_AA)

        y += spacing - 3
        if un_count > 0:
            cv2.putText(frame, f"ALERT: UNATTENDED BAGS: {un_count}", (15, y), self.config.FONT, 0.65, self.config.COLOR_BAG_UNATTENDED, 3, cv2.LINE_AA)
        else:
            cv2.putText(frame, "All Bags Monitored", (15, y), self.config.FONT, 0.6, self.config.COLOR_TEXT, 2, cv2.LINE_AA)

        runtime = str(stats.get('detector_runtime', 'torch')).upper()
        device = str(stats.get('detector_device', '')).upper()
        tracker_backend = str(stats.get('tracker_backend', 'deepsort'))

        y += spacing - 3
        detector_label = f"Detector: {runtime}"
        if device:
            detector_label += f" ({device})"
        cv2.putText(frame, detector_label, (15, y), self.config.FONT, 0.5, (180, 180, 180), 1, cv2.LINE_AA)

        y += spacing - 5
        cv2.putText(frame, f"Tracker: {tracker_backend} | Raw dets: {stats.get('raw_detections', 0)}", (15, y), self.config.FONT, 0.5, (180, 180, 180), 1, cv2.LINE_AA)

        y += spacing - 5
        cv2.putText(frame, f"Frame: {stats.get('frame_number', 0)}", (15, y), self.config.FONT, 0.5, (180, 180, 180), 1, cv2.LINE_AA)

        y2 = 30 + spacing + 10
        x2 = 280
        cv2.putText(frame, "PARAMETERS:", (x2, y2), self.config.FONT, 0.5, self.config.COLOR_HIGHLIGHT, 1, cv2.LINE_AA)
        y2 += 18
        cv2.putText(frame, f"Assign Dist: {self.config.ASSIGNMENT_DISTANCE}m", (x2, y2), self.config.FONT, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        y2 += 16
        cv2.putText(frame, f"Potential: {self.config.POTENTIAL_THRESHOLD}s", (x2, y2), self.config.FONT, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        y2 += 16
        cv2.putText(frame, f"Unattended: {self.config.UNATTENDED_THRESHOLD}s", (x2, y2), self.config.FONT, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        y2 += 16
        cv2.putText(frame, f"Lock Time: {self.config.OWNERSHIP_LOCK_TIME}s", (x2, y2), self.config.FONT, 0.45, (200, 200, 200), 1, cv2.LINE_AA)