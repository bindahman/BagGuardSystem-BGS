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
            if detection.get('pending'):
                color = self.config.COLOR_BAG_POTENTIAL
                thickness = self.config.LINE_THICKNESS
                frames_left = int(detection.get('pending_frames_left', 0))
                label = f"{class_name.upper()} [ID in {frames_left}]"
            elif bag_state:
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

        frame_h, frame_w = frame.shape[:2]
        panel_w = max(220, min(550, int(frame_w * 0.55)))
        panel_h = max(145, min(200, int(frame_h * 0.36)))
        panel_w = min(panel_w, max(frame_w - 4, 1))
        panel_h = min(panel_h, max(frame_h - 4, 1))
        compact_layout = panel_w < 480 or panel_h < 185
        scale = min(panel_w / 550.0, panel_h / 200.0)
        scale = max(0.55, min(1.0, scale))

        title_scale = max(0.5, 0.8 * scale)
        body_scale = max(0.38, 0.6 * scale)
        small_scale = max(0.34, 0.5 * scale)
        title_thickness = 2 if scale >= 0.85 else 1
        body_thickness = 2 if scale >= 0.9 else 1
        padding_x = max(12, int(15 * scale))
        title_y = max(24, int(30 * scale))
        divider_gap = max(8, int(10 * scale))
        spacing = max(16, int(25 * scale))
        small_spacing = max(14, int(18 * scale))

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (panel_w, panel_h), (20, 20, 20), -1)
        cv2.rectangle(overlay, (5, 5), (panel_w - 5, panel_h - 5), (40, 40, 40), -1)
        frame[:] = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)

        cv2.rectangle(frame, (0, 0), (panel_w, panel_h), self.config.COLOR_HIGHLIGHT, 2)

        ok_count = stats.get('bags_ok', 0)
        pot_count = stats.get('bags_potential', 0)
        un_count = stats.get('bags_unattended', 0)
        tracker_backend = str(stats.get('tracker_backend') or self.config.TRACKER.replace('.yaml', ''))

        cv2.putText(frame, "BAG GUARD SYSTEM", (padding_x, title_y), self.config.FONT, title_scale, self.config.COLOR_HIGHLIGHT, title_thickness, cv2.LINE_AA)
        divider_y = title_y + divider_gap
        cv2.line(frame, (padding_x, divider_y), (panel_w - padding_x, divider_y), self.config.COLOR_HIGHLIGHT, max(1, title_thickness), cv2.LINE_AA)

        if compact_layout:
            y = divider_y + spacing
            compact_lines = [
                (f"FPS: {stats.get('fps', 0)}", body_scale, self.config.COLOR_TEXT, body_thickness),
                (f"People: {stats.get('people_count', 0)} | Bags: {stats.get('bags_count', 0)}", body_scale, self.config.COLOR_PERSON, body_thickness),
                (f"Status: OK:{ok_count} POT:{pot_count} UN:{un_count}", body_scale, self.config.COLOR_BAG_OK, body_thickness),
            ]

            for text, font_scale, color, thickness in compact_lines:
                cv2.putText(frame, text, (padding_x, y), self.config.FONT, font_scale, color, thickness, cv2.LINE_AA)
                y += spacing

            if un_count > 0:
                alert_text = f"ALERT: UNATTENDED BAGS: {un_count}"
                alert_color = self.config.COLOR_BAG_UNATTENDED
            else:
                alert_text = "All Bags Monitored"
                alert_color = self.config.COLOR_TEXT
            cv2.putText(frame, alert_text, (padding_x, y), self.config.FONT, body_scale, alert_color, body_thickness, cv2.LINE_AA)
            y += spacing

            cv2.putText(frame, f"Tracker: {tracker_backend} | Frame: {stats.get('frame_number', 0)}", (padding_x, y), self.config.FONT, small_scale, (180, 180, 180), 1, cv2.LINE_AA)
            y += small_spacing
            cv2.putText(frame, f"Dist:{self.config.ASSIGNMENT_DISTANCE}m Pot:{self.config.POTENTIAL_THRESHOLD}s", (padding_x, y), self.config.FONT, small_scale, (200, 200, 200), 1, cv2.LINE_AA)
            y += small_spacing
            cv2.putText(frame, f"Unatt:{self.config.UNATTENDED_THRESHOLD}s Lock:{self.config.OWNERSHIP_LOCK_TIME}s", (padding_x, y), self.config.FONT, small_scale, (200, 200, 200), 1, cv2.LINE_AA)
            return

        y = divider_y + spacing
        cv2.putText(frame, f"FPS: {stats.get('fps', 0)}", (padding_x, y), self.config.FONT, body_scale, self.config.COLOR_TEXT, body_thickness, cv2.LINE_AA)

        y += spacing - 3
        cv2.putText(frame, f"People: {stats.get('people_count', 0)} | Bags: {stats.get('bags_count', 0)}", (padding_x, y), self.config.FONT, body_scale, self.config.COLOR_PERSON, body_thickness, cv2.LINE_AA)

        y += spacing - 3
        cv2.putText(frame, f"Status: OK:{ok_count} POT:{pot_count} UN:{un_count}", (padding_x, y), self.config.FONT, body_scale, self.config.COLOR_BAG_OK, body_thickness, cv2.LINE_AA)

        y += spacing - 3
        if un_count > 0:
            cv2.putText(frame, f"ALERT: UNATTENDED BAGS: {un_count}", (padding_x, y), self.config.FONT, max(body_scale, 0.48), self.config.COLOR_BAG_UNATTENDED, max(body_thickness + 1, 2), cv2.LINE_AA)
        else:
            cv2.putText(frame, "All Bags Monitored", (padding_x, y), self.config.FONT, body_scale, self.config.COLOR_TEXT, body_thickness, cv2.LINE_AA)

        y += spacing - 3
        cv2.putText(frame, f"Tracker: {tracker_backend}", (padding_x, y), self.config.FONT, small_scale, (180, 180, 180), 1, cv2.LINE_AA)

        y += spacing - 5
        cv2.putText(frame, f"Frame: {stats.get('frame_number', 0)}", (padding_x, y), self.config.FONT, small_scale, (180, 180, 180), 1, cv2.LINE_AA)

        y2 = divider_y + spacing
        x2 = max(int(panel_w * 0.52), padding_x + 190)
        x2 = min(x2, panel_w - 150)
        cv2.putText(frame, "PARAMETERS:", (x2, y2), self.config.FONT, small_scale, self.config.COLOR_HIGHLIGHT, 1, cv2.LINE_AA)
        y2 += small_spacing
        cv2.putText(frame, f"Assign Dist: {self.config.ASSIGNMENT_DISTANCE}m", (x2, y2), self.config.FONT, small_scale, (200, 200, 200), 1, cv2.LINE_AA)
        y2 += small_spacing
        cv2.putText(frame, f"Potential: {self.config.POTENTIAL_THRESHOLD}s", (x2, y2), self.config.FONT, small_scale, (200, 200, 200), 1, cv2.LINE_AA)
        y2 += small_spacing
        cv2.putText(frame, f"Unattended: {self.config.UNATTENDED_THRESHOLD}s", (x2, y2), self.config.FONT, small_scale, (200, 200, 200), 1, cv2.LINE_AA)
        y2 += small_spacing
        cv2.putText(frame, f"Lock Time: {self.config.OWNERSHIP_LOCK_TIME}s", (x2, y2), self.config.FONT, small_scale, (200, 200, 200), 1, cv2.LINE_AA)