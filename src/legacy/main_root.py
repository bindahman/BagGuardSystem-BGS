# src/main.py
from ultralytics import YOLO
import cv2
import time
import os
import argparse
from collections import Counter, deque
import math

# ============================================================
# BagGuard System (BGS)
# Phase C: Ownership + Persistence + (Temp) Unattended
# Professional additions included:
# - Frozen config via CFG (single source of truth)
# - Ultralytics tracking (ByteTrack/BOTSort) recommended
# - Trig-based lateral distance estimate (HFOV + person height)
# - Trend/history-based ownership decision (rolling window mean)
# - Ownership hysteresis + lock time + release time
# - EMA smoothing for angles (reduces jitter)
# - Cleanup of stale bag states (memory safe)
# - Optional saving output video + optional showing window
# - Debug overlay panel (counts + FPS + settings)
# ============================================================

# -----------------------------
# Frozen parameters (Supervisor feedback: "freeze the parameters")
# You can still override via CLI, but during runtime they remain fixed.
# -----------------------------
CFG = {
    # Visual styling (BGR)
    "COLOR_PERSON": (255, 200, 0),
    "COLOR_BAG": (255, 0, 255),
    "COLOR_TEXT": (255, 255, 255),
    "COLOR_BG": (0, 0, 0),
    "COLOR_COUNT": (0, 255, 0),
    "COLOR_OK": (255, 0, 255),
    "COLOR_POT": (0, 255, 255),
    "COLOR_UNATT": (0, 0, 255),

    # Trig distance assumptions
    "HFOV_DEG": 70.0,      # common webcam range ~60-75
    "PERSON_H_M": 1.70,    # assumed height in meters

    # Ownership thresholds (meters)
    "ASSIGN_M": 1.20,          # assign if within this distance (after confirm)
    "KEEP_M": 1.60,            # keep owner if within this distance
    "SWITCH_MARGIN_M": 0.35,   # require improvement to switch owners
    "ASSIGN_CONFIRM_SEC": 0.40,  # candidate must stay close for this long

    # Trend window / persistence controls (key "trend" improvement)
    "HISTORY_LEN": 15,      # last N frames per person candidate
    "LOCK_SEC": 2.0,        # once assigned, don't switch until lock expires (unless very strong reason)
    "RELEASE_SEC": 1.5,     # if owner is far for >= this time, allow switching easier
    "MIN_CANDIDATE_SAMPLES": 5,  # need at least this many distance samples before trusting mean

    # EMA smoothing (reduces jitter in angles and reduces ownership flicker)
    "EMA_ALPHA": 0.30,      # 0.2-0.4 is typical; higher = more reactive

    # Stale cleanup
    "BAG_STATE_TTL_SEC": 8.0,   # if bag not seen for this long -> remove state

    # Temporary unattended testing
    "POTENTIAL_SEC": 1.0,
    "UNATTENDED_SEC": 4.0,

    # Debug overlay
    "SHOW_FPS": True,
    "SHOW_SETTINGS": True,
}

# -----------------------------
# Model/class helpers (IMPORTANT: removes COCO-ID assumptions)
# -----------------------------
def names_map_from_model(model):
    """Return model.names as a dict {id: name} regardless of underlying type."""
    return model.names if isinstance(model.names, dict) else {i: n for i, n in enumerate(model.names)}

def build_class_id_sets(model):
    """
    Build class-id sets dynamically from model.names to avoid COCO-ID assumptions.
    Returns (target_class_ids, bag_class_ids, missing_names)
    """
    names_map = names_map_from_model(model)
    name_to_id = {str(v).lower(): int(k) for k, v in names_map.items()}

    wanted = ["person", "backpack", "handbag", "suitcase"]
    missing = [n for n in wanted if n not in name_to_id]

    target_ids = {name_to_id[n] for n in wanted if n in name_to_id}
    bag_ids = {name_to_id[n] for n in ["backpack", "handbag", "suitcase"] if n in name_to_id}

    return target_ids, bag_ids, missing

# -----------------------------
# Helpers
# -----------------------------
def now_sec() -> float:
    return time.time()

def bbox_centroid(bbox):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) // 2, (y1 + y2) // 2)

def frame_diag(frame):
    h, w = frame.shape[:2]
    return (w * w + h * h) ** 0.5

def euclid(p1, p2) -> float:
    dx = float(p1[0] - p2[0])
    dy = float(p1[1] - p2[1])
    return (dx * dx + dy * dy) ** 0.5

def normalized_distance(p1, p2, frame):
    d = euclid(p1, p2)
    diag = max(1.0, frame_diag(frame))
    return d / diag

# -----------------------------
# Trig-based distance (HFOV + person height)
# "Realistic-ish" meter estimate:
# - estimate depth Z using person bbox height + focal px from HFOV
# - compute lateral X on same depth plane
# - distance between bag and person = |X_bag - X_person| in meters
# -----------------------------
def focal_px_from_hfov(img_w: int, hfov_deg: float) -> float:
    hfov = math.radians(max(1e-6, hfov_deg))
    return (img_w / 2.0) / math.tan(hfov / 2.0)

def angle_from_cx(cx: float, img_cx: float, f_px: float) -> float:
    return math.atan((cx - img_cx) / max(1e-6, f_px))

def depth_from_person_height(bbox_h_px: float, person_h_m: float, f_px: float) -> float:
    h_px = max(1.0, float(bbox_h_px))
    return (f_px * float(person_h_m)) / h_px

def x_from_angle_and_depth(theta: float, Z: float) -> float:
    return float(Z) * math.tan(theta)

def ema(prev, new, alpha: float):
    if prev is None:
        return new
    return (1.0 - alpha) * prev + alpha * new

# -----------------------------
# Drawing helpers
# -----------------------------
def draw_label_with_bg(frame, x, y, text, box_color, text_color, scale=0.6):
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)

    x1 = max(0, int(x))
    y2 = max(0, int(y))
    y1 = max(0, int(y2 - th - 10))
    x2 = x1 + tw + 8

    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, -1)
    cv2.putText(frame, text, (x1 + 4, y2 - 4), font, scale, text_color, thickness)

def draw_counts_panel(frame, counts: Counter, fps_val=None, settings_lines=None):
    # compact HUD
    panel_x, panel_y = 8, 8
    panel_w = 320
    panel_h = 160 if (fps_val is not None or settings_lines) else 130

    cv2.rectangle(
        frame,
        (panel_x, panel_y),
        (panel_x + panel_w, panel_y + panel_h),
        CFG["COLOR_BG"],
        -1
    )

    cv2.putText(
        frame, "BGS HUD",
        (panel_x + 10, panel_y + 24),
        cv2.FONT_HERSHEY_SIMPLEX, 0.75,
        CFG["COLOR_TEXT"], 2
    )

    y = panel_y + 55
    for name in ["person", "backpack", "handbag", "suitcase"]:
        cv2.putText(
            frame,
            f"{name}: {counts.get(name, 0)}",
            (panel_x + 10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.70,
            CFG["COLOR_COUNT"],
            2
        )
        y += 24

    if fps_val is not None:
        cv2.putText(
            frame,
            f"FPS: {fps_val:.1f}",
            (panel_x + 10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.70,
            CFG["COLOR_TEXT"],
            2
        )
        y += 24

    if settings_lines:
        for line in settings_lines[:3]:  # keep it tidy
            cv2.putText(
                frame,
                line,
                (panel_x + 10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                CFG["COLOR_TEXT"],
                1
            )
            y += 20

# -----------------------------
# Ownership persistence (Trend + Lock + Release)
# -----------------------------
class BagOwnershipTrend:
    """
    Persistent association between bag_id and owner_id using:
    - rolling distance history per (bag_id, person_id)
    - mean distance decision (trend)
    - assignment confirmation time
    - keep distance hysteresis
    - lock period (prevents rapid switching)
    - release period (if owner far for some time -> easier switch)
    """
    def __init__(self,
                 assign_m, keep_m, switch_margin_m,
                 assign_confirm_sec,
                 history_len, lock_sec, release_sec,
                 min_candidate_samples,
                 bag_state_ttl_sec,
                 ema_alpha):

        self.assign_m = float(assign_m)
        self.keep_m = float(keep_m)
        self.switch_margin_m = float(switch_margin_m)
        self.assign_confirm_sec = float(assign_confirm_sec)

        self.history_len = int(history_len)
        self.lock_sec = float(lock_sec)
        self.release_sec = float(release_sec)
        self.min_candidate_samples = int(min_candidate_samples)

        self.bag_state_ttl_sec = float(bag_state_ttl_sec)
        self.ema_alpha = float(ema_alpha)

        # bag_id -> state dict
        self.state = {}

    def _get(self, bag_id):
        if bag_id not in self.state:
            self.state[bag_id] = {
                "owner_id": None,
                "owner_since": None,
                "lock_until": None,
                "last_seen": None,
                "last_close_time": None,  # last time within keep_m of owner
                "candidate_owner": None,
                "candidate_since": None,

                # distance history per person: pid -> deque([dist_m,...])
                "hist": {},
                # smoothed bag theta for stability: bag_id -> theta_ema
                "theta_ema": None,
            }
        return self.state[bag_id]

    def cleanup(self, t):
        # remove bags not seen recently
        dead = []
        for bid, st in self.state.items():
            if st["last_seen"] is None:
                continue
            if (t - st["last_seen"]) > self.bag_state_ttl_sec:
                dead.append(bid)
        for bid in dead:
            del self.state[bid]

    def update_seen(self, bag_id, t):
        st = self._get(bag_id)
        st["last_seen"] = t

    def update_bag_theta(self, bag_id, theta):
        st = self._get(bag_id)
        st["theta_ema"] = ema(st["theta_ema"], theta, self.ema_alpha)

    def push_distance(self, bag_id, person_id, dist_m):
        st = self._get(bag_id)
        hist = st["hist"].get(person_id)
        if hist is None:
            hist = deque(maxlen=self.history_len)
            st["hist"][person_id] = hist
        hist.append(float(dist_m))

    def _mean_dist(self, st, person_id):
        hist = st["hist"].get(person_id)
        if not hist:
            return None
        if len(hist) < self.min_candidate_samples:
            return None
        return sum(hist) / len(hist)

    def decide_owner(self, bag_id, t, visible_people_ids):
        """
        Returns (best_pid, best_mean_dist).
        Only considers people with enough history samples.
        """
        st = self._get(bag_id)
        best_pid = None
        best_md = None

        for pid in visible_people_ids:
            md = self._mean_dist(st, pid)
            if md is None:
                continue
            if best_md is None or md < best_md:
                best_md = md
                best_pid = pid

        return best_pid, best_md

    def associate(self, bag_id, t, best_pid, best_md, current_owner_dist):
        """
        Apply persistence:
        - If no owner: assign only after confirm time and within assign_m.
        - If owner exists:
            keep if within keep_m.
            otherwise, switching depends on lock+release and margin.
        """
        st = self._get(bag_id)

        # No candidate -> nothing
        if best_pid is None or best_md is None:
            st["candidate_owner"] = None
            st["candidate_since"] = None
            return st["owner_id"]

        # No owner yet -> candidate confirm
        if st["owner_id"] is None:
            if st["candidate_owner"] != best_pid:
                st["candidate_owner"] = best_pid
                st["candidate_since"] = t
            else:
                if (t - (st["candidate_since"] or t)) >= self.assign_confirm_sec and best_md <= self.assign_m:
                    st["owner_id"] = best_pid
                    st["owner_since"] = t
                    st["lock_until"] = t + self.lock_sec
                    st["last_close_time"] = t
                    st["candidate_owner"] = None
                    st["candidate_since"] = None
            return st["owner_id"]

        # Owner exists
        owner_id = st["owner_id"]

        # If we have current owner distance and it's close enough -> keep
        if current_owner_dist is not None and current_owner_dist <= self.keep_m:
            st["last_close_time"] = t
            return owner_id

        # Owner is far: figure out if we are allowed to switch
        locked = (st["lock_until"] is not None and t < st["lock_until"])

        # If owner has been far for long enough -> release easier
        if st["last_close_time"] is not None:
            far_for = t - st["last_close_time"]
        else:
            far_for = 999.0

        released = far_for >= self.release_sec

        # Strong-switch rule:
        # - If locked and not released -> require BIG improvement
        # - If released or not locked -> normal margin + assign threshold
        if best_pid != owner_id:
            if current_owner_dist is None:
                # if owner not visible -> allow switch if candidate is reasonably close
                should_switch = (best_md <= self.assign_m)
            else:
                if locked and not released:
                    # locked -> be strict
                    should_switch = (best_md + (self.switch_margin_m * 2.0)) < current_owner_dist
                else:
                    should_switch = (best_md + self.switch_margin_m) < current_owner_dist

            # Also require candidate not too far
            if should_switch and best_md <= (self.keep_m + 0.5):
                st["owner_id"] = best_pid
                st["owner_since"] = t
                st["lock_until"] = t + self.lock_sec
                if best_md <= self.keep_m:
                    st["last_close_time"] = t

        return st["owner_id"]

    def bag_status(self, bag_id, t, owner_id, dist_now, potential_sec, unattended_sec, keep_m):
        """
        Temporary unattended status:
        - track last_close_time as "within keep_m"
        - far_for = t - last_close_time
        """
        st = self._get(bag_id)

        if owner_id is None:
            return "NO_OWNER"

        # update last_close_time if close now
        if dist_now is not None and dist_now <= keep_m:
            st["last_close_time"] = t

        last_close = st["last_close_time"]
        far_for = 999.0 if last_close is None else (t - last_close)

        if far_for >= unattended_sec:
            return "UNATTENDED"
        if far_for >= potential_sec:
            return "POTENTIAL"
        return "OK"

# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(
        description="BagGuard System - Phase C Ownership (Trig Distance + Trend Persistence + Temp Unattended)"
    )
    parser.add_argument("--source", default="0", help="0 for webcam OR path to video file")

    # Default now points to your custom model weights in repo
    parser.add_argument("--model", default="models/yolo26x.pt", help="YOLO model weights path")

    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")

    parser.add_argument("--show", action="store_true", help="Show window")
    parser.add_argument("--save", action="store_true", help="Save output video to outputs/")
    parser.add_argument("--out", default="outputs/detection_output.mp4", help="Output path if --save")

    # Recommended tracking
    parser.add_argument("--use_ultra_track", action="store_true",
                        help="Use Ultralytics built-in tracking (recommended for stable IDs)")
    parser.add_argument("--tracker", default="bytetrack", choices=["bytetrack", "botsort"],
                        help="Tracker type if --use_ultra_track")

    # Allow overriding frozen parameters (but still fixed during runtime)
    parser.add_argument("--hfov", type=float, default=CFG["HFOV_DEG"], help="Approx webcam HFOV degrees (common: 60-75).")
    parser.add_argument("--person_h", type=float, default=CFG["PERSON_H_M"], help="Assumed person height in meters.")

    parser.add_argument("--assign_m", type=float, default=CFG["ASSIGN_M"])
    parser.add_argument("--keep_m", type=float, default=CFG["KEEP_M"])
    parser.add_argument("--switch_margin_m", type=float, default=CFG["SWITCH_MARGIN_M"])
    parser.add_argument("--assign_confirm_sec", type=float, default=CFG["ASSIGN_CONFIRM_SEC"])

    parser.add_argument("--history_len", type=int, default=CFG["HISTORY_LEN"])
    parser.add_argument("--lock_sec", type=float, default=CFG["LOCK_SEC"])
    parser.add_argument("--release_sec", type=float, default=CFG["RELEASE_SEC"])
    parser.add_argument("--min_candidate_samples", type=int, default=CFG["MIN_CANDIDATE_SAMPLES"])
    parser.add_argument("--ema_alpha", type=float, default=CFG["EMA_ALPHA"])

    # Temporary unattended testing
    parser.add_argument("--potential_sec", type=float, default=CFG["POTENTIAL_SEC"])
    parser.add_argument("--unattended_sec", type=float, default=CFG["UNATTENDED_SEC"])

    # Extra debug
    parser.add_argument("--debug", action="store_true", help="Extra debug overlay text")

    args = parser.parse_args()

    model = YOLO(args.model)

    # Build class-id sets dynamically (no COCO assumptions)
    TARGET_CLASS_IDS, BAG_CLASS_IDS, missing = build_class_id_sets(model)
    if missing:
        print("WARNING: Missing expected classes in model.names:", missing)
        print("model.names =", model.names)
        print("If detections look wrong, your dataset class names differ from: person/backpack/handbag/suitcase")

    source = 0 if str(args.source).strip() == "0" else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Cannot open source: {args.source}")
        print("Tip: use --source 0 for webcam or provide a valid video path.")
        return

    # Writer
    writer = None
    if args.save:
        out_dir = os.path.dirname(args.out)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        fps_cap = cap.get(cv2.CAP_PROP_FPS)
        if fps_cap is None or fps_cap <= 1e-3:
            fps_cap = 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.out, fourcc, fps_cap, (w, h))

    ownership = BagOwnershipTrend(
        assign_m=args.assign_m,
        keep_m=args.keep_m,
        switch_margin_m=args.switch_margin_m,
        assign_confirm_sec=args.assign_confirm_sec,
        history_len=args.history_len,
        lock_sec=args.lock_sec,
        release_sec=args.release_sec,
        min_candidate_samples=args.min_candidate_samples,
        bag_state_ttl_sec=CFG["BAG_STATE_TTL_SEC"],
        ema_alpha=args.ema_alpha,
    )

    tracker_yaml = "bytetrack.yaml" if args.tracker == "bytetrack" else "botsort.yaml"
    window_title = "BagGuard System – Phase C Ownership (Trend + Trig Distance)"

    # FPS meter
    last_fps_t = now_sec()
    fps = 0.0
    fps_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t = now_sec()
        ownership.cleanup(t)

        H, W = frame.shape[:2]
        img_cx = W / 2.0
        f_px = focal_px_from_hfov(W, args.hfov)

        # --- inference/tracking ---
        if args.use_ultra_track:
            res = model.track(frame, persist=True, verbose=False, conf=args.conf, tracker=tracker_yaml)[0]
        else:
            res = model(frame, verbose=False, conf=args.conf)[0]

        counts = Counter()
        people = []
        bags = []

        # robust names mapping
        names_map = names_map_from_model(model)

        if res.boxes is not None:
            for box in res.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])

                if cls_id not in TARGET_CLASS_IDS:
                    continue

                label = str(names_map.get(cls_id, cls_id))
                counts[label] += 1

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                bbox = (x1, y1, x2, y2)
                cx, cy = bbox_centroid(bbox)

                tid = None
                if hasattr(box, "id") and box.id is not None:
                    try:
                        tid = int(box.id[0])
                    except Exception:
                        tid = None

                det = {
                    "id": tid,
                    "cls_id": cls_id,
                    "label": label,
                    "conf": conf,
                    "bbox": bbox,
                    "centroid": (cx, cy),
                }

                # person class = whatever your model says it is (NOT assumed 0)
                if label.lower() == "person":
                    bbox_h = max(1, y2 - y1)
                    Z = depth_from_person_height(bbox_h, args.person_h, f_px)
                    theta = angle_from_cx(cx, img_cx, f_px)
                    X = x_from_angle_and_depth(theta, Z)
                    det["Z"] = Z
                    det["theta"] = theta
                    det["X"] = X
                    people.append(det)
                else:
                    theta = angle_from_cx(cx, img_cx, f_px)
                    det["theta_raw"] = theta
                    bags.append(det)

        # --- Build lookups ---
        visible_people_ids = [p["id"] for p in people if p["id"] is not None]
        people_by_id = {p["id"]: p for p in people if p["id"] is not None}

        # --- Ownership trend updates ---
        bag_to_owner = {}
        bag_to_dist_m = {}
        bag_status = {}

        for b in bags:
            bag_id = b["id"]
            if bag_id is None:
                continue  # persistence needs stable bag IDs

            ownership.update_seen(bag_id, t)

            # Smooth bag theta to reduce jitter
            ownership.update_bag_theta(bag_id, b["theta_raw"])
            st = ownership.state.get(bag_id)
            theta_b = st["theta_ema"] if st and st["theta_ema"] is not None else b["theta_raw"]

            # Push distances for all visible people into trend history
            for p in people:
                pid = p["id"]
                if pid is None:
                    continue
                Zp = p["Z"]
                Xp = p["X"]
                Xb = x_from_angle_and_depth(theta_b, Zp)  # bag on person depth plane
                dist_m = abs(Xb - Xp)
                ownership.push_distance(bag_id, pid, dist_m)

            # Decide best candidate using trend mean
            best_pid, best_md = ownership.decide_owner(bag_id, t, visible_people_ids)

            # Current owner distance (trend mean preferred, but we also compute "now" for status)
            st = ownership.state.get(bag_id)
            current_owner_id = st["owner_id"] if st else None

            current_owner_dist_now = None
            if current_owner_id is not None and current_owner_id in people_by_id:
                p = people_by_id[current_owner_id]
                Zp = p["Z"]
                Xp = p["X"]
                Xb = x_from_angle_and_depth(theta_b, Zp)
                current_owner_dist_now = abs(Xb - Xp)

            # For switching decisions, use trend mean for current owner if available
            current_owner_mean = None
            if current_owner_id is not None and st is not None:
                current_owner_mean = ownership._mean_dist(st, current_owner_id)

            owner_id = ownership.associate(
                bag_id=bag_id,
                t=t,
                best_pid=best_pid,
                best_md=best_md if best_md is not None else 999.0,
                current_owner_dist=current_owner_mean
            )

            bag_to_owner[bag_id] = owner_id

            # For display: prefer "now" distance to show live gap
            if owner_id is not None and current_owner_dist_now is not None:
                bag_to_dist_m[bag_id] = current_owner_dist_now
            else:
                bag_to_dist_m[bag_id] = best_md

            bag_status[bag_id] = ownership.bag_status(
                bag_id=bag_id,
                t=t,
                owner_id=owner_id,
                dist_now=bag_to_dist_m[bag_id],
                potential_sec=args.potential_sec,
                unattended_sec=args.unattended_sec,
                keep_m=args.keep_m
            )

        # --- Draw people ---
        for p in people:
            x1, y1, x2, y2 = p["bbox"]
            pid = p["id"]
            conf = p["conf"]
            label = p["label"]

            cv2.rectangle(frame, (x1, y1), (x2, y2), CFG["COLOR_PERSON"], 3)
            text = f"{label} #{pid} {conf:.2f}" if pid is not None else f"{label} {conf:.2f}"
            draw_label_with_bg(frame, x1, max(25, y1), text, CFG["COLOR_BG"], CFG["COLOR_TEXT"])

        # --- Draw bags ---
        for b in bags:
            x1, y1, x2, y2 = b["bbox"]
            bid = b["id"]
            conf = b["conf"]
            label = b["label"]

            if bid is None:
                cv2.rectangle(frame, (x1, y1), (x2, y2), CFG["COLOR_BAG"], 2)
                draw_label_with_bg(frame, x1, max(25, y1), f"{label} {conf:.2f}", CFG["COLOR_BG"], CFG["COLOR_TEXT"])
                continue

            status = bag_status.get(bid, "NO_OWNER")
            owner = bag_to_owner.get(bid, None)
            dist_m = bag_to_dist_m.get(bid, None)

            if status == "UNATTENDED":
                color = CFG["COLOR_UNATT"]
            elif status == "POTENTIAL":
                color = CFG["COLOR_POT"]
            else:
                color = CFG["COLOR_OK"]

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            dist_str = f"{dist_m:.2f}m" if dist_m is not None else "NA"
            if owner is None:
                text = f"{label} #{bid} {conf:.2f} | owner:? | d:{dist_str} | {status}"
            else:
                text = f"{label} #{bid} {conf:.2f} | owner:{owner} | d:{dist_str} | {status}"

            draw_label_with_bg(frame, x1, max(25, y1), text, CFG["COLOR_BG"], CFG["COLOR_TEXT"])

            # optional debug: show lock/release info
            if args.debug:
                st = ownership.state.get(bid, {})
                lock_until = st.get("lock_until")
                last_close = st.get("last_close_time")
                lock_left = max(0.0, (lock_until - t)) if lock_until else 0.0
                far_for = (t - last_close) if last_close else 999.0
                dbg = f"lock:{lock_left:.1f}s far:{far_for:.1f}s hist:{len(st.get('hist', {}))}"
                draw_label_with_bg(frame, x1, y2 + 24, dbg, CFG["COLOR_BG"], CFG["COLOR_TEXT"], scale=0.5)

        # --- FPS update ---
        fps_count += 1
        if (t - last_fps_t) >= 1.0:
            fps = fps_count / max(1e-6, (t - last_fps_t))
            fps_count = 0
            last_fps_t = t

        # --- HUD panel ---
        settings_lines = None
        if CFG["SHOW_SETTINGS"]:
            settings_lines = [
                f"tracker: {args.tracker if args.use_ultra_track else 'detect-only'}",
                f"assign/keep: {args.assign_m:.2f}/{args.keep_m:.2f}m  lock:{args.lock_sec:.1f}s",
                f"hist:{args.history_len}  ema:{args.ema_alpha:.2f}  HFOV:{args.hfov:.0f}",
            ]
        draw_counts_panel(
            frame,
            counts,
            fps_val=(fps if CFG["SHOW_FPS"] else None),
            settings_lines=settings_lines
        )

        # --- Save/show ---
        if writer is not None:
            writer.write(frame)

        if args.show:
            cv2.imshow(window_title, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    cap.release()
    if writer is not None:
        writer.release()
        print(f"Saved output to: {args.out}")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
