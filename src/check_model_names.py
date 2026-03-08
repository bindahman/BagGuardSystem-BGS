from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "models" / "yolo26x.pt"

print("SCRIPT:", Path(__file__).resolve())
print("ROOT:", ROOT)
print("WEIGHTS:", WEIGHTS)
print("WEIGHTS EXISTS?:", WEIGHTS.exists())

# If it exists, try loading
m = YOLO(str(WEIGHTS))
print("model.names =", m.names)
