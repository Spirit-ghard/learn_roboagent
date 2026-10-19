"""Official RoboBrain2.5 Quickstart: 3D Trajectory Prediction.

Expected output: a dictionary containing a list of image-space waypoints with
depth values, e.g. ``{'answer': '[(313, 483, 0.600), ...]'}``.
``./result/trajectory.jpg`` is written.
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from inference import UnifiedInference

model = UnifiedInference("/root/gpufree-data/models/RoboBrain2.5-4B", device_map="auto")

# Example:
prompt = "reach for the banana on the plate"
image = "./assets/demo/trajectory.jpg"

# Visualization results will be saved to ./result, if `plot=True`.
# Output is formatted as a list of tuples, i.e., [(x1, y1, d1), (x2, y2, d2), ...],
# where each tuple contains the x and y coordinates and the depth of the point.
pred = model.inference(prompt, image, task="trajectory", plot=True, do_sample=False)
print(f"Prediction:\n{pred}")
