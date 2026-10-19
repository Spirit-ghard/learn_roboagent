"""Official RoboBrain2.5 Quickstart: Visual Grounding (VG).

Expected output: a dictionary whose answer describes the red-hat person's
bounding box. With ``plot=True``, ``./result/grounding.jpg`` is also written.
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
prompt = "the person wearing a red hat"
image = "./assets/demo/grounding.jpg"

# Visualization results will be saved to ./result, if `plot=True`.
pred = model.inference(prompt, image, task="grounding", plot=True, do_sample=False)
print(f"Prediction:\n{pred}")
