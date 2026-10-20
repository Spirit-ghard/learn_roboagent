"""Official RoboBrain2.5 Quickstart: Visual Grounding (VG).

Expected output: a dictionary whose answer describes the red-hat person's
bounding box. With ``plot=True``, ``./assets/results/grounding.jpg`` is also written.
"""

import sys
from pathlib import Path

EXAMPLE_DIR = Path(__file__).resolve().parent
SECTION_DIR = EXAMPLE_DIR.parent
sys.path.insert(0, str(EXAMPLE_DIR))

from inference import UnifiedInference

model = UnifiedInference("/root/gpufree-data/models/RoboBrain2.5-4B", device_map="auto")

# Example:
prompt = "the person wearing a red hat"
image = str(SECTION_DIR / "原始图片" / "grounding.jpg")

# Visualization results will be saved to ./assets/results, if `plot=True`.
pred = model.inference(prompt, image, task="grounding", plot=True, do_sample=False)
print(f"Prediction:\n{pred}")
