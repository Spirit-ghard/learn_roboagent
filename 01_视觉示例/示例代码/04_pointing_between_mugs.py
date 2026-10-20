"""Official RoboBrain2.5 Quickstart: Referring Prediction.

Expected output: a dictionary containing a point between the two mugs, for
example ``{'answer': '[(593, 737)]'}``; ``./assets/results/pointing.jpg`` is written.
"""

import sys
from pathlib import Path

EXAMPLE_DIR = Path(__file__).resolve().parent
SECTION_DIR = EXAMPLE_DIR.parent
sys.path.insert(0, str(EXAMPLE_DIR))

from inference import UnifiedInference

model = UnifiedInference("/root/gpufree-data/models/RoboBrain2.5-4B", device_map="auto")

# Example:
prompt = "Identify spot within the vacant space that's between the two mugs"
image = str(SECTION_DIR / "原始图片" / "pointing.jpg")

# Visualization results will be saved to ./assets/results, if `plot=True`.
pred = model.inference(prompt, image, task="pointing", plot=True, do_sample=True)
print(f"Prediction:\n{pred}")
