"""Official RoboBrain2.5 Quickstart: Affordance Prediction.

Expected output: a dictionary containing a 2D point such as
``{'answer': '[(910, 597)]'}``; ``./assets/results/affordance.jpg`` is written.
"""

import sys
from pathlib import Path

EXAMPLE_DIR = Path(__file__).resolve().parent
SECTION_DIR = EXAMPLE_DIR.parent
sys.path.insert(0, str(EXAMPLE_DIR))

from inference import UnifiedInference

model = UnifiedInference("/root/gpufree-data/models/RoboBrain2.5-4B", device_map="auto")

# Example:
prompt = "the affordance area for holding the cup"
image = str(SECTION_DIR / "原始图片" / "affordance.jpg")

# Visualization results will be saved to ./assets/results, if `plot=True`.
pred = model.inference(prompt, image, task="pointing", plot=True, do_sample=False)
print(f"Prediction:\n{pred}")
