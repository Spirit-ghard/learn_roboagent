"""Official RoboBrain2.5 Quickstart: Affordance Prediction.

Expected output: a dictionary containing a 2D point such as
``{'answer': '[(910, 597)]'}``; ``./result/affordance.jpg`` is written.
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
prompt = "the affordance area for holding the cup"
image = "./assets/demo/affordance.jpg"

# Visualization results will be saved to ./result, if `plot=True`.
pred = model.inference(prompt, image, task="pointing", plot=True, do_sample=False)
print(f"Prediction:\n{pred}")
