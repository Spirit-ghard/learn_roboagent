"""Official RoboBrain2.5 Quickstart: Referring Prediction.

Expected output: a dictionary containing a point between the two mugs, for
example ``{'answer': '[(593, 737)]'}``; ``./result/pointing.jpg`` is written.
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
prompt = "Identify spot within the vacant space that's between the two mugs"
image = "./assets/demo/pointing.jpg"

# Visualization results will be saved to ./result, if `plot=True`.
pred = model.inference(prompt, image, task="pointing", plot=True, do_sample=True)
print(f"Prediction:\n{pred}")
