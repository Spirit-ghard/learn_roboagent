"""Official RoboBrain2.5 Quickstart: Navigation.

Expected output: one dictionary containing a point inside the toilet, followed
by one containing a point inside the sofa. Previously observed points are
``[(416, 368)]`` and ``[(480, 935)]``; outputs vary by model sampling.
``./assets/results/navigation.jpg`` is written.
"""

import sys
from pathlib import Path

EXAMPLE_DIR = Path(__file__).resolve().parent
SECTION_DIR = EXAMPLE_DIR.parent
sys.path.insert(0, str(EXAMPLE_DIR))

from inference import UnifiedInference

model = UnifiedInference("/root/gpufree-data/models/RoboBrain2.5-4B", device_map="auto")

# Example 1:
prompt_1 = "Identify spot within toilet in the house"
image = str(SECTION_DIR / "原始图片" / "navigation.jpg")

# Visualization results will be saved to ./assets/results, if `plot=True`.
pred = model.inference(prompt_1, image, task="pointing", plot=True, do_sample=True)
print(f"Prediction:\n{pred}")

# Example 2:
prompt_2 = "Identify spot within the sofa in the house"
image = str(SECTION_DIR / "原始图片" / "navigation.jpg")

# Visualization results will be saved to ./assets/results, if `plot=True`.
pred = model.inference(prompt_2, image, task="pointing", plot=True, do_sample=True)
print(f"Prediction:\n{pred}")
