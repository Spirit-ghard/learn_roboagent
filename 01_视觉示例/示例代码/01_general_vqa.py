"""Official RoboBrain2.5 Quickstart: General VQA.
代码1结果
Prediction:
{'answer': 'This image shows two cats lying on a pink blanket. One cat is lying on its side, while the other is lying on its stomach. There are two remote controls visible in the image, one near each cat. The cats appear to be relaxed and comfortable in their position. The setting suggests a cozy and calm environment.'}
"""

import sys
from pathlib import Path

EXAMPLE_DIR = Path(__file__).resolve().parent
SECTION_DIR = EXAMPLE_DIR.parent
sys.path.insert(0, str(EXAMPLE_DIR))

from inference import UnifiedInference

model = UnifiedInference("/root/gpufree-data/models/RoboBrain2.5-4B", device_map="auto")

# Example:
prompt = "What is shown in this image?"
# Official Quickstart URL:
# image = "http://images.cocodataset.org/val2017/000000039769.jpg"
# Use the project copy in this offline container to avoid network wait.
image = str(SECTION_DIR / "原始图片" / "coco.jpg")

pred = model.inference(prompt, image, task="general")
print(f"Prediction:\n{pred}")
