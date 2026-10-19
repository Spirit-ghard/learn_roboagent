"""Official RoboBrain2.5 Quickstart: Temporal Value Estimation.

This example belongs to the official Robo-Dopamine project, not the local
``inference.py`` wrapper. Expected output is a processed-episode message and
an output directory containing the visualization/results.
"""

import os

from examples.inference import GRMInference

# model = GRMInference("tanhuajie2001/Robo-Dopamine-GRM-3B")
model = GRMInference("/root/gpufree-data/models/RoboBrain2.5-4B")

TASK_INSTRUCTION = "organize the table"
BASE_DEMO_PATH = "./examples/demo_table"
GOAL_IMAGE_PATH = "./examples/demo_table/goal_image.png"
OUTPUT_ROOT = "./results"

output_dir = model.run_pipeline(
    cam_high_path=os.path.join(BASE_DEMO_PATH, "cam_high.mp4"),
    cam_left_path=os.path.join(BASE_DEMO_PATH, "cam_left_wrist.mp4"),
    cam_right_path=os.path.join(BASE_DEMO_PATH, "cam_right_wrist.mp4"),
    out_root=OUTPUT_ROOT,
    task=TASK_INSTRUCTION,
    frame_interval=30,
    batch_size=1,
    goal_image=GOAL_IMAGE_PATH,
    eval_mode="incremental",
    visualize=True,
)
print(f"Episode ({BASE_DEMO_PATH}) processed with Incremental-Mode. Output at: {output_dir}")
