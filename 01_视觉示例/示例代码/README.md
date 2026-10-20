# RoboBrain2.5 official Quickstart examples

The seven files mirror the seven Quickstart sections in the official
RoboBrain2.5 README. Examples 01-06 call `UnifiedInference` directly with the
official prompts and task arguments. They use the locally cached 4B checkpoint
and do not contact Hugging Face. Example 07 is the separate Robo-Dopamine API.

1. `01_general_vqa.py`: General VQA
2. `02_grounding_red_hat.py`: Visual Grounding
3. `03_pointing_affordance.py`: Affordance Prediction
4. `04_pointing_between_mugs.py`: Referring Prediction
5. `05_navigation_toilet.py`: Navigation, including the official toilet and sofa prompts
6. `06_trajectory_banana.py`: 3D Trajectory Prediction
7. `07_temporal_value_estimation.py`: Temporal Value Estimation through Robo-Dopamine

Run one directly from the example directory:

```bash
cd /root/Desktop/RoboBrain/01_视觉示例/示例代码
/opt/conda/envs/brain_env/bin/python -u 01_general_vqa.py
```

Examples 01-06 use `/root/gpufree-data/models/RoboBrain2.5-4B`. The official
8B checkpoint is intentionally not used because it is unavailable offline and
does not fit this deployment target. Example 07 is an official Robo-Dopamine
pipeline and is kept for reference only; it is excluded from
`run_all_brain_examples.sh` because that repository and its demo videos are
not installed.

Original input images are stored in `../原始图片/`. Annotated output
images are stored in `../标注图片/`; the output contains the predicted
points, bounding boxes, or trajectory. General VQA uses the project copy of
the official COCO image at `assets/original/coco.jpg`; the upstream URL is kept in the file as a comment
because this container has no outbound network.

The expected output shape for every file is documented at the top of that
file. Visual tasks save annotations under `../标注图片/` when `plot=True`.
