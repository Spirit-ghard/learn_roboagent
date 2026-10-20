import os
import re
from typing import Union
from pathlib import Path

import cv2
import copy
from qwen_vl_utils import process_vision_info
from transformers import AutoModelForImageTextToText, AutoProcessor


class UnifiedInference:
    """Minimal official-compatible RoboBrain2.5 inference wrapper."""

    def __init__(self, model_id="/root/gpufree-data/models/RoboBrain2.5-4B", device_map="auto"):
        self.model_id = model_id
        self._plot_counts = {}
        local_only = Path(model_id).is_dir()
        load_kwargs = {"dtype": "auto", "device_map": device_map}
        if local_only:
            load_kwargs["local_files_only"] = True
        self.model = AutoModelForImageTextToText.from_pretrained(model_id, **load_kwargs)
        processor_kwargs = {"local_files_only": True} if local_only else {}
        self.processor = AutoProcessor.from_pretrained(model_id, **processor_kwargs)

    def inference(
        self,
        text: str,
        image: Union[list, str],
        task="general",
        plot=False,
        do_sample=True,
        temperature=0.7,
        max_new_tokens=768,
    ):
        if isinstance(image, str):
            image = [image]
        if task not in {"general", "pointing", "trajectory", "grounding"}:
            raise ValueError(f"Unsupported task: {task}")
        if task != "general" and len(image) != 1:
            raise ValueError(f"{task} requires exactly one image")
        if task == "pointing":
            text = (
                f"{text}. Please provide its 2D coordinates. Answer as [(x, y)], "
                "where x and y are coordinates in the 0-1000 range."
            )
        elif task == "trajectory":
            text = (
                "Please predict 3D end-effector-centric waypoints to complete the "
                f'task successfully. The task is "{text}". Answer as '
                "[(x1, y1, d1), (x2, y2, d2), ...], with x/y in 0-1000."
            )
        elif task == "grounding":
            text = f"Please provide the bounding box for: {text}."

        messages = [{
            "role": "user",
            "content": [
                *[
                    {"type": "image", "image": p if p.startswith("http") else f"file://{p}"}
                    for p in image
                ],
                {"type": "text", "text": text},
            ],
        }]
        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[prompt], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt"
        )
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
        kwargs = {"max_new_tokens": max_new_tokens, "do_sample": do_sample}
        if do_sample:
            kwargs["temperature"] = temperature
        else:
            # The checkpoint ships sampling defaults (top_p/top_k/temperature).
            # Transformers 4.57 warns when those defaults accompany greedy
            # decoding. Give greedy decoding a clean generation config.
            generation_config = copy.deepcopy(self.model.generation_config)
            generation_config.do_sample = False
            generation_config.temperature = None
            generation_config.top_p = None
            generation_config.top_k = None
            kwargs["generation_config"] = generation_config
            # Do not re-apply the checkpoint's sampling defaults to this
            # explicitly greedy configuration (Transformers >=4.57).
            kwargs["use_model_defaults"] = False
        with __import__("torch").inference_mode():
            generated = self.model.generate(**inputs, **kwargs)
        trimmed = [out[len(inp):] for inp, out in zip(inputs["input_ids"], generated)]
        answer = self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        result = {"answer": answer}
        if plot and task in {"pointing", "grounding", "trajectory"}:
            output_path = self._plot(image[0], answer, task)
            if output_path:
                result["plot"] = output_path
        return result

    def _plot(self, path, answer, task):
        image = cv2.imread(path)
        if image is None:
            return None
        h, w = image.shape[:2]
        number = r"-?\d+(?:\.\d+)?"

        def pixel_x(value):
            value = float(value)
            scaled = value * (w if abs(value) <= 1 else w / 1000)
            return max(0, min(w - 1, round(scaled)))

        def pixel_y(value):
            value = float(value)
            scaled = value * (h if abs(value) <= 1 else h / 1000)
            return max(0, min(h - 1, round(scaled)))

        def xy(x, y):
            return pixel_x(x), pixel_y(y)

        def label(text, point, color):
            cv2.putText(
                image, text, (point[0] + 8, max(20, point[1] - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA,
            )

        if task == "pointing":
            points = re.findall(rf"\(\s*({number})\s*,\s*({number})\s*\)", answer)
            for index, (x, y) in enumerate(points, 1):
                pixel = xy(x, y)
                cv2.circle(image, pixel, 9, (0, 0, 255), -1)
                label(f"P{index} ({x}, {y})", pixel, (0, 0, 255))
        elif task == "grounding":
            boxes = re.findall(
                rf"\[\s*({number})\s*,\s*({number})\s*,\s*({number})\s*,\s*({number})\s*\]",
                answer,
            )
            for index, (x1, y1, x2, y2) in enumerate(boxes, 1):
                top_left, bottom_right = xy(x1, y1), xy(x2, y2)
                cv2.rectangle(image, top_left, bottom_right, (0, 190, 0), 3)
                label(f"B{index} [{x1}, {y1}, {x2}, {y2}]", top_left, (0, 150, 0))
        elif task == "trajectory":
            waypoints = re.findall(
                rf"\(\s*({number})\s*,\s*({number})\s*,\s*({number})\s*\)",
                answer,
            )
            pixels = [xy(x, y) for x, y, _ in waypoints]
            if len(pixels) > 1:
                cv2.polylines(image, [__import__("numpy").array(pixels)], False, (255, 0, 0), 4)
            for index, ((x, y, depth), pixel) in enumerate(zip(waypoints, pixels), 1):
                cv2.circle(image, pixel, 9, (255, 0, 0), -1)
                cv2.putText(
                    image, str(index), (pixel[0] - 5, pixel[1] + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA,
                )
            if waypoints:
                panel_height = min(h, 28 + 24 * len(waypoints))
                overlay = image.copy()
                cv2.rectangle(overlay, (8, 8), (min(w - 8, 310), panel_height), (20, 20, 20), -1)
                cv2.addWeighted(overlay, 0.72, image, 0.28, 0, image)
                for index, (x, y, depth) in enumerate(waypoints, 1):
                    cv2.putText(
                        image, f"P{index}: x={x}, y={y}, d={depth}", (16, 29 + 24 * (index - 1)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA,
                    )

        # This copy of the wrapper lives in ``01_视觉示例/示例代码``.
        # Keep annotated predictions alongside the example's input images.
        result_dir = Path(__file__).resolve().parent.parent / "标注图片"
        result_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(path).stem
        count = self._plot_counts.get(stem, 0) + 1
        self._plot_counts[stem] = count
        suffix = "" if count == 1 else f"_{count}"
        output_path = result_dir / f"{stem}{suffix}.jpg"
        cv2.imwrite(str(output_path), image)
        return str(output_path)
