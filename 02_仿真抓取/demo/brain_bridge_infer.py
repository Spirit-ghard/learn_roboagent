#!/usr/bin/env python3
"""Run one RoboBrain inference for the simulation bridge.

This file is launched with brain_env's Python so RoboTwin does not need to
share Python packages with RoboBrain.
"""

import argparse
import json
import sys
from pathlib import Path

BRAIN_CODE = Path("/root/Desktop/RoboBrain/01_视觉示例/示例代码")
sys.path.insert(0, str(BRAIN_CODE))
from inference import UnifiedInference


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("/root/gpufree-data/models/RoboBrain2.5-4B"))
    args = parser.parse_args()
    prompt = (
        "You are the high-level controller of a simulated robot. Inspect the image and "
        "return one JSON object only, with action chosen from [move, pick_and_place, wait]. "
        "If a robot, basket, and toy car are visible, choose pick_and_place. "
        "The JSON schema is {\"action\":\"pick_and_place\",\"target\":\"toycar_to_basket\"}."
    )
    print("loading RoboBrain model ...", file=sys.stderr, flush=True)
    model = UnifiedInference(str(args.model), device_map="auto")
    print("running RoboBrain inference ...", file=sys.stderr, flush=True)
    result = model.inference(
        str(prompt), str(args.image), task="general", do_sample=False, max_new_tokens=64
    )
    answer = result.get("answer", result) if isinstance(result, dict) else result
    print(json.dumps({"answer": answer}, ensure_ascii=False))


if __name__ == "__main__":
    main()
