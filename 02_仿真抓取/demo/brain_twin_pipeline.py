#!/usr/bin/env python3
"""End-to-end RoboBrain -> RoboTwin -> MPlib -> SAPIEN smoke test.

The brain runs in brain_env as a separate process.  The simulation process
owns the scene and planner, so the two environments remain independently
reproducible.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
import selectors
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml
from PIL import Image


BRIDGE_ROOT = Path(__file__).resolve().parent
ROBOTWIN_ROOT = Path(os.environ.get("ROBOTWIN_ROOT", "/root/gpufree-data/RoboTwin")).resolve()
os.chdir(ROBOTWIN_ROOT)  # RoboTwin asset paths are relative to its project directory.
sys.path.insert(0, str(ROBOTWIN_ROOT))
os.environ["ROBOTWIN_MINIMAL_ASSETS"] = "1"
os.environ["ROBOTWIN_SKIP_PLANNER"] = "0"

from envs.place_object_basket import place_object_basket  # noqa: E402
from demo_recording import DemoRecorder  # noqa: E402


def build_args(viewer=False):
    cfg_dir = ROBOTWIN_ROOT / "env_cfg" / "task_config"
    args = yaml.safe_load((cfg_dir / "demo_clean.yml").read_text())
    embodiment_cfg = yaml.safe_load((cfg_dir / "_embodiment_config.yml").read_text())["aloha-agilex"]
    robot_dir = ROBOTWIN_ROOT / embodiment_cfg["file_path"].lstrip("./")
    robot_cfg = yaml.safe_load((robot_dir / "config.yml").read_text())
    args.update(
        task_name="place_object_basket",
        now_ep_num=0,
        seed=0,
        # RoboTwin only creates its interactive Viewer when render_freq > 0.
        # RoboTwin advances at 250 Hz. Rendering every 10 simulation steps
        # gives about 25 FPS while leaving CPU time for physics and planning.
        render_freq=10 if viewer else 0,
        save_path=str(ROBOTWIN_ROOT / "data" / "minimal_pipeline"),
        left_robot_file=str(robot_dir),
        right_robot_file=str(robot_dir),
        left_embodiment_config=robot_cfg,
        right_embodiment_config=robot_cfg,
        dual_arm_embodied=True,
        embodiment_name="aloha-agilex",
        need_plan=True,
    )
    return args


def run_brain(image_path: Path, render_callback=None):
    brain_python = "/opt/conda/envs/brain_env/bin/python"
    bridge = BRIDGE_ROOT / "brain_bridge_infer.py"
    brain_code = "/root/Desktop/RoboBrain/01_视觉示例/示例代码"
    command = [brain_python, str(bridge), "--image", str(image_path)]
    brain_env = os.environ.copy()
    brain_env["PYTHONPATH"] = brain_code + ":" + brain_env.get("PYTHONPATH", "")
    brain_env["PYTHONUNBUFFERED"] = "1"
    print("[4/9] starting RoboBrain in brain_env ...", flush=True)
    process = subprocess.Popen(
        command,
        cwd=brain_code,
        env=brain_env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=1,
    )
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ, "stdout")
    selector.register(process.stderr, selectors.EVENT_READ, "stderr")
    stdout_lines = []
    stderr_lines = []
    deadline = __import__("time").monotonic() + 600
    while selector.get_map():
        if __import__("time").monotonic() > deadline:
            process.kill()
            raise TimeoutError("RoboBrain inference exceeded 600 seconds")
        # SAPIEN's Viewer is owned by this process. Keep pumping its native
        # window while the separate brain process is loading/inferencing.
        if render_callback is not None:
            render_callback()
        # Keep native window events and camera updates at 30 Hz while the
        # separate Python process loads and runs the vision model.
        for key, _ in selector.select(timeout=1 / 30):
            line = key.fileobj.readline()
            if line == "":
                selector.unregister(key.fileobj)
                continue
            line = line.rstrip()
            if key.data == "stdout":
                stdout_lines.append(line)
                if line.startswith("{"):
                    print("[brain] result received", flush=True)
            else:
                stderr_lines.append(line)
                print(f"[brain] {line}", flush=True)
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(
            "RoboBrain bridge failed:\n" + "\n".join(stderr_lines[-100:]) + "\n" + "\n".join(stdout_lines[-20:])
        )
    lines = [line for line in stdout_lines if line.strip()]
    if not lines:
        raise RuntimeError("RoboBrain bridge returned no output")
    payload = json.loads(lines[-1])
    return payload, "\n".join(stderr_lines[-20:])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=BRIDGE_ROOT.parent / '最近运行结果')
    parser.add_argument('--no-video', action='store_true')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--viewer', action='store_true', help='open the interactive SAPIEN Viewer')
    mode.add_argument('--headless', action='store_true', help='disable the interactive Viewer')
    parser.add_argument('--keep-viewer', action='store_true', help='keep the Viewer open after the run')
    options = parser.parse_args()
    display_available = bool(os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY'))
    viewer_enabled = options.viewer or (not options.headless and display_available)
    if options.viewer and not display_available:
        raise RuntimeError('Cannot open SAPIEN Viewer: DISPLAY/WAYLAND_DISPLAY is not set. Use --headless.')
    output_dir = options.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    stages = []
    recorder = None
    report = {
        'started_at': datetime.now().astimezone().isoformat(),
        'robotwin_task_success': False, 'execution_returned': False,
        'scene_seed': 0, 'stages': stages,
        'controller': 'RoboBrain high-level skill selection + MPlib + SAPIEN',
        'geometry_source': 'RoboTwin actor poses and grasp annotations',
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (output_dir / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    task = place_object_basket()
    try:
        print("[1/9] creating RoboTwin SAPIEN scene ...", flush=True)
        task.setup_demo(**build_args(viewer=viewer_enabled))
        print("[2/9] scene ready: robot + basket + toycar", flush=True)
        viewer_state = {"active": viewer_enabled}
        viewer_stats = {"frames": 0, "started": time.monotonic()}
        if viewer_enabled:
            viewer_obj = task.viewer
            original_render = viewer_obj.render

            def guarded_render():
                if not viewer_state["active"] or viewer_obj.window is None:
                    viewer_state["active"] = False
                    task.render_freq = 0
                    return
                try:
                    original_render()
                except AttributeError as error:
                    if "should_close" not in str(error):
                        raise
                    viewer_state["active"] = False
                    task.render_freq = 0
                    print("[viewer] window closed; continuing without live rendering", flush=True)

            # RoboTwin calls this method internally from every dense-action
            # step, so the guard also covers a Viewer closed by the desktop.
            viewer_obj.render = guarded_render
            original_update_render = task._update_render
            sensor_clock = {"last": 0.0}

            def throttled_update_render(*args, **kwargs):
                # Keep the SAPIEN window responsive at the Viewer cadence;
                # camera readback is more expensive, so limit it to 15 Hz.
                now = time.monotonic()
                if now - sensor_clock["last"] >= 1 / 15:
                    original_update_render(*args, **kwargs)
                    sensor_clock["last"] = now

            task._update_render = throttled_update_render

        def render_viewer():
            if viewer_state["active"]:
                task._update_render()
                task.viewer.render()
                viewer_stats["frames"] += 1

        def plan_and_execute(arm_tag, target_pose, label):
            """Plan off the Viewer thread, then execute the returned path."""
            planner = task.robot.left_plan_path if arm_tag == "left" else task.robot.right_plan_path
            plan_started = time.monotonic()
            print(f"[planner] {label}: planning in background ...", flush=True)
            with ThreadPoolExecutor(max_workers=1, thread_name_prefix="mplib") as executor:
                future = executor.submit(planner, target_pose)
                while not future.done():
                    render_viewer()
                    time.sleep(1 / 30)
                result = future.result()
            elapsed = time.monotonic() - plan_started
            print(
                f"[planner] {label}: {result.get('status', 'Fail')} in {elapsed:.2f}s; executing ...",
                flush=True,
            )
            if result.get("status") != "Success":
                return False, result
            if arm_tag == "left":
                task.left_joint_path.append(result)
                control_seq = {
                    "left_arm": result, "left_gripper": None,
                    "right_arm": None, "right_gripper": None,
                }
            else:
                task.right_joint_path.append(result)
                control_seq = {
                    "left_arm": None, "left_gripper": None,
                    "right_arm": result, "right_gripper": None,
                }
            task.take_dense_action(control_seq)
            return True, result

        def settle_scene(steps):
            for step in range(steps):
                task.scene.step()
                if step % 5 == 0:
                    render_viewer()

        if viewer_enabled:
            render_viewer()
            print('[viewer] interactive SAPIEN window opened (target 25 FPS)', flush=True)
        observation = task.get_obs()
        rgb = np.asarray(observation["observation"]["head_camera"]["rgb"])
        image_path = output_dir / 'head_camera.png'
        image_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rgb).save(image_path)
        if not options.no_video:
            recorder = DemoRecorder(task, output_dir)
            recorder.capture('initial_scene')
        print(f"[3/9] captured head camera: {image_path}", flush=True)

        brain_payload, brain_stderr = run_brain(image_path, render_callback=render_viewer)
        brain_answer = str(brain_payload.get("answer", ""))
        # Parse the brain's nested JSON answer.  The adapter executes the
        # selected skill against real SAPIEN actors and returns contact data.
        decision = json.loads(brain_answer)
        if not isinstance(decision, dict) or decision.get('action') != 'pick_and_place' or decision.get('target') != 'toycar_to_basket':
            raise ValueError(f'Brain returned unsupported action; no motion executed: {brain_answer}')
        brain_action = decision['action']
        report['brain_answer'] = brain_answer
        print(f"[5/9] adapter received action={brain_action!r}; planning with MPlib ...", flush=True)
        arm = str(task.arm_tag)
        # Capture a deliberate carry pose BEFORE moving to the object. The
        # successful diagnostic precomputed this pose; computing displacement
        # after grasp instead selects a different, unreachable placing posture.
        carry_pose = np.asarray(task.get_arm_pose(arm), dtype=np.float64).copy()
        carry_pose[2] += 0.1
        plan = {"status": "Success", "position": np.empty((0, 6))}
        executed = False
        contact_count = 0
        def record_stage(name):
            state = {"stage": name, "object": task.object.get_pose().p.tolist(), "basket": task.basket.get_pose().p.tolist(), "ee": list(task.get_arm_pose(arm)), "gripper_contacts": len(task.get_gripper_actor_contact_position(task.object.get_name()))}
            stages.append(state)
            print('[stage] ' + json.dumps(state), flush=True)
            if recorder:
                recorder.stage = name
                recorder.capture(name)
        place_planned = False
        release_returned = False
        object_basket_contact = False
        object_table_contact = False
        basket_lift_returned = False
        if brain_action == "pick_and_place":
            pre_pose, grasp_pose = task.choose_grasp_pose(task.object, task.arm_tag, pre_dis=0.1)
            if pre_pose is None or grasp_pose is None:
                raise RuntimeError("RoboTwin could not generate a grasp pose")
            executed, plan = plan_and_execute(arm, pre_pose, "pre-grasp")
            record_stage('pre_grasp')
            if executed:
                executed, plan = plan_and_execute(arm, grasp_pose, "grasp")
            executed = executed and bool(task.move(task.close_gripper(arm)))
            record_stage('closed_gripper')
            if executed:
                executed, plan = plan_and_execute(arm, carry_pose, "carry")
            record_stage('carry')
            contact_count = len(task.get_gripper_actor_contact_position(task.object.get_name()))

            # Keep the object-to-gripper offset measured after grasping.  Aiming
            # the gripper at the basket center directly can push the dynamic
            # basket away and leave the object outside it.
            if executed and contact_count:
                ee_pose = np.asarray(task.get_arm_pose(arm), dtype=np.float64)
                object_offset = np.asarray(task.object.get_pose().p, dtype=np.float64) - ee_pose[:3]
                basket_position = np.asarray(task.basket.get_pose().p, dtype=np.float64)
                target_candidates = []
                for object_height in (0.24, 0.18, 0.12):
                    target_pose = ee_pose.copy()
                    target_pose[:3] = basket_position - object_offset + np.array([0.0, 0.0, object_height])
                    target_candidates.append(target_pose)
                for target_pose in target_candidates:
                    moved, candidate = plan_and_execute(arm, target_pose, "place above basket")
                    if moved:
                        plan = candidate
                        place_planned = True
                        record_stage('above_basket')
                        break
                if place_planned:
                    release_returned = bool(task.move(task.open_gripper(arm)))
                    settle_scene(500)
                    object_basket_contact = bool(task.check_actors_contact(task.object_name, task.basket_name))
                    object_table_contact = bool(task.check_actors_contact(task.object_name, "table"))
                    record_stage('released_and_settled')

                    # RoboTwin's official success condition also requires the
                    # basket to be lifted. Reuse the same arm so the demo does
                    # not add a second-arm IK dependency after the place.
                    if release_returned and object_basket_contact:
                        basket_arm = arm
                        basket_pre, basket_grasp = task.choose_grasp_pose(
                            task.basket, basket_arm, pre_dis=0.10, target_dis=0.02
                        )
                        if basket_pre is not None and basket_grasp is not None:
                            basket_lift_returned, basket_plan = plan_and_execute(
                                basket_arm, basket_pre, "basket pre-grasp"
                            )
                            if basket_lift_returned:
                                record_stage('basket_pre_grasp')
                                basket_lift_returned, basket_grasp_plan = plan_and_execute(
                                    basket_arm, basket_grasp, "basket grasp"
                                )
                                if not basket_lift_returned:
                                    raise RuntimeError('Basket grasp planning failed')
                                basket_lift_returned = basket_lift_returned and bool(task.move(task.close_gripper(basket_arm)))
                                record_stage('basket_grasp')
                                lift_pose = np.asarray(task.get_arm_pose(basket_arm), dtype=np.float64).copy()
                                lift_pose[2] += 0.08
                                if basket_lift_returned:
                                    basket_lift_returned, plan = plan_and_execute(
                                        basket_arm, lift_pose, "basket lift"
                                    )
                                settle_scene(1000)
                                record_stage('basket_lift')
        else:
            current_pose = np.asarray(task.get_arm_pose(arm), dtype=np.float64)
            target_pose = current_pose.copy()
            target_pose[0] += 0.03
            executed, plan = plan_and_execute(arm, target_pose, "bridge motion")
            if not executed:
                raise RuntimeError(f"MPlib could not plan bridge action: {plan}")
        after = task.get_obs()
        print(f"[6/9] pick/place trajectory returned: {executed}; new RGB shape={np.asarray(after['observation']['head_camera']['rgb']).shape}", flush=True)

        task_success = bool(task.check_success())
        object_basket_contact = bool(task.check_actors_contact(task.object_name, task.basket_name))
        object_table_contact = bool(task.check_actors_contact(task.object_name, 'table'))
        print(f"[7/9] release={release_returned}; object_basket_contact={object_basket_contact}; object_table_contact={object_table_contact}", flush=True)
        print(f"[8/9] basket lift returned: {basket_lift_returned}", flush=True)
        print(f"[9/9] SAPIEN task state returned; task_success={task_success}", flush=True)
        motion_plans = task.left_joint_path + task.right_joint_path
        report.update({
            "tested_at": datetime.now().astimezone().isoformat(),
            "stages": stages,
            "brain_answer": brain_answer,
            "brain_output_received": bool(brain_answer),
            "adapter_action": brain_action,
            "planner": type(task.robot.left_planner).__name__,
            "plan_status": 'Success' if task.plan_success else 'Fail',
            "planned_steps": sum(len(p.get('position', [])) for p in motion_plans),
            "executed_motion_plans": len(motion_plans),
            "approximate_z_offsets": [float(p.get('approximate_z_offset', 0)) for p in motion_plans],
            "execution_returned": bool(executed and place_planned and release_returned and basket_lift_returned),
            "gripper_contact_count": contact_count,
            "place_planned": place_planned,
            "release_returned": release_returned,
            "object_basket_contact": object_basket_contact,
            "object_table_contact": object_table_contact,
            "basket_lift_returned": basket_lift_returned,
            "object_position": np.asarray(task.object.get_pose().p).round(6).tolist(),
            "basket_position": np.asarray(task.basket.get_pose().p).round(6).tolist(),
            "post_action_rgb_shape": list(np.asarray(after["observation"]["head_camera"]["rgb"]).shape),
            "robotwin_task_success": task_success,
            "brain_stderr_tail": brain_stderr,
            "basket_lift_m": float(task.basket.get_pose().p[2] - task.start_height),
            "object_lift_m": float(task.object.get_pose().p[2] - task.object_start_height),
            "viewer_frames": viewer_stats["frames"],
            "viewer_fps_average": round(
                viewer_stats["frames"] / max(time.monotonic() - viewer_stats["started"], 1e-6), 2
            ) if viewer_enabled else 0.0,
        })
        Image.fromarray(np.asarray(after['observation']['head_camera']['rgb'])).save(image_path.parent / 'final_camera.png')
        (image_path.parent / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        if not task_success:
            raise RuntimeError('Official RoboTwin task check failed; see data/minimal_pipeline/report.json')
        if viewer_state["active"] and options.keep_viewer:
            input('\n仿真已完成，Viewer 保持打开。按回车关闭…')
    except BaseException as error:
        report['error'] = f'{type(error).__name__}: {error}'
        report['robotwin_task_success'] = False
        report['execution_returned'] = False
        raise
    finally:
        try:
            if recorder:
                recorder.close()
                report['video_frames'] = recorder.frames
                report['video'] = str(output_dir / 'demo.mp4')
        finally:
            report['elapsed_seconds'] = round(time.monotonic() - started, 3)
            report['viewer_enabled'] = viewer_enabled
            (output_dir / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
            print(json.dumps({k: v for k, v in report.items() if k != 'stages'}, ensure_ascii=False, indent=2), flush=True)
            task.close_env(clear_cache=True)


if __name__ == "__main__":
    main()
