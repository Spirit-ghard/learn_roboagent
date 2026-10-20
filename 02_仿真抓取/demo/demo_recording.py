"""Record RGB evidence via RoboTwin's existing trajectory picture hook."""
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


class DemoRecorder:
    def __init__(self, task, output_dir):
        self.task = task
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.stage = 'scene'
        self.frames = 0
        self.encoder = None
        self.original_picture = task._take_picture
        self.original_frequency = task.save_freq
        self.log = (self.output_dir / 'video_encoder.log').open('w')
        task._take_picture = self.capture
        task.save_freq = 50

    def capture(self, snapshot=None):
        task = self.task
        task._update_render()
        camera = task.cameras.static_camera_list[task.cameras.head_camera_id]
        camera.take_picture()
        rgb = (camera.get_picture('Color')[:, :, :3] * 255).clip(0, 255).astype(np.uint8)
        img = Image.fromarray(rgb)
        draw = ImageDraw.Draw(img)
        draw.rectangle((0, 0, img.width, 20), fill='black')
        draw.text((5, 4), self.stage, fill='white')
        if self.encoder is None:
            self.encoder = subprocess.Popen([
                'ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
                '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{img.width}x{img.height}',
                '-r', '5', '-i', '-', '-an', '-c:v', 'libx264', '-crf', '20',
                '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
                str(self.output_dir / 'demo.mp4'),
            ], stdin=subprocess.PIPE, stderr=self.log)
        self.encoder.stdin.write(np.asarray(img).tobytes())
        self.frames += 1
        if snapshot:
            img.save(self.output_dir / f'{snapshot}.png')
        return rgb

    def close(self):
        self.task._take_picture = self.original_picture
        self.task.save_freq = self.original_frequency
        code = 0
        if self.encoder is not None:
            self.encoder.stdin.close()
            code = self.encoder.wait(timeout=30)
        self.log.close()
        if code:
            raise RuntimeError('Video encoding failed; see video_encoder.log')
