"""Spawning the C++ video recorder alongside a deploy run.

Moved out of ``examples/19_deploy_policy.py``. Recording runs as a separate C++
process for the same reason the camera bridge does: an in-process Python recorder
lost frames under the rclpy executor and GIL contention that the deploy loop is
already fighting.
"""

import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


def _find_crisp_video_recorder_binary() -> list[str]:
    """Return argv prefix to launch crisp_video_recorder.

    Mirrors crisp_gym.deploy.cpp_sender._build_subprocess_argv discovery: prefer
    known install path, fall back to `ros2 run tum09_custom ...` which
    works if the user has the workspace setup.bash sourced.
    """
    candidate = Path(
        "/home/ali/Coding/Robot_Control/clearpath_remote_ws/install/"
        "tum09_custom/lib/tum09_custom/crisp_video_recorder"
    )
    if candidate.exists():
        return [str(candidate)]
    return ["ros2", "run", "tum09_custom", "crisp_video_recorder"]


class _VideoRecorder:
    """Spawn + manage the crisp_video_recorder C++ subprocess.

    Args:
        camera: a crisp_py.camera.Camera (used only for its config —
            the C++ binary subscribes to the topic directly, no IPC
            handoff of pixel data).
        out_path: mp4 file path. Folder must exist.
        fps: declared playback fps for the writer.
        log_path: optional file to redirect the subprocess's stderr/stdout
            into so any open() failure or runtime error is captured.

    Usage:
        rec = _VideoRecorder(camera, out_path, fps=20.0, log_path=...)
        rec.start()
        ...
        rec.stop(timeout=5.0)
    """

    def __init__(
        self,
        camera,
        out_path: Path,
        fps: float = 20.0,
        log_path: Path | None = None,
    ) -> None:
        self.camera = camera
        self.out_path = out_path
        self.fps = max(0.1, float(fps))
        self.log_path = log_path
        self._proc: subprocess.Popen | None = None
        self._log_fh = None

    def start(self) -> None:
        base = _find_crisp_video_recorder_binary()
        topic = self.camera.config.camera_color_image_topic
        argv = [
            *base,
            "--topic", str(topic),
            "--out",   str(self.out_path),
            "--fps",   f"{self.fps:.4f}",
        ]
        if self.log_path is not None:
            self._log_fh = open(self.log_path, "w")
            stdout = self._log_fh
            stderr = subprocess.STDOUT
        else:
            stdout = None
            stderr = None
        logger.info("spawning crisp_video_recorder: %s", " ".join(argv))
        self._proc = subprocess.Popen(
            argv, stdout=stdout, stderr=stderr,
        )

    def is_alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def stop(self, timeout: float = 5.0) -> None:
        if self._proc is None:
            return
        if self._proc.poll() is None:
            # SIGINT triggers rclcpp::shutdown → executor.spin() returns →
            # node destructor releases cv::VideoWriter with the lock held,
            # which flushes the mp4 trailer and closes the file. That's
            # what makes the output a valid playable mp4.
            self._proc.send_signal(2)  # SIGINT
            try:
                self._proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                logger.warning(
                    "crisp_video_recorder did not exit on SIGINT within "
                    "%.1fs; sending SIGTERM", timeout,
                )
                self._proc.terminate()
                try:
                    self._proc.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    self._proc.kill()
        rc = self._proc.returncode
        if rc != 0:
            logger.warning(
                "crisp_video_recorder exited with rc=%d (see %s)",
                rc, self.log_path or "stdout",
            )
        if self._log_fh is not None:
            try:
                self._log_fh.close()
            except Exception:
                pass
            self._log_fh = None
