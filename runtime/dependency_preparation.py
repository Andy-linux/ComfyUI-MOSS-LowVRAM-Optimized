"""Explicit preparation in the same Python that runs ComfyUI."""
import json
from pathlib import Path
import subprocess
import sys
from threading import Lock
import uuid


class DependencyPreparation:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.lock = Lock()
        self.process = None
        self.job_id = None
        self.result_path = None
        self.log_path = None

    def state(self):
        if self.process is None:
            return {"state": "idle", "restart_required": False}
        code = self.process.poll()
        log = self.log_path.read_text(encoding="utf-8", errors="replace")[-12000:]
        state = {"job_id": self.job_id, "state": "running" if code is None else "failed",
                 "restart_required": False, "log": log}
        if code is not None:
            if self.result_path.is_file():
                result = json.loads(self.result_path.read_text(encoding="utf-8"))
                state.update(state="complete" if code == 0 else "failed", result=result,
                             restart_required=bool(result["restart_required"]))
            else:
                state["error"] = f"环境准备退出（{code}）。详细日志见下方；可再次点击准备继续。"
        return state

    def start(self):
        with self.lock:
            if self.process is not None and self.process.poll() is None:
                return self.state()
            self.directory.mkdir(parents=True, exist_ok=True)
            self.job_id = uuid.uuid4().hex
            self.result_path = self.directory / (self.job_id + ".json")
            self.log_path = self.directory / (self.job_id + ".log")
            installer = Path(__file__).resolve().parents[1] / "install.py"
            with self.log_path.open("wb") as log:
                self.process = subprocess.Popen(
                    [sys.executable, "-s", "-u", str(installer), "--result-file", str(self.result_path)],
                    cwd=installer.parent, stdout=log, stderr=subprocess.STDOUT,
                )
            return self.state()
