import os
import re
import subprocess
import sys
import threading
import time

import config

PHASES = {
    "prepare": {"title": "Data & generation", "script": "prepare_data.py"},
    "experiment": {"title": "Search & evaluation", "script": "run_experiment.py"},
    "analyze": {"title": "Analysis", "script": "analyze.py"},
}

# "[12/62] ..." from run_experiment.py, "Step 2/6 ..." from the other two scripts
PROGRESS_PATTERN = re.compile(r"^\[\s*(\d+)/(\d+)\]|^Step (\d+)/(\d+)")


class JobRunner:
    """Runs the command-line scripts in the background, one at a time, and keeps their output.

    The web page uses exactly the same scripts as the command line, so both always agree.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._process = None
        self._stop_requested = False
        self._state = {"status": "idle", "phases": [], "current": None,
                       "progress": None, "lines": [], "started": None, "finished": None}

    @property
    def running(self) -> bool:
        return self._state["status"] == "running"

    def start(self, phases: list[str], fresh: bool) -> None:
        with self._lock:
            if self.running:
                raise RuntimeError("Something is already running. Wait for it to finish or stop it.")
            self._stop_requested = False
            self._state = {"status": "running", "phases": phases, "current": None,
                           "progress": None, "lines": [], "started": time.time(), "finished": None}
        threading.Thread(target=self._run, args=(phases, fresh), daemon=True).start()

    def stop(self) -> None:
        with self._lock:
            self._stop_requested = True
            if self._process and self._process.poll() is None:
                self._process.terminate()

    def snapshot(self, since: int = 0) -> dict:
        with self._lock:
            state = dict(self._state)
            state["line_count"] = len(state["lines"])
            state["lines"] = state["lines"][since:]
            return state

    def _add_line(self, line: str) -> None:
        with self._lock:
            self._state["lines"].append(line)
            match = PROGRESS_PATTERN.match(line.strip())
            if match:
                done, total = (match.group(1), match.group(2)) if match.group(1) else (match.group(3), match.group(4))
                self._state["progress"] = {"done": int(done), "total": int(total)}

    def _finish(self, status: str) -> None:
        with self._lock:
            self._state["status"] = status
            self._state["finished"] = time.time()
            self._process = None

    def _run(self, phases: list[str], fresh: bool) -> None:
        for phase in phases:
            if phase == "experiment" and fresh and config.RESULTS_CSV.exists():
                config.RESULTS_CSV.unlink()
                self._add_line("Deleted results/results.csv to start the search from scratch.")

            script = PHASES[phase]["script"]
            with self._lock:
                self._state["current"] = phase
                self._state["progress"] = None
            self._add_line(f"$ python {script}")

            process = subprocess.Popen(
                [sys.executable, "-u", script], cwd=config.PROJECT_DIR,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                env={**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONWARNINGS": "ignore"},
            )
            with self._lock:
                self._process = process
            for line in process.stdout:
                self._add_line(line.rstrip("\n"))
            return_code = process.wait()

            if self._stop_requested:
                self._add_line("Stopped.")
                return self._finish("stopped")
            if return_code != 0:
                self._add_line(f"{script} failed (exit code {return_code}).")
                return self._finish("failed")
            self._add_line("")

        self._finish("done")
