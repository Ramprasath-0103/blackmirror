from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

import config
from web import content, explorer
from web.jobs import PHASES, JobRunner

STATIC_DIR = Path(__file__).resolve().parent / "static"


def create_app() -> Flask:
    app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path="/static")
    app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0
    jobs = JobRunner()

    def error(message: str, code: int):
        return jsonify({"error": message}), code

    @app.get("/")
    def index():
        return send_from_directory(STATIC_DIR, "index.html")

    @app.get("/api/status")
    def status():
        return jsonify({**content.status(), "job_running": jobs.running})

    @app.get("/api/settings")
    def settings():
        return jsonify(content.settings())

    @app.get("/api/overview")
    def overview():
        return jsonify(content.overview())

    @app.get("/api/results")
    def results():
        return jsonify(content.results())

    @app.get("/api/dataset")
    def dataset():
        generation = request.args.get("generation", default=config.N_GENERATIONS, type=int)
        if not 0 <= generation <= config.N_GENERATIONS:
            return error("Unknown generation.", 400)
        return jsonify(content.dataset(generation))

    @app.get("/api/trace")
    def trace():
        generation = request.args.get("generation", type=int)
        ratio = request.args.get("ratio", type=float)
        valid = (generation == 0 and ratio == 1.0) or (
            generation is not None and 1 <= generation <= config.N_GENERATIONS and ratio in config.RATIOS)
        if not valid:
            return error("Pick generation 0 with 100% real data, or generation 1-10 with one of the ratios.", 400)
        if jobs.running:
            return error("The pipeline is running. Trace a run once it has finished.", 409)
        if not content.status()["phases"][0]["done"]:
            return error("There is no data yet. Run phase 1 (Data & generation) first.", 409)
        return jsonify(explorer.trace(generation, ratio))

    @app.post("/api/run")
    def run():
        body = request.get_json(silent=True) or {}
        phases = body.get("phases", [])
        if not phases or any(p not in PHASES for p in phases):
            return error("Unknown phase.", 400)
        try:
            jobs.start(phases, fresh=bool(body.get("fresh")))
        except RuntimeError as exc:
            return error(str(exc), 409)
        return jsonify({"started": phases})

    @app.post("/api/stop")
    def stop():
        jobs.stop()
        return jsonify({"stopping": True})

    @app.get("/api/job")
    def job():
        return jsonify(jobs.snapshot(since=request.args.get("since", default=0, type=int)))

    return app
