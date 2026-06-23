from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from signalforge.server.paper_api import (
    load_paper_ledger,
    load_paper_positions,
    load_paper_summary,
)
from signalforge.server.runner import (
    get_registered_scripts,
    get_run_state,
    run_script,
    stream_output,
)

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent.parent
_WEB_DIST = _REPO_ROOT / "web" / "dist"
_PUBLIC_DATA = _WEB_DIST / "data"


def create_app() -> FastAPI:
    app = FastAPI(
        title="SignalForge API",
        version="0.1.0",
        description="Interactive backend for the SignalForge paper-trading dashboard.",
    )

    _mount_static(app)
    _register_routes(app)
    return app


def _mount_static(app: FastAPI) -> None:
    """Serve the built dashboard and its data directory."""
    if _WEB_DIST.exists():
        app.mount("/assets", StaticFiles(directory=str(_WEB_DIST / "assets")), name="assets")

    if _PUBLIC_DATA.exists():
        app.mount("/data", StaticFiles(directory=str(_PUBLIC_DATA)), name="data")


def _register_routes(app: FastAPI) -> None:

    @app.get("/api/health")
    async def health():
        return {"status": "ok", "version": "0.1.0"}

    @app.get("/api/scripts")
    async def list_scripts():
        return get_registered_scripts()

    @app.post("/api/scripts/{name}/run")
    async def trigger_script(name: str, request: Request):
        body = await request.json() if request.headers.get("content-type") == "application/json" else {}
        overrides = body.get("args", {}) if isinstance(body, dict) else {}
        try:
            run_id = await run_script(name, overrides=overrides)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        return {"run_id": run_id}

    @app.get("/api/runs/{run_id}")
    async def get_run(run_id: str):
        state = get_run_state(run_id)
        if state is None:
            raise HTTPException(status_code=404, detail="run not found")
        return {
            "run_id": state.run_id,
            "script_name": state.script_name,
            "status": state.status.value,
            "started_at": state.started_at.isoformat() if state.started_at else None,
            "finished_at": state.finished_at.isoformat() if state.finished_at else None,
            "exit_code": state.exit_code,
            "output": state.output,
        }

    @app.get("/api/runs/{run_id}/stream")
    async def stream_run(run_id: str):
        state = get_run_state(run_id)
        if state is None:
            raise HTTPException(status_code=404, detail="run not found")

        async def event_generator():
            async for line in stream_output(run_id):
                if line is None:
                    yield {"event": "done", "data": json.dumps({"status": state.status.value})}
                    return
                yield {"event": "message", "data": line}

        return EventSourceResponse(event_generator())

    @app.get("/api/paper/summary")
    async def paper_summary():
        return load_paper_summary()

    @app.get("/api/paper/positions")
    async def paper_positions():
        return load_paper_positions()

    @app.get("/api/paper/ledger")
    async def paper_ledger():
        return load_paper_ledger()

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        """Serve the SPA for all non-API routes."""
        index = _WEB_DIST / "index.html"
        if index.exists():
            return FileResponse(str(index))
        return {"error": "dashboard not built; run 'npm run build' in web/"}


app = create_app()


def main() -> None:
    """CLI entry point: python -m signalforge.server.app"""
    import uvicorn
    uvicorn.run("signalforge.server.app:app", host="127.0.0.1", port=8080, reload=False)


if __name__ == "__main__":
    main()
