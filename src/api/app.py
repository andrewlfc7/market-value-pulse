from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.repository import build_repository, read_metadata
from api.profile_store import ProfileStore

app = FastAPI(title="Market Value Pulse API", version="0.6.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["GET"],
    allow_headers=["*"],
)
repository = build_repository()
profile_store = ProfileStore()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/players")
def players() -> dict[str, object]:
    rows = repository.players()
    return {"data": rows, "count": len(rows)}


@app.get("/api/players/{player_id}")
def player(player_id: str) -> dict[str, object]:
    row = repository.player(player_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Player not found")
    return {"data": row}


@app.get("/api/players/{player_id}/profile")
def player_profile(player_id: str) -> dict[str, object]:
    row = profile_store.player_profile(player_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Player profile not found")
    return {"data": row}


@app.get("/api/players/{player_id}/similar-players")
def similar_players(player_id: str, limit: int = 10) -> dict[str, object]:
    rows = profile_store.similar_players(
        player_id,
        limit=max(1, min(limit, 25)),
    )
    return {"data": rows, "count": len(rows)}


@app.get("/api/catalog")
def catalog() -> dict[str, object]:
    return {"data": read_metadata("catalog")}


@app.get("/api/lineage")
def lineage() -> dict[str, object]:
    return {"data": read_metadata("lineage")}


@app.get("/api/pipeline-runs")
def pipeline_runs(limit: int = 25) -> dict[str, object]:
    rows = repository.pipeline_runs(limit=max(1, min(limit, 100)))
    return {"data": rows, "count": len(rows)}
