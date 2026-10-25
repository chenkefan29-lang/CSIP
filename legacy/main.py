from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from simulator import simulate_treatment

app = FastAPI(title="污水处理模拟系统")

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR / "frontend"

app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/")
def home():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/api/simulate")
def get_simulation(
    scene: str = "normal",
    aeration: float = 3.0,
    recycle_ratio: float = 1.0,
):
    if scene not in ["normal", "shock"]:
        scene = "normal"

    return simulate_treatment(scene, aeration, recycle_ratio)