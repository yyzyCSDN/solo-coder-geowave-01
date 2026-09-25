from __future__ import annotations
import os
from fastapi import FastAPI
from geowave.service.state import create_state
from geowave.service.api import build_router

def create_app(root: str | None=None) -> FastAPI:
    app = FastAPI(title='GeoWave Seismic Lab', version='1.0.0')
    state = create_state(root or os.getenv('GEOWAVE_DATA_DIR'))
    app.state.geowave = state
    app.include_router(build_router(state))
    return app
app = create_app()

def main() -> None:
    import uvicorn
    uvicorn.run('geowave.service.app:app', host='127.0.0.1', port=int(os.getenv('PORT', '8080')))
