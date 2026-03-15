from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import stocks, news, watchlist
from db import init_db
import uvicorn

app = FastAPI(title="Nifty Trading Assistant API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup():
    init_db()

app.include_router(stocks.router, prefix="/api/stocks", tags=["stocks"])
app.include_router(news.router, prefix="/api/news", tags=["news"])
app.include_router(watchlist.router, prefix="/api/watchlist", tags=["watchlist"])

@app.get("/api/health")
def health():
    return {"status": "ok", "mode": "phase1"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)