# Deploying Meridian to Render

This folder adds a small API layer and a static frontend on top of your
existing LangGraph pipeline. It doesn't change any of your pipeline code
(`graph.py`, `planner_node.py`, `weather_node.py`, `tools.py`, `tide_tool.py`,
etc.) — it just calls it.

## 1. Files in this bundle

| File | Purpose |
|---|---|
| `server.py` | FastAPI app. Serves `static/index.html` at `/` and runs the graph via `POST /api/ask`. |
| `static/index.html` | The frontend — single file, no build step. |
| `requirements.txt` | Dependencies for the API server + your existing pipeline (drops `streamlit`/`pydeck`, which the new frontend replaces). |
| `render.yaml` | Render Blueprint — lets you deploy with "New +  → Blueprint" instead of clicking through settings by hand. |
| `Procfile` | Fallback start command if you deploy without the blueprint. |
| `.env.example` | Which environment variables to set, and where they're already defaulted. |

## 2. Put it together

Copy `server.py`, `static/`, `requirements.txt`, `render.yaml`, `Procfile`,
and `.env.example` into the **root of your existing repo**, alongside
`graph.py`, `state.py`, `planner_node.py`, `weather_node.py`, `ocean_node.py`,
`tide_node.py`, `tide_tool.py`, `cyclone_node.py`, `ecosystem_node.py`,
`ecosystem_tool.py`, `pfz_node.py`, `pfz_tool.py`, `gis_node.py`,
`gis_tool.py`, `tools.py`, `schemas.py`, and `recommendation_node.py`.
`server.py` imports `marine_graph` straight from `graph.py`, so nothing else
needs to move.

Your old `app.py` (the Streamlit UI) and its own `requirements.txt` can stay
in the repo untouched — this deployment simply doesn't use them. If you
want to keep both UIs, add `streamlit` and `pydeck` back to
`requirements.txt`.

## 3. Push to GitHub, then deploy

1. Commit and push the merged repo to GitHub (public or private — Render
   supports both).
2. In the Render dashboard: **New +** → **Blueprint** → point it at the repo.
   Render reads `render.yaml` and creates the web service automatically.
   - If you'd rather not use the blueprint, create a **Web Service** by hand:
     Build command `pip install -r requirements.txt`, start command
     `uvicorn server:app --host 0.0.0.0 --port $PORT`.
3. Set the required environment variable in the service's **Environment**
   tab: `GROQ_API_KEY` (see `.env.example` for the full list — the marine
   data service URLs already have working defaults baked into
   `render.yaml`).
4. Deploy. Render assigns a URL like
   `https://meridian-marine-intelligence.onrender.com` — that's your whole
   app, frontend and API together, nothing else to host.

## 4. Notes

- The free/starter Render plan spins down when idle, so the first request
  after a period of inactivity can take 30–60 seconds — the frontend's
  loading state accounts for this.
- `POST /api/ask` runs the full graph synchronously and returns once the
  final recommendation is ready (typically 15–40s depending on which
  specialist agents the planner selects). If you later want live streaming
  step-by-step updates in the UI, that endpoint is the place to swap in
  Server-Sent Events using the same `marine_graph.stream(...)` call already
  used inside it.
- `GET /api/health` is wired up as the Render health check.
