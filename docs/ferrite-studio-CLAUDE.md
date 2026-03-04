# CLAUDE.md — ferrite-studio

> Copy this file to the root of the `ferrite-studio` repository as `CLAUDE.md`.
> It gives every Claude Code session (and all subagents) full context about this project.

---

## What this project is

`ferrite-studio` is the web platform layer for `ferrite-nn` — a from-scratch Rust neural network library. It consists of:

1. **React SPA** (`frontend/`) — Vite + React 19 + TypeScript + shadcn/ui + Recharts
2. **Python FastAPI** (`api/`) — authentication, user accounts, model registry, request proxy to the Rust service
3. **Docker Compose** (`docker/`) — orchestrates all services for team deployment

It is intentionally **separate** from the `ferrite-nn` repository. The Rust training/inference logic lives there; this repo contains only the Python and TypeScript layers.

---

## Repository structure

```
ferrite-studio/
  frontend/                ← React SPA
    package.json
    vite.config.ts         ← dev proxy: /api → localhost:7878 (Rust service)
    tsconfig.json
    tailwind.config.ts
    src/
      api/                 ← typed fetch wrappers for every endpoint
      hooks/               ← useSSE.ts, useModels.ts, etc.
      pages/               ← ArchitectPage, DatasetPage, TrainPage, EvaluatePage, TestPage
      components/
        architect/
        dataset/
        train/             ← LiveLossChart.tsx uses SSE
        evaluate/
        test/              ← InputModeToggle.tsx, CanvasDraw.tsx, ResultCard.tsx
        ui/                ← shadcn/ui generated components
  api/                     ← Python FastAPI service
    main.py
    auth/                  ← JWT, OAuth (GitHub + Google)
    models/                ← SQLAlchemy ORM models
    routes/                ← auth.py, proxy.py, models.py, sharing.py
    requirements.txt
  docker/
    docker-compose.yml
    nginx.conf
    Dockerfile.api
    Dockerfile.frontend
  CLAUDE.md                ← this file
  README.md
```

---

## The Rust backend (ferrite-studio does NOT own this code)

The `ferrite-nn` repository runs a Rust binary (`cargo run --bin studio --release`) that serves the REST API on `http://127.0.0.1:7878`.

**Do not modify Rust code from this repo.** If a backend change is needed, it belongs in the `ferrite-nn` repository.

The full API reference lives at:
`../ferrite-nn/docs/api-reference.md` (sibling directory)

Key points:
- **All endpoints are prefixed `/api/`**
- Default: `http://127.0.0.1:7878`
- In production: FastAPI proxies `/api/*` to the Rust service after auth
- The Rust service is stateless regarding users — FastAPI injects `X-User-Id` header

---

## REST API — quick reference

See `../ferrite-nn/docs/api-reference.md` for the full reference. Summary:

| Method | Path | Description |
|---|---|---|
| GET | `/api/architect` | Current network spec + hyperparams |
| POST | `/api/architect/save` | Save architecture (JSON body) |
| GET | `/api/dataset` | Current dataset state |
| POST | `/api/dataset/upload` | Upload CSV (multipart) |
| POST | `/api/dataset/upload-idx` | Upload IDX binary files (multipart) |
| POST | `/api/dataset/builtin` | Load built-in dataset (JSON body) |
| GET | `/api/train` | Training status + epoch history |
| POST | `/api/train/start` | Start training |
| POST | `/api/train/stop` | Stop training |
| GET | `/api/train/events` | SSE stream (real-time epoch stats) |
| GET | `/api/evaluate` | Epoch history + metrics + confusion matrix |
| GET | `/api/evaluate/export` | Download epoch history as CSV |
| GET | `/api/test?model=NAME` | Available models + selected model info |
| POST | `/api/test/infer` | Run inference (multipart) |
| POST | `/api/test/import-model` | Import a model JSON file (multipart) |
| GET | `/api/models` | List all trained models |
| GET | `/api/models/:name/download` | Download model JSON |

### SSE event format (GET /api/train/events)

```
event: epoch
data: {"epoch":1,"total_epochs":50,"train_loss":0.32,"val_loss":0.31,"train_accuracy":0.91,"val_accuracy":0.92,"elapsed_ms":843}

event: done
data: {"model_path":"trained_models/my_model.json","elapsed_total_ms":42000,"epochs_completed":50}

event: stopped
data: {"model_path":"...","elapsed_total_ms":8000,"epoch_reached":10,"total_epochs":50}

event: failed
data: {"reason":"..."}
```

---

## Frontend stack

| Layer | Choice | Notes |
|---|---|---|
| Build | Vite 6 | ESM-native, fast HMR |
| Framework | React 19 + TypeScript | |
| UI components | shadcn/ui (Radix + Tailwind CSS) | Run `npx shadcn@latest add <component>` to add |
| Charts | Recharts | Used for live loss/accuracy curves in Train tab |
| Data fetching | TanStack Query v5 | Cache + background refresh |
| Routing | React Router v7 | Tab navigation |
| Forms | React Hook Form | Architect layer builder |

### Dev setup

```bash
cd frontend
npm install
npm run dev       # starts at :5173, proxies /api → :7878
```

The Vite proxy in `vite.config.ts`:
```ts
server: {
  proxy: {
    '/api': 'http://127.0.0.1:7878',
  },
},
```

Ensure the Rust studio is running (`cargo run --bin studio --release` from the `ferrite-nn` repo) before starting the frontend dev server.

### Build for production

```bash
cd frontend && npm run build    # outputs to frontend/dist/
```

The Docker Compose setup serves `frontend/dist/` via Nginx.

---

## The five workflow tabs

The studio has a linear workflow. Each tab corresponds to a step:

| Tab | Page component | Key API calls | Unlocked when |
|---|---|---|---|
| 1. Architect | `ArchitectPage.tsx` | GET/POST `/api/architect` | Always |
| 2. Dataset | `DatasetPage.tsx` | GET/POST `/api/dataset/*` | Spec saved |
| 3. Train | `TrainPage.tsx` | POST start/stop + SSE | Dataset loaded |
| 4. Evaluate | `EvaluatePage.tsx` | GET `/api/evaluate` | Training done |
| 5. Test | `TestPage.tsx` | GET/POST `/api/test/*` | Always |

The `tab_unlock` bitmask from each GET response tells the frontend which tabs to enable.

### Test tab — input modes

The Test tab supports three input modes based on the model's `input_type` metadata:

- **`input_type = null` or missing** — show a three-way manual selector (Numeric / Grayscale Image / RGB Image). For image modes, user enters W×H dimensions.
- **`input_type = { "type": "ImageGrayscale", "width": W, "height": H }`** — show Upload Image | Draw toggle. Draw mode: 280×280 canvas, white-on-black (MNIST style).
- **`input_type = { "type": "ImageRgb", "width": W, "height": H }`** — show file upload only.
- **`input_type = { "type": "Numeric" }`** — show textarea only.

All inference submissions use `multipart/form-data` with a hidden `input_mode` field.

---

## Python API (FastAPI) — Phase 3

Not yet implemented. Will be added in Phase 3.

### Purpose

FastAPI sits between Nginx and the Rust service. It:
1. Validates JWT tokens (httpOnly cookies)
2. Injects `X-User-Id` header on all proxied requests to the Rust service
3. Manages users, model registry, and sharing in PostgreSQL
4. Handles OAuth (GitHub + Google via Authlib)

### Stack

```
fastapi, uvicorn[standard]
sqlalchemy[asyncio], asyncpg       ← PostgreSQL async ORM
python-jose[cryptography]          ← JWT
passlib[bcrypt]                    ← password hashing
authlib                            ← OAuth providers
httpx                              ← async proxy to Rust service
alembic                            ← DB migrations
```

### Auth endpoints (when implemented)

```
POST /auth/register
POST /auth/login       → sets httpOnly access + refresh token cookies
POST /auth/logout
POST /auth/refresh
GET  /auth/oauth/github   → redirect
GET  /auth/oauth/github/callback
GET  /auth/oauth/google   → redirect
GET  /auth/oauth/google/callback
GET  /auth/me
```

---

## Docker Compose (Phase 4)

Not yet implemented. Services:
- `nginx` — serves React build, proxies `/api` + `/auth` to FastAPI
- `fastapi` — Python auth + proxy service
- `rust-studio` — Rust training/inference service (`ghcr.io/yourname/ferrite-studio-rust`)
- `postgres` — User data, model registry, experiments

---

## Conventions

- **No Rust code in this repo** — all Rust lives in `ferrite-nn`
- **No ML logic in this repo** — training, inference, and model math is in the Rust service
- **API calls go through `src/api/`** — typed wrappers, never raw `fetch` in components
- **shadcn/ui components** — always prefer extending shadcn over custom CSS
- **Recharts** for charts — TanStack Query for data, never direct fetch in chart components
- **All `/api/*` requests** are proxied to Rust in dev, forwarded via FastAPI in production

---

## Related repositories

| Repo | What it contains |
|---|---|
| `ferrite-nn` | Rust neural network library + Rust studio API binary |
| `ferrite-studio` (this repo) | React SPA + Python FastAPI + Docker Compose |

The full long-term roadmap lives at `../ferrite-nn/ROADMAP.md`.
The full REST API reference lives at `../ferrite-nn/docs/api-reference.md`.
