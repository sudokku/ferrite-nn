# ferrite-nn — Long-Term Development Roadmap

> Last updated: 2026-03-01
> Status: Active planning — no features committed yet beyond what is currently implemented.

---

## Table of Contents

1. [Where We Are Today](#1-where-we-are-today)
2. [Repository Strategy](#2-repository-strategy)
3. [Phase 0 — Git Hygiene & Cargo Workspace](#phase-0--git-hygiene--cargo-workspace-restructure)
4. [Phase 1 — Rust Backend: tiny_http → Axum REST API](#phase-1--rust-backend-tinyhttp--axum-rest-api)
5. [Phase 2 — React SPA Frontend](#phase-2--react-spa-frontend)
6. [Phase 3 — ferrite-studio: FastAPI Auth + User Accounts](#phase-3--ferrite-studio-fastapi-auth--user-accounts)
7. [Phase 4 — Docker Compose Deployment](#phase-4--docker-compose-deployment)
8. [Future Considerations](#future-considerations)
9. [Key Decisions Log](#key-decisions-log)

---

## 1. Where We Are Today

### Codebase inventory

| Component | Location | LOC | Tech |
|---|---|---|---|
| Neural network library | `src/` | ~800 | Rust |
| Studio server + handlers | `studio/` | ~3,472 | Rust (tiny_http) |
| Studio HTML/CSS/JS monolith | `studio/assets/studio.html` | ~996 | Vanilla HTML/JS |
| Examples | `examples/` | ~430 | Rust |
| **Total** | | **~5,700** | |

### Current architecture

```
Browser ← HTML/CSS/JS (full-page server-side renders) → tiny_http Rust server
                                                              │
                                                        Arc<Mutex<StudioState>>
                                                        (all in-memory, no DB)
```

- **No authentication** — localhost only, single shared state, no user isolation
- **No persistence** beyond model `.json` files written to `trained_models/`
- **SSE** for real-time training progress (works well, should be preserved)
- **5 tabs**: Architect → Dataset → Train → Evaluate → Test

### What the library currently exports (public API)

`Network`, `NetworkSpec`, `LayerSpec`, `ModelMetadata`, `InputType`, `Layer`, `Matrix`,
`ActivationFunction` (9 variants), `LossType` (5 variants), `MseLoss`, `CrossEntropyLoss`,
`BceLoss`, `MaeLoss`, `HuberLoss`, `Sgd`, `EpochStats`, `TrainConfig`, `train_loop`,
`train_network`

### Immediate git issues (must fix before any restructure)

- [ ] `studio/util/idx.rs` is **untracked** (181 lines — will not exist in a fresh clone)
- [ ] `trained_models/*.json` not gitignored — model artifacts accumulate as untracked files
- [ ] `.DS_Store` not gitignored (appears at root and `examples/`)
- [ ] No git remote configured — repo has never been pushed anywhere

---

## 2. Repository Strategy

### The question: one repo or many?

**Short answer:** Keep everything in one Rust Cargo workspace for now. When the React + FastAPI platform is built, put it in a **separate repo** (`ferrite-studio`). Never use git submodules.

### Why not git submodules

The developer community is near-unanimous on this. Specific Cargo/Rust issues:

- [`rust-lang/cargo#6041`](https://github.com/rust-lang/cargo/issues/6041) — workspace support for git submodules is explicitly broken
- [`rust-lang/cargo#10278`](https://github.com/rust-lang/cargo/issues/10278) — Cargo fetches submodule contents on every `cargo update`
- Detached HEAD state on clone; every contributor must know `--recurse-submodules`; CI requires extra setup steps

### Why not split the library and studio repos now

The library API is at `v0.1.0` (semver: "anything can change"). Active development across `src/` and `studio/` is tightly coupled — changing a type in the library immediately breaks the studio, and the compiler tells you immediately. Splitting now would create a "two-commit, two-repo" workflow for what should be a one-line atomic change.

**Split signals to watch for** (split when ≥2 are true):
1. Library API stable for one full minor version cycle without breaking changes
2. External consumers depending on `ferrite-nn` (other projects, other people)
3. Studio release cycle meaningfully different from library's
4. You want to keep the studio private while keeping the library open source

### Industry precedent

| Project | Library repo | Platform repo | Relationship |
|---|---|---|---|
| PyTorch | `pytorch/pytorch` | `pytorch/serve` (TorchServe) | Separate repos, PyPI dep |
| TensorFlow | `tensorflow/tensorflow` | `tensorflow/tfx`, `tensorflow/serving` | Separate repos, PyPI dep |
| Hugging Face | `huggingface/transformers` | `huggingface/huggingface_hub` | Separate repos, PyPI dep |
| Grafana | `grafana/grafana` (Go+React) | `grafana/grafana-plugin-sdk-go` | SDK in its own repo |
| MLflow | `mlflow/mlflow` (monorepo: Python + React) | — | Monorepo (UI + server inseparable) |

The pattern: **the core library ships separately from the platform once the API is stable.** Until then, a monorepo is the pragmatic choice.

### Target repository layout (end state)

```
Repo 1: ferrite-nn  (Rust, open source)
  ├── ferrite-nn library  → published to crates.io when API stabilizes
  └── Rust studio binary  → compiled into a Docker image (the training microservice)

Repo 2: ferrite-studio  (Python + TypeScript, open or private)
  ├── React SPA frontend
  ├── Python FastAPI gateway  (auth, user data, proxy to Rust service)
  └── Docker Compose          (nginx + fastapi + rust-service + postgres)
```

The two repos never share source code at the VCS level. The Rust training service is a Docker container that FastAPI talks to over HTTP. **No submodules, no path hacks.**

---

## Phase 0 — Git Hygiene & Cargo Workspace Restructure

> **When:** Before any other work. Takes ~1 hour.

### 0.1 Fix .gitignore

```gitignore
/target
/trained_models/
.DS_Store
**/.DS_Store
examples/mnist_data/
```

### 0.2 Commit untracked files

```bash
git add studio/util/idx.rs
git commit -m "Add IDX binary parser utility"
```

### 0.3 Restructure to a proper Cargo workspace

**Current (one package, conflated):**
```toml
# Cargo.toml
[package]
name = "ferrite-nn"
version = "0.1.0"

[lib]  path = "src/lib.rs"
[[bin]]  name = "studio"  path = "studio/main.rs"
[[example]]  name = "mnist"  path = "examples/mnist.rs"
```

**Target (workspace with separate crates):**
```
ferrite-nn/                         ← git repo root
  Cargo.toml                        ← workspace manifest
  Cargo.lock                        ← single lockfile for the whole workspace
  README.md
  ROADMAP.md
  .gitignore
  │
  crates/
    ferrite-nn/                     ← the library crate
      Cargo.toml                    ←   name = "ferrite-nn", version = "0.1.0"
      src/                          ←   deps: rand, serde, serde_json only
      README.md
    │
    ferrite-studio/                 ← the studio binary crate
      Cargo.toml                    ←   name = "ferrite-studio", version = "0.1.0"
                                    ←   [dep] ferrite-nn = { path = "../../crates/ferrite-nn" }
                                    ←   [dep] axum, tokio, serde, serde_json, image
      src/                          ←   (currently studio/*.rs)
      assets/
        studio.html                 ←   (currently studio/assets/studio.html)
  │
  examples/
    xor.rs
    mnist.rs
    mnist_data/                     ←  gitignored
```

**Root `Cargo.toml`:**
```toml
[workspace]
members = [
    "crates/ferrite-nn",
    "crates/ferrite-studio",
]
resolver = "2"
```

**Benefits:**
- `cargo publish -p ferrite-nn` works without studio being involved
- `cargo build -p ferrite-studio` builds only the studio
- `cargo test --workspace` runs all tests
- The library's `Cargo.toml` drops to 3 deps (rand, serde, serde_json) — clean
- `image` and future web deps live only in the studio crate

### 0.4 Two small library API additions (improve the public surface before the split)

These replace fragile internal field accesses currently used in the studio:

```rust
// src/network/network.rs — add two methods:

/// Returns the number of input neurons the network expects.
pub fn input_size(&self) -> usize {
    self.layers.first().map(|l| l.weights.cols).unwrap_or(0)
}

/// Returns a reference to the output layer's activation function.
pub fn output_activation(&self) -> Option<&ActivationFunction> {
    self.layers.last().map(|l| &l.activator)
}
```

These clean up `studio/handlers/test.rs` which currently reads
`network.layers[0].weights.cols` and `network.layers.last().unwrap().activator` directly.

---

## Phase 1 — Rust Backend: tiny_http → Axum REST API

> **When:** After Phase 0. This is the foundation for the React SPA.
> **Goal:** Keep the Rust binary as the training/inference microservice; convert it from SSR HTML to a JSON REST API.

### Why axum

- Tokio-native (async) — handles SSE streaming, concurrent requests far better than tiny_http's thread-per-request model
- Axum 0.7 requires rustc ≥ 1.75 (already met)
- Built-in SSE support via `axum::response::Sse` — no hacks needed
- Excellent middleware ecosystem (tower)
- Battle-tested: used in production at major companies

### Cargo.toml additions for ferrite-studio

```toml
[dependencies]
ferrite-nn = { path = "../../crates/ferrite-nn" }
axum        = "0.7"
tokio       = { version = "1", features = ["full"] }
tower       = "0.4"
tower-http  = { version = "0.5", features = ["cors", "fs"] }
serde       = { version = "1", features = ["derive"] }
serde_json  = "1"
image       = { version = "0.24", default-features = false, features = ["png","jpeg","bmp","gif"] }
```

### New REST API surface

All existing routes become JSON endpoints. The SSE stream format is preserved (only the HTTP plumbing changes).

```
GET  /api/models                     → [{name, input_type, output_labels, ...}]
GET  /api/architect                  → {spec, hyperparams}
POST /api/architect/save             → {ok: true}
POST /api/dataset/upload             → {rows, features, labels, preview_rows}
POST /api/dataset/upload-idx         → {rows, features, labels}
POST /api/dataset/builtin            → {name, rows, features, labels}
GET  /api/train                      → {status, spec, hyperparams, dataset_summary}
POST /api/train/start                → {ok: true} or {error: "..."}
POST /api/train/stop                 → {ok: true}
GET  /api/train/events               → SSE stream (format unchanged)
GET  /api/evaluate                   → {epoch_history, metrics, confusion_matrix}
GET  /api/evaluate/export            → file download (CSV)
GET  /api/test                       → {models: [...], selected: "..."}
POST /api/test/infer                 → {result_type, prediction, confidence, all_scores}
POST /api/test/import-model          → {name} (redirect preserved as 303)
GET  /api/models/:name/download      → file download (JSON)
```

### State migration

Replace `Arc<Mutex<StudioState>>` with axum's `State` extractor — same semantics, cleaner API:

```rust
#[tokio::main]
async fn main() {
    let state = Arc::new(Mutex::new(StudioState::new()));
    let app = Router::new()
        .route("/api/models", get(handlers::models::list))
        .route("/api/train/events", get(handlers::train_sse::stream))
        // ...
        .with_state(state);
    let listener = tokio::net::TcpListener::bind("0.0.0.0:7878").await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
```

### Serving the React SPA

During development: Vite dev server proxies `/api/*` to `:7878`. In production: axum serves the built React assets from a `dist/` directory via `tower_http::services::ServeDir`.

```rust
// serve React build from dist/ in production
.fallback_service(ServeDir::new("dist").append_index_html_on_directories(true))
```

### What is preserved

- SSE event format (`event: epoch\ndata: {...}`) — unchanged, only the HTTP layer changes
- Model JSON format — unchanged
- `trained_models/` directory convention — unchanged (for now; made configurable via env var)
- All 5 workflow steps — unchanged in concept, implemented as API calls

### Delete

- `studio/render.rs` — no longer needed; React handles rendering
- `studio/assets/studio.html` — replaced by React build output

---

## Phase 2 — React SPA Frontend

> **When:** Alongside or immediately after Phase 1.
> **Goal:** Replace the 996-line HTML monolith with a proper component tree.

### Stack

| Layer | Choice | Rationale |
|---|---|---|
| Build | Vite 6 + TypeScript | Fast HMR, ESM-native, standard for React 2025 |
| Framework | React 19 | Largest ecosystem for ML dashboards; best charting libs |
| UI components | shadcn/ui (Radix + Tailwind) | Accessible, unstyled base; clean aesthetic that matches current studio |
| Charts | Recharts | React-native, SSE-friendly, composable — best for live loss curves |
| Data fetching | TanStack Query v5 | Caching, background refresh, optimistic updates |
| Routing | React Router v7 | Tab navigation, URL persistence |
| Forms | React Hook Form | Architect tab has complex nested layer forms |

### Directory structure

```
crates/ferrite-studio/frontend/         ← lives inside the studio crate
  package.json
  vite.config.ts                        ← proxy: /api → localhost:7878 (dev)
  tsconfig.json
  tailwind.config.ts
  src/
    main.tsx
    App.tsx                             ← tab router
    api/
      client.ts                         ← base fetch wrapper
      architect.ts                      ← useArchitect(), useSaveSpec()
      dataset.ts                        ← useDataset(), useUploadCSV(), useUploadIDX()
      train.ts                          ← useTrainStart(), useTrainStop()
      evaluate.ts                       ← useEvaluate()
      models.ts                         ← useModels(), useModelDownload()
      test.ts                           ← useInfer()
    hooks/
      useSSE.ts                         ← EventSource wrapper with reconnect
    pages/
      ArchitectPage.tsx
      DatasetPage.tsx
      TrainPage.tsx
      EvaluatePage.tsx
      TestPage.tsx
    components/
      architect/
        LayerRow.tsx
        LayerBuilder.tsx                ← add/remove/reorder layers
        HyperparamForm.tsx
        ArchitectureSummary.tsx
      dataset/
        FileUpload.tsx
        IDXUpload.tsx
        BuiltinDatasetPicker.tsx
        DataPreviewTable.tsx
      train/
        TrainControls.tsx               ← Start / Stop buttons
        LiveLossChart.tsx               ← Recharts + useSSE hook
        EpochProgressBar.tsx
      evaluate/
        MetricsTable.tsx
        LossCurveChart.tsx
        ConfusionMatrix.tsx
        ExportButton.tsx
      test/
        ModelSelector.tsx
        InputModeToggle.tsx             ← Numeric / Grayscale / RGB
        CanvasDraw.tsx                  ← 280×280 canvas, white-on-black
        ImageUpload.tsx
        ResultCard.tsx                  ← softmax bars, sigmoid output, raw values
      ui/                              ← shadcn/ui generated components
        button.tsx, card.tsx, badge.tsx, tabs.tsx, toggle.tsx, ...
```

### Real-time training chart

```typescript
// hooks/useSSE.ts
export function useTrainingSSE(onEpoch: (stats: EpochStats) => void) {
  useEffect(() => {
    const es = new EventSource('/api/train/events');
    es.addEventListener('epoch', (e) => onEpoch(JSON.parse(e.data)));
    es.addEventListener('done',  () => es.close());
    es.addEventListener('error', () => es.close());
    return () => es.close();
  }, []);
}
```

The `LiveLossChart` component subscribes to this hook and appends data points to Recharts `<LineChart>` state in real time.

### Build output wired into axum

```bash
cd frontend && npm run build        # outputs to dist/
cargo run -p ferrite-studio         # axum serves dist/ + /api/* routes
```

---

## Phase 3 — ferrite-studio: FastAPI Auth + User Accounts

> **When:** After Phase 1 + 2 are stable and the team is ready to host for multiple users.
> **Lives in:** A **new, separate git repository** — `ferrite-studio`.

### Why a separate repo at this point

By Phase 3, ferrite-nn is a Docker-deployable Rust binary. FastAPI communicates with it over HTTP — there is **zero source-level coupling** between the Python code and the Rust code. No git submodules, no path hacks. The two repos are as independent as a frontend and a database.

```
ferrite-studio/               ← NEW git repo
  api/                        ← Python FastAPI service
    main.py
    auth/
      jwt.py                  ← token generation + validation
      oauth.py                ← GitHub + Google OAuth flows
    models/
      user.py                 ← SQLAlchemy models
      model_registry.py
      experiment.py
    routes/
      auth.py
      proxy.py                ← proxy validated requests → Rust service
      models.py               ← model registry CRUD
      sharing.py              ← share tokens, permissions
    requirements.txt          ← fastapi, uvicorn, sqlalchemy, asyncpg, authlib, python-jose, bcrypt
  frontend/                   ← React SPA (moved from ferrite-nn repo's studio crate)
    package.json
    src/
  docker/
    docker-compose.yml
    nginx.conf
    Dockerfile.api
    Dockerfile.frontend       ← nginx serving Vite build
  docs/
    api-reference.md
    deployment.md
  README.md
```

### FastAPI stack

| Concern | Choice |
|---|---|
| Web framework | FastAPI + Uvicorn |
| Database ORM | SQLAlchemy 2.0 async + asyncpg |
| JWT | python-jose (HS256, rotating refresh tokens) |
| Password hashing | passlib[bcrypt] |
| OAuth | Authlib (GitHub + Google providers) |
| Proxy to Rust | httpx (async HTTP client) |

### PostgreSQL schema

```sql
-- Users
CREATE TABLE users (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email         TEXT UNIQUE NOT NULL,
    display_name  TEXT,
    hashed_pw     TEXT,                        -- null for OAuth-only accounts
    oauth_provider TEXT,                       -- 'github' | 'google' | null
    oauth_id      TEXT,
    created_at    TIMESTAMPTZ DEFAULT now()
);

-- JWT refresh tokens (stored hashed, revocable)
CREATE TABLE refresh_tokens (
    id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id    UUID REFERENCES users(id) ON DELETE CASCADE,
    token_hash TEXT NOT NULL,
    expires_at TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ DEFAULT now()
);

-- Per-user model registry (the actual .json files live in the Rust service's file tree)
CREATE TABLE models (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    owner_id      UUID REFERENCES users(id) ON DELETE CASCADE,
    name          TEXT NOT NULL,
    file_path     TEXT NOT NULL,              -- relative path on the Rust service filesystem
    input_type    JSONB,
    output_labels JSONB,
    created_at    TIMESTAMPTZ DEFAULT now(),
    UNIQUE(owner_id, name)
);

-- Experiment history (epoch stats from training runs)
CREATE TABLE experiments (
    id             UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_id       UUID REFERENCES models(id) ON DELETE CASCADE,
    owner_id       UUID REFERENCES users(id) ON DELETE CASCADE,
    hyperparams    JSONB,
    epoch_history  JSONB,
    created_at     TIMESTAMPTZ DEFAULT now()
);

-- Model sharing
CREATE TABLE shares (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_id        UUID REFERENCES models(id) ON DELETE CASCADE,
    shared_by_id    UUID REFERENCES users(id) ON DELETE CASCADE,
    share_token     TEXT UNIQUE NOT NULL,      -- for public share links
    shared_with_id  UUID REFERENCES users(id), -- null = public link
    permission      TEXT DEFAULT 'read',       -- 'read' | 'write'
    expires_at      TIMESTAMPTZ,
    created_at      TIMESTAMPTZ DEFAULT now()
);
```

### Auth endpoints

```
POST /auth/register                email + password → user + tokens
POST /auth/login                   email + password → httpOnly cookie (access + refresh)
POST /auth/logout                  clear cookies + revoke refresh token
POST /auth/refresh                 rotate refresh token
GET  /auth/oauth/github            redirect → GitHub consent
GET  /auth/oauth/github/callback   code → user + tokens
GET  /auth/oauth/google            redirect → Google consent
GET  /auth/oauth/google/callback   code → user + tokens
GET  /auth/me                      current user profile
```

### Request flow

```
Browser
  → FastAPI (auth middleware validates JWT cookie)
        ├── /auth/*          → handled locally
        ├── /api/*           → inject X-User-Id header → proxy to Rust service
        └── /                → Nginx serves React build
```

The Rust service becomes user-aware without its own auth: it trusts the `X-User-Id` header from FastAPI (never exposed to the internet directly) and scopes file paths accordingly:

```
trained_models/
  {user_id}/
    mnist.json
    my_model.json
```

### User-facing features (Phase 3)

| Feature | Description |
|---|---|
| Login / Register | Email + password; OAuth via GitHub or Google |
| Per-user models | Each user sees only their own trained models |
| Per-user datasets | Uploaded CSVs/IDX files stored per user |
| Model sharing | Share a model via a link (read-only by default); optional expiry |
| Experiment history | Every training run saves hyperparams + epoch stats to Postgres |
| Public model gallery | (Optional) Make a model publicly browsable |

---

## Phase 4 — Docker Compose Deployment

> **When:** After Phase 3, when ready to host for the team.

### docker-compose.yml (overview)

```yaml
services:
  nginx:
    image: nginx:alpine
    ports: ["80:80", "443:443"]
    volumes:
      - ./docker/nginx.conf:/etc/nginx/nginx.conf
      - ./frontend/dist:/usr/share/nginx/html   # React build (static)
    depends_on: [fastapi]

  fastapi:
    build: ./docker/Dockerfile.api
    environment:
      DATABASE_URL: postgresql+asyncpg://postgres:postgres@postgres:5432/ferrite
      RUST_SERVICE_URL: http://rust-studio:7878
      JWT_SECRET: ${JWT_SECRET}
      GITHUB_CLIENT_ID: ${GITHUB_CLIENT_ID}
      GITHUB_CLIENT_SECRET: ${GITHUB_CLIENT_SECRET}
      GOOGLE_CLIENT_ID: ${GOOGLE_CLIENT_ID}
      GOOGLE_CLIENT_SECRET: ${GOOGLE_CLIENT_SECRET}
    depends_on: [postgres, rust-studio]

  rust-studio:
    image: ghcr.io/yourname/ferrite-studio-rust:latest  # built from ferrite-nn repo
    volumes:
      - model-data:/app/trained_models
    environment:
      TRUST_X_USER_ID: "true"

  postgres:
    image: postgres:17-alpine
    volumes:
      - postgres-data:/var/lib/postgresql/data
    environment:
      POSTGRES_DB: ferrite
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}

volumes:
  model-data:
  postgres-data:
```

### Nginx routing

```nginx
# Static React SPA
location / {
    try_files $uri $uri/ /index.html;
}

# Auth + user API → FastAPI
location /auth/ { proxy_pass http://fastapi:8000; }
location /api/  { proxy_pass http://fastapi:8000; }
# FastAPI proxies /api/* → Rust service internally
```

---

## Future Considerations

These are not planned but worth thinking about when they become relevant:

### Publish `ferrite-nn` to crates.io

When: external consumers exist or the API has been stable for 2+ versions.
Action: publish `crates/ferrite-nn` as a standalone crate. The studio then becomes a real downstream consumer and an example of the public API.

### PyO3 bindings (Python API for ferrite-nn)

The Python FastAPI service currently treats the Rust studio as a black box (HTTP calls). If ML researchers want to call ferrite-nn training directly from Python scripts or Jupyter notebooks, [maturin](https://www.maturin.rs/) + [pyo3](https://pyo3.rs/) would expose the library as a native Python extension module. This would be a new build target in the `crates/ferrite-nn/` crate.

### Horizontal scaling of training jobs

The current design runs training in a single background thread per studio instance. For a larger team with concurrent training runs:
- Extract training into a **job queue** (e.g., RQ + Redis, or Celery)
- FastAPI submits jobs; multiple Rust studio instances (workers) pick them up
- SSE events forwarded from worker → FastAPI → browser via Redis pub/sub

### S3-compatible model storage

Replace the local filesystem (`trained_models/`) with MinIO (self-hosted) or AWS S3 for model files and datasets. Minimal code change: replace `std::fs::read/write` with presigned URL generation.

### `ferrite-nn` GPU support

Currently the library runs on CPU only (pure Rust, no BLAS/CUDA). If GPU acceleration becomes desirable:
- CUDA bindings via `cudarc` crate (Rust-native)
- Or: expose a Python API (PyO3) and delegate GPU ops to PyTorch under the hood for complex models

---

## Key Decisions Log

| Decision | Choice | Rationale | Date |
|---|---|---|---|
| Repo structure (now) | Single Cargo workspace, split crates | API unstable; atomic commits needed; no submodule pain | 2026-03-01 |
| Repo structure (future) | Two separate repos (ferrite-nn, ferrite-studio) | No source coupling after HTTP boundary; different tech stacks | 2026-03-01 |
| Git submodules | Never | Documented Cargo bugs, universal developer friction, no benefit over alternatives | 2026-03-01 |
| Frontend framework | React 19 + Vite + TypeScript | Largest ML dashboard ecosystem; best charting libs; shadcn/ui aesthetic match | 2026-03-01 |
| API backend | Python FastAPI | User preference; OAuth library support; auto-OpenAPI docs; easy deploy | 2026-03-01 |
| Training microservice | Rust axum (from ferrite-nn repo) | Can't easily call Rust from Python without PyO3; keep ML logic in Rust | 2026-03-01 |
| Auth tokens | JWT in httpOnly cookies, rotating refresh | Safer than localStorage; standard for team/SaaS tools | 2026-03-01 |
| OAuth providers | GitHub + Google | Developer-friendly; most common for technical audiences | 2026-03-01 |
| Database | PostgreSQL | Right size for 5–20 users; async support in both Python and Rust | 2026-03-01 |
| File storage | Local filesystem → MinIO/S3 (future) | Start simple; migrate without code rewrites when needed | 2026-03-01 |
| Training progress | SSE (preserved) | Already working well; simpler than WebSocket for unidirectional stream | 2026-03-01 |
| Input type selection | Metadata-based (no layer-size inference) | Avoids false positives (e.g. 100-input bank model ≠ image) | 2026-03-01 |
