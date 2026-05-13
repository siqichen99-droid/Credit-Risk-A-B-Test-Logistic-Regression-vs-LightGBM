# Phase 3 — Docker Setup Guide

## What Docker is and why it matters

Your FastAPI server currently runs only on your laptop because it depends on
your specific Python version, your venv, and your folder structure. Docker
solves this by packaging your entire environment — Python, all libraries, the
model files, and the API code — into a single self-contained unit called a
**container**.

The result: one command runs your API identically on any machine in the world —
your laptop, a colleague's computer, a cloud server, or a company's
infrastructure. This is exactly how production APIs are deployed at companies
like Fannie Mae, Stripe, and every major tech company.

**The interview talking point:** "I containerized the API using Docker so the
model can be deployed to any environment — local, staging, or cloud — with a
single command and zero environment configuration."

---

## Step 1 — Install Docker Desktop

1. Go to **https://www.docker.com/products/docker-desktop**
2. Click "Download for Windows"
3. Run the installer — it will ask you to restart your computer
4. After restart, open Docker Desktop from the Start menu
5. Wait for the Docker whale icon in the taskbar to stop animating
   (this means Docker is running — takes about 30 seconds)

Verify Docker is working — open a terminal and run:
```bash
docker --version
```
You should see something like `Docker version 26.x.x`

---

## Step 2 — Organize your project files

Your project folder needs these files in the root before building:

```
credit_risk_ab_test/
Credit-Risk-A-B-Test-Logistic-Regression-vs-LightGBM/
├── Dockerfile              ← new
├── docker-compose.yml      ← new
├── .dockerignore           ← new
├── requirements.txt        ← already exists, update with new version
├── main.py                 ← already exists
├── models/
│   ├── model_b_lightgbm.pkl
│   ├── model_a_logistic.pkl
│   └── scaler.pkl
├── results/
│   └── feature_cols.txt
└── mlruns/                 ← already exists from Phase 2
```

Copy `Dockerfile`, `docker-compose.yml`, and `.dockerignore` into the same
folder as your `main.py`.

---

## Step 3 — Build the Docker image

Open your terminal, navigate to your project folder, then run:

```bash
docker build -t credit-risk-api .
```

**What this command means:**
- `docker build` — create a new image
- `-t credit-risk-api` — tag (name) the image `credit-risk-api`
- `.` — use the Dockerfile in the current directory

**What happens during the build (you will see these steps):**
```
Step 1/8: FROM python:3.11-slim          ← downloads base Python image (~150MB)
Step 2/8: WORKDIR /app                   ← sets working directory
Step 3/8: ENV ...                        ← sets environment variables
Step 4/8: RUN apt-get install libgomp1   ← installs LightGBM dependency
Step 5/8: COPY requirements.txt .        ← copies requirements file
Step 6/8: RUN pip install ...            ← installs all packages (takes 2-3 min)
Step 7/8: COPY main.py models/ results/  ← copies your API and model files
Step 8/8: CMD uvicorn main:app ...       ← sets the startup command
```

The first build takes 3-5 minutes (downloading base image + installing packages).
Subsequent builds are much faster because Docker caches layers that haven't changed.

---

## Step 4 — Run with Docker Compose (recommended)

Docker Compose starts both the API and MLflow UI together with one command:

```bash
docker-compose up
```

You will see logs from both services streaming in the terminal:
```
credit_risk_api     | INFO: Uvicorn running on http://0.0.0.0:8000
credit_risk_mlflow  | INFO: Uvicorn running on http://0.0.0.0:5000
```

Open two browser tabs:
- **API:**    http://127.0.0.1:8000/docs
- **MLflow:** http://127.0.0.1:5000

To stop both services: press `Ctrl + C` in the terminal.
To run in the background (detached mode): `docker-compose up -d`
To stop detached containers: `docker-compose down`

---

## Step 5 — Alternatively: run just the API container

If you only want the API without MLflow:

```bash
docker run -p 8000:8000 credit-risk-api
```

**What this means:**
- `docker run` — start a container from an image
- `-p 8000:8000` — map port 8000 on your laptop to port 8000 in the container
- `credit-risk-api` — the image name we built in Step 3

Test it in a new terminal:
```bash
python test_api.py
```

The output should be identical to Phase 1 — same predictions, same decisions.

---

## Step 6 — Verify the container is running

```bash
docker ps
```

You should see:
```
CONTAINER ID   IMAGE              STATUS         PORTS
abc123def456   credit-risk-api    Up 2 minutes   0.0.0.0:8000->8000/tcp
```

Check the health status:
```bash
docker inspect credit_risk_api --format='{{.State.Health.Status}}'
```
Should return: `healthy`

View logs from a running container:
```bash
docker logs credit_risk_api
```

---

## Key Docker concepts for interviews

**What is an image?**
A read-only blueprint — like a recipe. It contains your code, Python, all
libraries, and configuration. The image is built once and can be run anywhere.
`credit-risk-api` is your image.

**What is a container?**
A running instance of an image — like a dish cooked from the recipe. You can
run multiple containers from the same image simultaneously. Each container is
isolated from your laptop's environment.

**What is a layer?**
Each line in the Dockerfile creates a layer. Docker caches layers — if
`requirements.txt` hasn't changed, Docker reuses the cached pip install layer
on the next build instead of reinstalling everything. This is why the order
in a Dockerfile matters: put things that change rarely (pip install) before
things that change often (your code).

**Why does this matter for production?**
In a production environment, your model never runs "on someone's laptop." It
runs in a container on a cloud server (AWS, Azure, GCP). Docker ensures the
container behaves identically in development and production because the
environment is fully specified and portable.

**What does `--host 0.0.0.0` mean in the CMD?**
By default, uvicorn only listens on `127.0.0.1` (localhost), which is
inaccessible from outside the container. `0.0.0.0` tells uvicorn to listen
on all network interfaces, making the API reachable from your laptop through
Docker's port mapping.

---

## Common errors and fixes

**Error: `docker: command not found`**
→ Docker Desktop is not running. Open it from the Start menu and wait for
the whale icon to stop animating.

**Error: `Cannot connect to the Docker daemon`**
→ Same as above — Docker Desktop needs to be running first.

**Error: `FileNotFoundError: models/model_b_lightgbm.pkl`**
→ The `models/` folder is not in the same directory as your Dockerfile.
Double-check your project structure matches Step 2.

**Error: `port is already allocated`**
→ Something is already using port 8000 (your Phase 1 uvicorn server).
Stop it first: find the terminal running uvicorn and press Ctrl+C.

**Build is very slow**
→ Normal on first build — Docker is downloading the Python base image and
installing packages. Subsequent builds use the cache and take under 30 seconds.

---

## Phase 3 complete checklist

- [ ] Docker Desktop installed and running (whale icon in taskbar)
- [ ] `Dockerfile`, `docker-compose.yml`, `.dockerignore` in project root
- [ ] `docker build -t credit-risk-api .` completes successfully
- [ ] `docker-compose up` starts both services
- [ ] http://127.0.0.1:8000/docs loads inside Docker
- [ ] http://127.0.0.1:5000 shows MLflow UI inside Docker
- [ ] `docker ps` shows container as healthy
- [ ] `python test_api.py` returns same results as Phase 1

**Next: Phase 4 — Deploy to a live public URL**
