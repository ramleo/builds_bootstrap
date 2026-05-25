# deploy/cloud-platforms.md — Step 14: Generic Cloud Deployment

Imported by @deploy/cloud.md.

---

## Step 14 — Deploy to Any Cloud Platform (via Cloud Deploy Agent)
Perform this step after the Dockerfile and GitHub repo exist (Steps 11–12).

### 14a — Platform Selection

| Platform | Best For | Free Tier | Config File |
|---|---|---|---|
| **Render** | Simplest deploy from GitHub | ✅ Yes | `render.yaml` |
| **Fly.io** | Global edge, fast cold starts | ✅ Yes | `fly.toml` |
| **Railway** | One-click GitHub deploy | ✅ Yes | `railway.toml` |
| **AWS ECS (Fargate)** | Production, auto-scaling | ❌ Paid | `task-definition.json` |
| **AWS App Runner** | Easiest managed AWS container | ✅ Free tier | `apprunner.yaml` |
| **GCP Cloud Run** | Serverless containers, pay-per-use | ✅ Free tier | `cloudrun.yaml` |
| **Azure Container Apps** | Serverless containers on Azure | ✅ Free tier | `containerapp.yaml` |

### 14b — Prerequisites Checklist
```
✅ app.py              — FastAPI app at project root
✅ Dockerfile          — multi-stage build at project root
✅ requirements.txt    — pinned library versions
✅ models/             — final_pipeline.pkl + label_encoder.pkl
✅ .dockerignore       — excludes data/, plots/, tests/, docs/
✅ GitHub repo         — code pushed and up to date
✅ Docker image built  — verified locally with smoke tests
```

### 14c — Platform-Specific Deploy Commands

#### 🚁 Fly.io
```bash
brew install flyctl && fly auth login
fly launch --name <project-name> --region lax --no-deploy
# Edit fly.toml: set internal_port = 8000
fly deploy
curl https://<project-name>.fly.dev/health
```
**fly.toml template:**
```toml
app = "<project-name>"
primary_region = "lax"
[env]
  PORT = "8000"
[http_service]
  internal_port = 8000
  force_https = true
  auto_stop_machines = true
  auto_start_machines = true
  min_machines_running = 0
```

#### 🚂 Railway
```bash
npm install -g @railway/cli && railway login
railway init && railway up
railway variables set PORT=8000
railway open
```
**railway.toml template:**
```toml
[build]
builder = "DOCKERFILE"
dockerfilePath = "Dockerfile"
[deploy]
startCommand = "uvicorn app:app --host 0.0.0.0 --port $PORT"
restartPolicyType = "ON_FAILURE"
restartPolicyMaxRetries = 3
```

#### 🟠 AWS App Runner
```bash
brew install awscli && aws configure
aws ecr create-repository --repository-name <project-name>
aws ecr get-login-password | docker login --username AWS \
  --password-stdin <account-id>.dkr.ecr.<region>.amazonaws.com
docker tag <image-name>:latest \
  <account-id>.dkr.ecr.<region>.amazonaws.com/<project-name>:latest
docker push <account-id>.dkr.ecr.<region>.amazonaws.com/<project-name>:latest
aws apprunner create-service --cli-input-json file://apprunner.yaml
```

#### 🔵 GCP Cloud Run
```bash
brew install google-cloud-sdk && gcloud auth login
gcloud config set project <project-id>
gcloud services enable run.googleapis.com containerregistry.googleapis.com
gcloud builds submit --tag gcr.io/<project-id>/<project-name>:latest
gcloud run deploy <project-name> \
  --image gcr.io/<project-id>/<project-name>:latest \
  --platform managed --region us-central1 \
  --allow-unauthenticated --port 8000 --set-env-vars PORT=8000
gcloud run services describe <project-name> --format "value(status.url)"
```

#### 🟦 Azure Container Apps
```bash
brew install azure-cli && az login
az group create --name <project-name>-rg --location eastus
az containerapp env create --name <project-name>-env \
  --resource-group <project-name>-rg --location eastus
az acr create --resource-group <project-name>-rg \
  --name <project-name>acr --sku Basic
az acr login --name <project-name>acr
docker tag <image-name>:latest <project-name>acr.azurecr.io/<project-name>:latest
docker push <project-name>acr.azurecr.io/<project-name>:latest
az containerapp create --name <project-name> \
  --resource-group <project-name>-rg \
  --environment <project-name>-env \
  --image <project-name>acr.azurecr.io/<project-name>:latest \
  --target-port 8000 --ingress external --env-vars PORT=8000
```

### 14d — Universal Smoke Tests
Replace `<LIVE_URL>` with your deployed service URL, and `<SAMPLE_PAYLOAD>` with a valid JSON object from your dataset.
```bash
curl https://<LIVE_URL>/health
curl -X POST https://<LIVE_URL>/predict \
  -H "Content-Type: application/json" \
  -d '<SAMPLE_PAYLOAD>'
curl -X POST https://<LIVE_URL>/predict/batch \
  -H "Content-Type: application/json" \
  -d '[<SAMPLE_PAYLOAD>, <SAMPLE_PAYLOAD_2>]'
open https://<LIVE_URL>/docs
```

### 14e — Create `docs/cloud_deployment_guide.md`
Delegate to the Documentation Agent to write `docs/cloud_deployment_guide.md` with:
- Platform comparison table (Step 14a)
- Prerequisites checklist
- Step-by-step instructions for the chosen platform
- Universal smoke-test commands with the live URL filled in
- Cost / free-tier notes for the chosen platform
- Teardown / cleanup commands to avoid unexpected charges

### 14f — Push to GitHub
```bash
git add docs/cloud_deployment_guide.md
git add .
git commit -m "Add cloud deployment config and guide for <platform>"
git push origin main
```