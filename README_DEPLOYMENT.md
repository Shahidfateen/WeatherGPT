# WeatherGPT - Production Deployment Guide
**Target Platforms: Google Cloud Run & Render (with Docker & Native Python support)**

---

## 1. Quick Architecture & Configuration Summary

| Feature | Production Specification |
| :--- | :--- |
| **Framework** | FastAPI (Python 3.11 / 3.14) with Uvicorn ASGI Server |
| **Port Binding** | Dynamic via `${PORT:-8080}` (compatible with Cloud Run `8080` & Render `10000`) |
| **Health Check Endpoint** | `GET /health` or `GET /api/health` (HTTP 200 OK) |
| **Docker Base Image** | `python:3.11-slim` with layer-cached dependencies |
| **Render Blueprint** | `render.yaml` (Free Tier Web Service ready) |
| **Google Cloud Run** | `cloudbuild.yaml` + `deploy_cloudrun.bat` / `deploy_cloudrun.sh` |

---

## 2. Option A: Deploy to Render (Easiest - Free Public HTTPS Link)

Render provides a completely free tier with an instant public `https://<your-app-name>.onrender.com` link and automatic SSL certificate.

### Step-by-Step Instructions:

1. **Push your code to GitHub**:
   Double-click `deploy_render.bat` or run:
   ```bash
   git init
   git add .
   git commit -m "Deploy WeatherGPT"
   git remote add origin https://github.com/YOUR_USERNAME/WeatherGPT.git
   git branch -M main
   git push -u origin main
   ```

2. **Connect to Render**:
   - Go to [dashboard.render.com](https://dashboard.render.com) and log in.
   - Click **New +** > **Web Service**.
   - Select your **WeatherGPT** GitHub repository.

3. **Configure Settings (Automatically detected by `render.yaml`)**:
   - **Name**: `weathergpt-ai` (or your preferred name)
   - **Root Directory**: `weathergpt-backend` (or leave empty if using root)
   - **Runtime**: `Python` (or `Docker`)
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
   - **Plan**: Free

4. **Environment Variables (Optional)**:
   - `GEMINI_API_KEY`: *(Optional)* Your Google Gemini API Key. (If omitted, users can enter their key directly in the interactive UI modal).
   - `OPEN_METEO_API_KEY`: *(Optional)* Open-Meteo key (default uses Open-Meteo free tier without key).

5. **Live Public URL**:
   Once Render completes the build (~1-2 minutes), your live presentation link will be:
   ```
   https://weathergpt-ai.onrender.com
   ```

---

## 3. Option B: Deploy to Google Cloud Run (Fully Managed Serverless Container)

Google Cloud Run packages the container via `Dockerfile` and gives you an autoscaling serverless deployment with a public HTTPS link:
```
https://weathergpt-<project-hash>-uc.a.run.app
```

### Method 1: Using Google Cloud Shell (Zero Local Installation)
1. Open the [Google Cloud Shell](https://shell.cloud.google.com).
2. Clone or upload the WeatherGPT folder:
   ```bash
   git clone https://github.com/YOUR_USERNAME/WeatherGPT.git
   cd WeatherGPT
   ```
3. Run the automated deployment script:
   ```bash
   chmod +x deploy_cloudrun.sh
   ./deploy_cloudrun.sh
   ```
4. Cloud Run will build the Docker container via Cloud Build and output your live HTTPS link.

### Method 2: From Local Windows Terminal (with gcloud CLI installed)
1. Install [Google Cloud SDK](https://cloud.google.com/sdk/docs/install) if not already installed.
2. Authenticate and select your project:
   ```cmd
   gcloud auth login
   gcloud config set project YOUR_PROJECT_ID
   ```
3. Run `deploy_cloudrun.bat` or run:
   ```cmd
   gcloud run deploy weathergpt \
     --source . \
     --platform managed \
     --region us-central1 \
     --allow-unauthenticated \
     --port 8080 \
     --memory 512Mi \
     --cpu 1
   ```
4. When asked **"Allow unauthenticated invocations to [weathergpt]?"**, enter `y`.
5. Cloud Run will display:
   ```
   Service URL: https://weathergpt-xxxxxx-uc.a.run.app
   ```

---

## 4. Verification and Health Checks

Once deployed to either platform, verify the deployment:

- **Web UI**: Visit the root URL `https://<your-service-url>/` in your browser.
- **Health Check**: Visit `https://<your-service-url>/health` to see the JSON health report:
  ```json
  {
    "status": "online",
    "service": "WeatherGPT Backend",
    "version": "1.0.0",
    "supported_languages_count": 13
  }
  ```
- **API Documentation**: Visit `https://<your-service-url>/docs` to view the interactive OpenAPI Swagger UI.
