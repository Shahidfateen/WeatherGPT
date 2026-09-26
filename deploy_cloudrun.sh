#!/bin/bash
# ================================================================
# Deploy WeatherGPT directly to Google Cloud Run
# Works in Google Cloud Shell or any terminal with gcloud installed
# ================================================================

set -e

echo "=== Deploying WeatherGPT to Google Cloud Run ==="

# Check gcloud CLI
if ! command -v gcloud &> /dev/null; then
    echo "gcloud CLI could not be found. Please install Google Cloud SDK or run inside Google Cloud Shell."
    exit 1
fi

PROJECT_ID=$(gcloud config get-value project)
echo "Current Google Cloud Project: $PROJECT_ID"

echo "Deploying container to Cloud Run..."
gcloud run deploy weathergpt \
    --source . \
    --platform managed \
    --region us-central1 \
    --allow-unauthenticated \
    --port 8080 \
    --memory 512Mi \
    --cpu 1 \
    --min-instances 0 \
    --max-instances 5

echo "=== Deployment Succeeded! Live HTTPS link displayed above ==="
