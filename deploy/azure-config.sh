#!/bin/bash
# Azure Container Apps Deployment Helper Script
#
# This script helps with local development and debugging of the Azure deployment.
# It sets up environment variables and creates necessary resources.
#
# Usage:
#   ./deploy/azure-config.sh [environment]
#
# Examples:
#   ./deploy/azure-config.sh          # Uses 'prod' environment
#   ./deploy/azure-config.sh dev      # Uses 'dev' environment
#   ./deploy/azure-config.sh staging  # Uses 'staging' environment

set -e

# Configuration
ENVIRONMENT="${1:-prod}"
LOCATION="${AZURE_LOCATION:-eastus}"
BASE_NAME="michman-pdf"

# Resource names
RESOURCE_GROUP="rg-${BASE_NAME}-${ENVIRONMENT}"
ACR_NAME="acr${BASE_NAME//\-/}${ENVIRONMENT}"
CONTAINER_APP_ENV="cae-${BASE_NAME}-${ENVIRONMENT}"
CONTAINER_APP="ca-${BASE_NAME}-${ENVIRONMENT}"

echo "========================================"
echo "Azure Container Apps Deployment Helper"
echo "========================================"
echo ""
echo "Environment: ${ENVIRONMENT}"
echo "Location: ${LOCATION}"
echo "Resource Group: ${RESOURCE_GROUP}"
echo "Container Registry: ${ACR_NAME}"
echo "Container App: ${CONTAINER_APP}"
echo ""

# Check if Azure CLI is installed
if ! command -v az &> /dev/null; then
    echo "Error: Azure CLI is not installed."
    echo "Install it from: https://docs.microsoft.com/en-us/cli/azure/install-azure-cli"
    exit 1
fi

# Check if logged in
if ! az account show &> /dev/null; then
    echo "Not logged in to Azure. Running 'az login'..."
    az login
fi

# Show current subscription
echo "Current Azure subscription:"
az account show --query "{Name:name, ID:id}" -o table
echo ""

# Function to create resource group
create_resource_group() {
    echo "Creating resource group ${RESOURCE_GROUP}..."
    az group create \
        --name "${RESOURCE_GROUP}" \
        --location "${LOCATION}" \
        --tags environment="${ENVIRONMENT}" application="${BASE_NAME}" \
        --output none
    echo "Resource group created."
}

# Function to deploy infrastructure
deploy_infrastructure() {
    echo ""
    echo "Deploying infrastructure with Bicep..."
    echo "This will create: ACR, Container Apps Environment, and Container App"
    echo ""

    # Check if secrets are set
    if [ -z "$AZURE_API_KEY" ]; then
        echo "Warning: AZURE_API_KEY not set. Please set environment variables:"
        echo "  export AZURE_API_KEY=your_key"
        echo "  export AZURE_OPENAI_API_KEY=your_key"
        echo "  export AZURE_DOCUMENT_INTELLIGENCE_KEY=your_key"
        echo "  export GEMINI_API_KEY=your_key"
        echo "  export API_KEY=your_key"
        echo "  export MISTRAL_API_URL=your_url"
        echo "  export AZURE_OPENAI_API_URL=your_url"
        echo "  export AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT=your_endpoint"
        exit 1
    fi

    az deployment group create \
        --resource-group "${RESOURCE_GROUP}" \
        --template-file infra/main.bicep \
        --parameters environmentName="${ENVIRONMENT}" \
        --parameters azureApiKey="${AZURE_API_KEY}" \
        --parameters azureOpenAiApiKey="${AZURE_OPENAI_API_KEY}" \
        --parameters azureDocumentIntelligenceKey="${AZURE_DOCUMENT_INTELLIGENCE_KEY}" \
        --parameters geminiApiKey="${GEMINI_API_KEY}" \
        --parameters apiKey="${API_KEY}" \
        --parameters mistralApiUrl="${MISTRAL_API_URL}" \
        --parameters azureOpenAiApiUrl="${AZURE_OPENAI_API_URL}" \
        --parameters azureDocumentIntelligenceEndpoint="${AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT}" \
        --output table
}

# Function to build and push image locally
build_and_push() {
    echo ""
    echo "Building and pushing Docker image..."

    # Get ACR login server
    ACR_LOGIN_SERVER=$(az acr show --name "${ACR_NAME}" --query loginServer -o tsv)

    # Login to ACR
    az acr login --name "${ACR_NAME}"

    # Build and push
    IMAGE_TAG="${ACR_LOGIN_SERVER}/${BASE_NAME}:latest"
    docker build -t "${IMAGE_TAG}" -f Dockerfile.prod .
    docker push "${IMAGE_TAG}"

    echo "Image pushed: ${IMAGE_TAG}"
}

# Function to update container app
update_container_app() {
    echo ""
    echo "Updating Container App..."

    ACR_LOGIN_SERVER=$(az acr show --name "${ACR_NAME}" --query loginServer -o tsv)

    az containerapp update \
        --name "${CONTAINER_APP}" \
        --resource-group "${RESOURCE_GROUP}" \
        --image "${ACR_LOGIN_SERVER}/${BASE_NAME}:latest"

    echo "Container App updated."
}

# Function to show app URL
show_app_url() {
    echo ""
    FQDN=$(az containerapp show \
        --name "${CONTAINER_APP}" \
        --resource-group "${RESOURCE_GROUP}" \
        --query properties.configuration.ingress.fqdn -o tsv 2>/dev/null || echo "")

    if [ -n "$FQDN" ]; then
        echo "========================================"
        echo "Application URL: https://${FQDN}"
        echo "Health Check: https://${FQDN}/health"
        echo "========================================"
    fi
}

# Function to show logs
show_logs() {
    echo ""
    echo "Streaming logs from Container App..."
    az containerapp logs show \
        --name "${CONTAINER_APP}" \
        --resource-group "${RESOURCE_GROUP}" \
        --follow
}

# Function to get ACR credentials
get_acr_credentials() {
    echo ""
    echo "ACR Credentials for GitHub Actions:"
    echo "========================================"
    ACR_PASSWORD=$(az acr credential show --name "${ACR_NAME}" --query "passwords[0].value" -o tsv)
    echo "ACR_NAME: ${ACR_NAME}"
    echo "ACR_PASSWORD: ${ACR_PASSWORD}"
    echo ""
    echo "Add this as a GitHub secret named 'ACR_PASSWORD'"
}

# Main menu
echo "Select an action:"
echo "  1) Create resource group"
echo "  2) Deploy infrastructure (Bicep)"
echo "  3) Build and push Docker image"
echo "  4) Update Container App"
echo "  5) Show application URL"
echo "  6) Stream logs"
echo "  7) Get ACR credentials"
echo "  8) Full deployment (1 + 2 + 3)"
echo "  q) Quit"
echo ""
read -p "Enter choice: " choice

case $choice in
    1) create_resource_group ;;
    2) deploy_infrastructure ;;
    3) build_and_push ;;
    4) update_container_app ;;
    5) show_app_url ;;
    6) show_logs ;;
    7) get_acr_credentials ;;
    8)
        create_resource_group
        deploy_infrastructure
        build_and_push
        show_app_url
        ;;
    q|Q) echo "Exiting." ;;
    *) echo "Invalid choice." ;;
esac
