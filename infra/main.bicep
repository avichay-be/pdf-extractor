@description('Environment name (e.g., dev, staging, prod)')
param environmentName string = 'prod'

@description('Location for all resources')
param location string = resourceGroup().location

@description('Base name for resources')
param baseName string = 'michman-pdf'

@description('Container image tag to deploy')
param imageTag string = 'latest'

// Secrets passed from deployment
@secure()
param azureApiKey string

@secure()
param azureOpenAiApiKey string

@secure()
param azureDocumentIntelligenceKey string

@secure()
param geminiApiKey string

@secure()
param apiKey string

// API URLs
param mistralApiUrl string

param azureOpenAiApiUrl string

param azureDocumentIntelligenceEndpoint string

// Resource naming
var resourceSuffix = '${baseName}-${environmentName}'
var containerRegistryName = replace('acr${baseName}${environmentName}', '-', '')
var containerAppEnvironmentName = 'cae-${resourceSuffix}'
var containerAppName = 'ca-${resourceSuffix}'

var tags = {
  environment: environmentName
  application: 'michman-pdf-extractor'
  managedBy: 'bicep'
}

// Container Registry
module containerRegistry 'modules/container-registry.bicep' = {
  name: 'containerRegistry'
  params: {
    name: containerRegistryName
    location: location
    tags: tags
    sku: 'Basic'
  }
}

// Reference to deployed ACR for listCredentials (avoids BCP181 error)
resource existingAcr 'Microsoft.ContainerRegistry/registries@2023-07-01' existing = {
  name: containerRegistryName
  dependsOn: [
    containerRegistry
  ]
}

// Container Apps Environment
module containerAppEnvironment 'modules/container-app-environment.bicep' = {
  name: 'containerAppEnvironment'
  params: {
    name: containerAppEnvironmentName
    location: location
    tags: tags
  }
}

// Container App
module containerApp 'modules/container-app.bicep' = {
  name: 'containerApp'
  params: {
    name: containerAppName
    location: location
    tags: tags
    environmentId: containerAppEnvironment.outputs.environmentId
    containerImage: '${containerRegistry.outputs.loginServer}/${baseName}:${imageTag}'
    containerRegistryLoginServer: containerRegistry.outputs.loginServer
    containerRegistryUsername: containerRegistryName
    containerRegistryPassword: existingAcr.listCredentials().passwords[0].value
    azureApiKey: azureApiKey
    azureOpenAiApiKey: azureOpenAiApiKey
    azureDocumentIntelligenceKey: azureDocumentIntelligenceKey
    geminiApiKey: geminiApiKey
    apiKey: apiKey
    mistralApiUrl: mistralApiUrl
    azureOpenAiApiUrl: azureOpenAiApiUrl
    azureDocumentIntelligenceEndpoint: azureDocumentIntelligenceEndpoint
  }
}

// Outputs
@description('Container Registry login server')
output containerRegistryLoginServer string = containerRegistry.outputs.loginServer

@description('Container Registry name')
output containerRegistryName string = containerRegistry.outputs.name

@description('Container App URL')
output containerAppUrl string = containerApp.outputs.url

@description('Container App FQDN')
output containerAppFqdn string = containerApp.outputs.fqdn

@description('Container Apps Environment name')
output containerAppEnvironmentName string = containerAppEnvironment.outputs.environmentName
