@description('Name of the Container App')
param name string

@description('Location for the Container App')
param location string = resourceGroup().location

@description('Tags to apply to the Container App')
param tags object = {}

@description('Resource ID of the Container Apps Environment')
param environmentId string

@description('Container image to deploy')
param containerImage string

@description('Container registry login server')
param containerRegistryLoginServer string

@description('Container registry username')
@secure()
param containerRegistryUsername string

@description('Container registry password')
@secure()
param containerRegistryPassword string

@description('CPU cores allocated to the container')
param cpuCore string = '1.0'

@description('Memory allocated to the container')
param memorySize string = '2Gi'

@description('Minimum number of replicas')
param minReplicas int = 1

@description('Maximum number of replicas')
param maxReplicas int = 5

@description('Target port for the container')
param targetPort int = 8000

// Secrets for the application
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

@description('Mistral API URL')
param mistralApiUrl string

@description('Azure OpenAI API URL')
param azureOpenAiApiUrl string

@description('Azure Document Intelligence endpoint')
param azureDocumentIntelligenceEndpoint string

resource containerApp 'Microsoft.App/containerApps@2023-05-01' = {
  name: name
  location: location
  tags: tags
  properties: {
    managedEnvironmentId: environmentId
    configuration: {
      activeRevisionsMode: 'Single'
      ingress: {
        external: true
        targetPort: targetPort
        transport: 'http'
        allowInsecure: false
        traffic: [
          {
            weight: 100
            latestRevision: true
          }
        ]
      }
      registries: [
        {
          server: containerRegistryLoginServer
          username: containerRegistryUsername
          passwordSecretRef: 'registry-password'
        }
      ]
      secrets: [
        {
          name: 'registry-password'
          value: containerRegistryPassword
        }
        {
          name: 'azure-api-key'
          value: azureApiKey
        }
        {
          name: 'azure-openai-api-key'
          value: azureOpenAiApiKey
        }
        {
          name: 'azure-document-intelligence-key'
          value: azureDocumentIntelligenceKey
        }
        {
          name: 'gemini-api-key'
          value: geminiApiKey
        }
        {
          name: 'api-key'
          value: apiKey
        }
      ]
    }
    template: {
      containers: [
        {
          name: name
          image: containerImage
          resources: {
            cpu: json(cpuCore)
            memory: memorySize
          }
          env: [
            {
              name: 'AZURE_API_KEY'
              secretRef: 'azure-api-key'
            }
            {
              name: 'AZURE_OPENAI_API_KEY'
              secretRef: 'azure-openai-api-key'
            }
            {
              name: 'AZURE_DOCUMENT_INTELLIGENCE_KEY'
              secretRef: 'azure-document-intelligence-key'
            }
            {
              name: 'GEMINI_API_KEY'
              secretRef: 'gemini-api-key'
            }
            {
              name: 'API_KEY'
              secretRef: 'api-key'
            }
            {
              name: 'MISTRAL_API_URL'
              value: mistralApiUrl
            }
            {
              name: 'AZURE_OPENAI_API_URL'
              value: azureOpenAiApiUrl
            }
            {
              name: 'AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT'
              value: azureDocumentIntelligenceEndpoint
            }
            {
              name: 'ENABLE_CROSS_VALIDATION'
              value: 'true'
            }
            {
              name: 'VALIDATION_PROVIDER'
              value: 'openai'
            }
          ]
          probes: [
            {
              type: 'Liveness'
              httpGet: {
                path: '/health'
                port: targetPort
                scheme: 'HTTP'
              }
              initialDelaySeconds: 30
              periodSeconds: 30
              failureThreshold: 3
              timeoutSeconds: 5
            }
            {
              type: 'Readiness'
              httpGet: {
                path: '/health'
                port: targetPort
                scheme: 'HTTP'
              }
              initialDelaySeconds: 10
              periodSeconds: 10
              failureThreshold: 3
              timeoutSeconds: 5
            }
          ]
        }
      ]
      scale: {
        minReplicas: minReplicas
        maxReplicas: maxReplicas
        rules: [
          {
            name: 'http-scaling'
            http: {
              metadata: {
                concurrentRequests: '10'
              }
            }
          }
        ]
      }
    }
  }
}

@description('The FQDN of the Container App')
output fqdn string = containerApp.properties.configuration.ingress.fqdn

@description('The URL of the Container App')
output url string = 'https://${containerApp.properties.configuration.ingress.fqdn}'

@description('The name of the Container App')
output name string = containerApp.name

@description('The resource ID of the Container App')
output resourceId string = containerApp.id
