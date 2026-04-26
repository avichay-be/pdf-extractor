@description('Name of the Container Apps Environment')
param name string

@description('Location for the environment')
param location string = resourceGroup().location

@description('Tags to apply to the environment')
param tags object = {}

@description('Name of the Log Analytics workspace')
param logAnalyticsWorkspaceName string = '${name}-logs'

// Log Analytics Workspace for Container Apps
resource logAnalyticsWorkspace 'Microsoft.OperationalInsights/workspaces@2022-10-01' = {
  name: logAnalyticsWorkspaceName
  location: location
  tags: tags
  properties: {
    sku: {
      name: 'PerGB2018'
    }
    retentionInDays: 30
  }
}

// Container Apps Environment
resource containerAppEnvironment 'Microsoft.App/managedEnvironments@2023-05-01' = {
  name: name
  location: location
  tags: tags
  properties: {
    appLogsConfiguration: {
      destination: 'log-analytics'
      logAnalyticsConfiguration: {
        customerId: logAnalyticsWorkspace.properties.customerId
        sharedKey: logAnalyticsWorkspace.listKeys().primarySharedKey
      }
    }
    zoneRedundant: false
  }
}

@description('The resource ID of the Container Apps Environment')
output environmentId string = containerAppEnvironment.id

@description('The name of the Container Apps Environment')
output environmentName string = containerAppEnvironment.name

@description('The default domain of the Container Apps Environment')
output defaultDomain string = containerAppEnvironment.properties.defaultDomain

@description('The resource ID of the Log Analytics workspace')
output logAnalyticsWorkspaceId string = logAnalyticsWorkspace.id
