@description('Name of the Azure Container Registry')
param name string

@description('Location for the registry')
param location string = resourceGroup().location

@description('Tags to apply to the registry')
param tags object = {}

@description('SKU for the registry')
@allowed([
  'Basic'
  'Standard'
  'Premium'
])
param sku string = 'Basic'

resource containerRegistry 'Microsoft.ContainerRegistry/registries@2023-07-01' = {
  name: name
  location: location
  tags: tags
  sku: {
    name: sku
  }
  properties: {
    adminUserEnabled: true
    publicNetworkAccess: 'Enabled'
  }
}

@description('The login server URL of the registry')
output loginServer string = containerRegistry.properties.loginServer

@description('The name of the registry')
output name string = containerRegistry.name

@description('The resource ID of the registry')
output resourceId string = containerRegistry.id
