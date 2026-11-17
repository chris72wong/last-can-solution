param([switch]$DemoOnly)
$ErrorActionPreference = 'Stop'
& (Join-Path $PSScriptRoot 'tools\setup.ps1') -DemoOnly:$DemoOnly
