<#
.SYNOPSIS
Installs or updates the projekt-base conda environment.

.DESCRIPTION
This script creates or updates the projekt-base environment which contains all dependencies
for running all methods in the project. All methods are executed through this single environment.

.PARAMETER Recreate
Remove an existing environment first, then create it again from the YAML file.

.PARAMETER ListOnly
Print the environment name and details without creating or updating anything.

.EXAMPLE
.\install-conda-envs.ps1

Creates or updates the projekt-base environment.

.EXAMPLE
.\install-conda-envs.ps1 -Recreate

Deletes the existing projekt-base environment and creates it again from scratch.

.EXAMPLE
.\install-conda-envs.ps1 -ListOnly

Shows environment details without making any changes.

.NOTES
- The projekt-base environment contains all dependencies for all methods.
- All methods are executed through: conda run -n projekt-base python -m launcher.gui
- Conda output (solver/install) is displayed in real-time.
#>
[CmdletBinding()]
param(
    [switch]$Recreate,
    [switch]$ListOnly
)

$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent $PSScriptRoot
$envDir = Join-Path $repoRoot 'conda_envs'
$baseEnvFile = Join-Path $envDir 'base.yml'

if (-not (Test-Path $baseEnvFile)) {
    throw "base.yml not found at $baseEnvFile"
}

function Get-CondaCommand {
    if ($env:CONDA_EXE -and (Test-Path $env:CONDA_EXE)) {
        return $env:CONDA_EXE
    }

    $condaCommand = Get-Command conda -ErrorAction SilentlyContinue
    if ($null -ne $condaCommand) {
        return $condaCommand.Source
    }

    $fallbackCandidates = @(
        'C:\ProgramData\anaconda3\Scripts\conda.exe'
        'C:\ProgramData\miniconda3\Scripts\conda.exe'
        "$env:USERPROFILE\anaconda3\Scripts\conda.exe"
        "$env:USERPROFILE\miniconda3\Scripts\conda.exe"
        "$env:LOCALAPPDATA\anaconda3\Scripts\conda.exe"
        "$env:LOCALAPPDATA\miniconda3\Scripts\conda.exe"
    )

    foreach ($fallback in $fallbackCandidates) {
        if ($fallback -and (Test-Path $fallback)) {
            return $fallback
        }
    }

    throw 'Could not find conda. Put it in PATH or install Anaconda at C:\ProgramData\anaconda3.'
}

function Get-EnvNameFromFile {
    param([Parameter(Mandatory = $true)][string]$Path)

    foreach ($line in Get-Content -Path $Path -TotalCount 20) {
        if ($line -match '^\s*name:\s*(.+?)\s*$') {
            return $Matches[1].Trim()
        }
    }

    throw "Could not read env name from $Path"
}

function Test-EnvExists {
    param(
        [Parameter(Mandatory = $true)][string]$CondaCommand,
        [Parameter(Mandatory = $true)][string]$EnvName
    )

    $jsonText = & $CondaCommand env list --json | Out-String
    $envList = $jsonText | ConvertFrom-Json

    foreach ($prefix in $envList.envs) {
        if ((Split-Path -Path $prefix -Leaf) -eq $EnvName) {
            return $true
        }
    }

    return $false
}

function Invoke-CondaStreaming {
    param(
        [Parameter(Mandatory = $true)][string]$CondaCommand,
        [Parameter(Mandatory = $true)][string[]]$Arguments,
        [Parameter(Mandatory = $true)][string]$OperationLabel
    )

    Write-Host "  Conda output (live):" -ForegroundColor DarkGray
    Write-Host "  > conda $($Arguments -join ' ')" -ForegroundColor DarkGray
    & $CondaCommand @Arguments

    if ($LASTEXITCODE -ne 0) {
        throw "Failed to $OperationLabel"
    }
}

$condaCommand = Get-CondaCommand
$envName = Get-EnvNameFromFile -Path $baseEnvFile

if ($ListOnly) {
    Write-Host ''
    Write-Host "Environment to be processed:" -ForegroundColor Cyan
    Write-Host "  Name: $envName" -ForegroundColor DarkGray
    Write-Host "  File: $baseEnvFile" -ForegroundColor DarkGray
    Write-Host ''
    exit 0
}

Write-Host ''
Write-Host "Processing: base" -ForegroundColor Cyan

$envExists = Test-EnvExists -CondaCommand $condaCommand -EnvName $envName

if ($Recreate -and $envExists) {
    Write-Host "  Step: remove existing env $envName" -ForegroundColor Yellow
    Invoke-CondaStreaming -CondaCommand $condaCommand -Arguments @('env', 'remove', '-n', $envName, '-y') -OperationLabel "remove environment $envName"
    $envExists = $false
}

if ($envExists) {
    Write-Host "  Step: update env $envName" -ForegroundColor Green
    Invoke-CondaStreaming -CondaCommand $condaCommand -Arguments @('env', 'update', '-f', $baseEnvFile, '--prune') -OperationLabel "update environment $envName"
}
else {
    Write-Host "  Step: create env $envName" -ForegroundColor Green
    Invoke-CondaStreaming -CondaCommand $condaCommand -Arguments @('env', 'create', '-f', $baseEnvFile) -OperationLabel "create environment $envName"
}

Write-Host "  Done: $envName" -ForegroundColor Green

Write-Host ''
Write-Host "Installation completed" -ForegroundColor Cyan
Write-Host 'Done.'