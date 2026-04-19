<#
.SYNOPSIS
Installs or updates conda environments from the conda_envs folder.

.DESCRIPTION
By default, the script processes every *.yml file in conda_envs.
You can also target one or more methods by name, using the same base name as the YAML file.

.PARAMETER Methods
One or more method names, for example yolov8_seg or resnet_cls.

.PARAMETER All
Process every environment file in conda_envs.

.PARAMETER Recreate
Remove an existing environment first, then create it again from the YAML file.

.PARAMETER ListOnly
Print the resolved environment names and files without creating or updating anything.

.EXAMPLE
.\install-conda-envs.ps1

Updates or creates every environment defined in conda_envs.

.EXAMPLE
.\install-conda-envs.ps1 -Methods yolov8_seg

Installs or updates only the YOLOv8 segmentation environment.

.EXAMPLE
.\install-conda-envs.ps1 -Methods yolov8_seg -Recreate

Deletes the existing yolov8_seg environment and creates it again.

.EXAMPLE
.\install-conda-envs.ps1 -ListOnly

Shows which environment names will be used.
#>
[CmdletBinding()]
param(
    [string[]]$Methods,
    [switch]$All,
    [switch]$Recreate,
    [switch]$ListOnly
)

$ErrorActionPreference = 'Stop'

$repoRoot = $PSScriptRoot
$envDir = Join-Path $repoRoot 'conda_envs'

function Get-CondaCommand {
    $condaCommand = Get-Command conda -ErrorAction SilentlyContinue
    if ($null -ne $condaCommand) {
        return $condaCommand.Source
    }

    $fallback = 'C:\ProgramData\anaconda3\Scripts\conda.exe'
    if (Test-Path $fallback) {
        return $fallback
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

function Get-TargetEnvFiles {
    if ($All -or -not $Methods) {
        return Get-ChildItem -Path $envDir -Filter '*.yml' | Sort-Object Name
    }

    $files = foreach ($method in $Methods) {
        $candidate = Join-Path $envDir "$method.yml"
        if (-not (Test-Path $candidate)) {
            throw "Unknown method '$method'. Expected a file at $candidate"
        }

        Get-Item $candidate
    }

    return $files
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

$condaCommand = Get-CondaCommand
$envFiles = Get-TargetEnvFiles

foreach ($envFile in $envFiles) {
    $envName = Get-EnvNameFromFile -Path $envFile.FullName
    $envExists = Test-EnvExists -CondaCommand $condaCommand -EnvName $envName

    if ($ListOnly) {
        Write-Host "[$($envFile.BaseName)] $envName -> $($envFile.FullName)"
        continue
    }

    if ($Recreate -and $envExists) {
        Write-Host "Removing existing env: $envName"
        & $condaCommand env remove -n $envName -y
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to remove $envName"
        }
        $envExists = $false
    }

    if ($envExists) {
        Write-Host "Updating env: $envName"
        & $condaCommand env update -f $envFile.FullName --prune
    }
    else {
        Write-Host "Creating env: $envName"
        & $condaCommand env create -f $envFile.FullName
    }

    if ($LASTEXITCODE -ne 0) {
        throw "Failed to process $envName"
    }
}

Write-Host 'Done.'