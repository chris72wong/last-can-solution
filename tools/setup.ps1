param([switch]$DemoOnly)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$setupPreviousTemp = $env:TEMP
$setupPreviousTmp = $env:TMP
Push-Location $projectRoot
try {
    New-Item -ItemType Directory -Path '.cache\tmp' -Force | Out-Null
    $env:TEMP = (Resolve-Path '.cache\tmp').Path
    $env:TMP = $env:TEMP
    if (-not (Test-Path '.venv\Scripts\python.exe')) {
        py -3.12 -m venv .venv
        if ($LASTEXITCODE -ne 0) { throw 'Could not create Python 3.12 virtual environment.' }
    }
    & '.venv\Scripts\python.exe' -m ensurepip --upgrade
    if ($LASTEXITCODE -ne 0) { throw 'Could not install pip in the virtual environment.' }
    if ($DemoOnly) {
        & '.venv\Scripts\python.exe' -m pip install -r requirements-demo.txt --cache-dir .cache\pip
        if ($LASTEXITCODE -ne 0) { throw 'Demo dependency installation failed.' }
        Write-Output 'Demo dependencies ready. Run .venv\Scripts\python.exe main.py --demo'
        return
    }
    & '.venv\Scripts\python.exe' -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128 --cache-dir .cache\pip
    if ($LASTEXITCODE -ne 0) { throw 'CUDA PyTorch installation failed.' }
    $dependencyFile = if (Test-Path 'requirements-lock.txt') { 'requirements-lock.txt' } else { 'requirements.txt' }
    & '.venv\Scripts\python.exe' -m pip install -r $dependencyFile --cache-dir .cache\pip
    if ($LASTEXITCODE -ne 0) { throw 'Application dependency installation failed.' }
    & '.venv\Scripts\python.exe' tools\check_environment.py
    if ($LASTEXITCODE -ne 0) { throw 'GPU/environment check failed; see output above.' }
} finally {
    [Environment]::SetEnvironmentVariable('TEMP', $setupPreviousTemp, 'Process')
    [Environment]::SetEnvironmentVariable('TMP', $setupPreviousTmp, 'Process')
    Pop-Location
}
