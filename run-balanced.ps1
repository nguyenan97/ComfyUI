param([ValidateRange(1024, 65535)][int]$Port = 8188)
$ErrorActionPreference = 'Stop'
$python = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (!(Test-Path -LiteralPath $python)) { throw "Missing Python environment: $python" }
$logDirectory = Join-Path $PSScriptRoot 'user\default\local-run'
New-Item -ItemType Directory -Force -Path $logDirectory | Out-Null
$serverArgs = @(
    'main.py', '--listen', '127.0.0.1', '--port', "$Port",
    '--lowvram', '--reserve-vram', '1.5', '--force-fp32', '--fp32-vae',
    '--disable-dynamic-vram', '--disable-async-offload', '--disable-cuda-malloc',
    '--offline', '--disable-all-custom-nodes', '--preview-method', 'none'
)
$serverProcess = Start-Process -FilePath $python -ArgumentList $serverArgs `
    -WorkingDirectory $PSScriptRoot -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput (Join-Path $logDirectory 'balanced.stdout.log') `
    -RedirectStandardError (Join-Path $logDirectory 'balanced.stderr.log')
$serverProcess.Id | Set-Content -LiteralPath (Join-Path $logDirectory 'balanced.pid')
Write-Output "ComfyUI starting: http://127.0.0.1:$Port (launcher PID $($serverProcess.Id))"
