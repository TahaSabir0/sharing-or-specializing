<#
.SYNOPSIS
Prepare Alex's E4/E5 decks and run the permitted smoke test on Blackwell GPU 1.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File code/train/smoke_blackwell.ps1 `
    -Config code/configs/e4/modxray_q35_s1337.yaml

The remote branch/config must already be present in Alex's personal clone. Use -Pull
only after pushing the branch; agent forwarding from ~/.ssh/config supplies GitHub auth.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^code/configs/e[45]/[a-z0-9_]+\.yaml$')]
    [string]$Config,

    [ValidatePattern('^[a-z0-9_]+$')]
    [string]$Who = 'alex',

    [string]$RemoteRepo = '',

    [string]$RemoteDecks = '',

    [switch]$Pull,
    [switch]$NoEval
)

$ErrorActionPreference = 'Stop'
if (-not $RemoteRepo) {
    $RemoteRepo = "/data/ugrip2026/journal/people/$Who/repo"
}
if (-not $RemoteDecks) {
    $RemoteDecks = "/data/ugrip2026/journal/people/$Who/group_decks"
}
$personalRoot = "/data/ugrip2026/journal/people/$Who/"
if (-not $RemoteRepo.StartsWith($personalRoot) -or -not $RemoteDecks.StartsWith($personalRoot)) {
    throw 'RemoteRepo and RemoteDecks must stay inside the selected teammate personal directory.'
}
if ($RemoteRepo -notmatch '^/[A-Za-z0-9_./-]+$' -or $RemoteDecks -notmatch '^/[A-Za-z0-9_./-]+$') {
    throw 'Remote paths contain unsupported shell characters.'
}
$pullCommand = if ($Pull) { 'git pull --ff-only' } else { ':' }
$noEvalArg = if ($NoEval) { ' --no-eval' } else { '' }

$remoteScript = @"
set -euo pipefail
cd '$RemoteRepo'
$pullCommand
echo "[blackwell] repo: `$(git rev-parse --abbrev-ref HEAD) @ `$(git rev-parse --short HEAD)"
echo '[blackwell] GPU 1:'
nvidia-smi -i 1 --query-gpu=index,name,memory.used,memory.free,utilization.gpu --format=csv
if [ -f /data/ugrip2026/journal/runs/smoke/.holder ]; then
  echo "[blackwell] smoke holder: `$(cat /data/ugrip2026/journal/runs/smoke/.holder)"
fi
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
python code/data/build_group_decks.py \
  --source_dir /data/ugrip2026/journal/data \
  --out_dir '$RemoteDecks'
export JOURNAL_DATA='$RemoteDecks'
bash code/train/smoke.sh '$Who' '$Config'$noEvalArg
"@

$bytes = [System.Text.Encoding]::UTF8.GetBytes($remoteScript)
$encoded = [Convert]::ToBase64String($bytes)
Write-Host "Running Alex smoke test through SSH alias 'blackwell' (GPU 1 only)..."
& ssh blackwell "echo '$encoded' | base64 --decode | bash"
if ($LASTEXITCODE -ne 0) {
    throw "Blackwell smoke test failed with exit code $LASTEXITCODE"
}
