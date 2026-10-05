# EmbodiedAI one-shot verification (PowerShell wrapper)
#
# Usage:
#   .\scripts\verify.ps1            # full: pytest suite + CLI self-checks
#   .\scripts\verify.ps1 -NoPytest  # CLI self-checks only
#
# Exit code: 0 = VERIFY_OK, non-zero = at least one check failed.
# NOTE: all assertions live in src/embodied_ai/verification.py; this script only
#       orchestrates, so there is no second copy of the checks to maintain.
# NOTE: kept ASCII-only on purpose (PowerShell 5.1 reads BOM-less .ps1 as the
#       ANSI codepage; non-ASCII bytes can corrupt parsing).

param(
    [switch]$NoPytest
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

$env:PYTHONPATH = Join-Path $repoRoot "src"
$env:PYTHONIOENCODING = "utf-8"

Write-Host "=== EmbodiedAI verify (repo: $repoRoot) ===" -ForegroundColor Cyan

$cliArgs = @("-m", "embodied_ai.cli", "verify")
if ($NoPytest) { $cliArgs += "--no-pytest" }

& python @cliArgs
$code = $LASTEXITCODE

Write-Host ""
if ($code -eq 0) {
    Write-Host "RESULT=VERIFY_OK" -ForegroundColor Green
} else {
    Write-Host "RESULT=VERIFY_FAIL (exit=$code)" -ForegroundColor Red
}
exit $code
