$ErrorActionPreference = "Stop"

$scriptPath = Join-Path $PSScriptRoot "..\fetch_pcb_stability_samples.ps1"
$content = Get-Content -Raw $scriptPath

if ($content -match '\[string\]\$OutputDirectory\s*=\s*\(Join-Path\s+\$PSScriptRoot') {
    throw "OutputDirectory must not use PSScriptRoot in a parameter default expression"
}

if ($content -notmatch 'if\s*\(\[string\]::IsNullOrWhiteSpace\(\$OutputDirectory\)\)') {
    throw "OutputDirectory must be initialized after parameter binding"
}

Write-Host "PASS: output directory is initialized after parameter binding"
