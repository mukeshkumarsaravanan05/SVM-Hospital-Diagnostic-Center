$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path

Set-Location $ProjectRoot

Write-Host "=============================================="
Write-Host "SVM HOSPITAL DIAGNOSTIC CENTER"
Write-Host "AI-Assisted Chest X-ray Disease Prediction"
Write-Host "=============================================="
Write-Host ""

python -m ui.main_window_FINAL
