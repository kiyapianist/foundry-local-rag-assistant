param(
    [int]$Port = 8501
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)

Set-Location $ProjectRoot
$env:RAG_OFFLINE_MODE = "1"

& ".\venv\Scripts\streamlit.exe" run "app.py" `
    --server.port $Port `
    --server.address "127.0.0.1" `
    --server.headless true `
    --browser.gatherUsageStats false
