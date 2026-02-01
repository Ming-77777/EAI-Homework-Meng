$ErrorActionPreference = "Stop"
$PyVersion = "3.13"

try {
  Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned -Force -ErrorAction Stop
} catch {
  Write-Host "WARNING: Could not set execution policy (possibly managed by IT)."
}

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
  Write-Host "ERROR: uv is not installed."
  Write-Host "Install uv first:"
  Write-Host '  powershell -ExecutionPolicy ByPass -NoProfile -Command "irm https://astral.sh/uv/install.ps1 | iex"'
  exit 1
}

Write-Host "Installing Python $PyVersion..."
uv python install $PyVersion

Write-Host "Creating virtual environment and installing dependencies..."
uv sync

Write-Host "`u{2713} Setup complete!"
Write-Host ""
Write-Host "Next steps:"
Write-Host "  1. Open in VS Code: code ."
Write-Host "  2. Update README.md with your project description"
Write-Host "  3. Start coding in src/"
