<#
.SYNOPSIS
Transcribes Urdu lecture recordings to Word documents (.docx).
.EXAMPLE
.\Transcribe.ps1 'C:\Lectures\lecture1.mp3'
.EXAMPLE
.\Transcribe.ps1 'C:\Lectures'
.EXAMPLE
.\Transcribe.ps1 'C:\Lectures\lecture1.mp3' --start 00:10:00 --duration 00:05:00
.EXAMPLE
.\Transcribe.ps1 'C:\Lectures' --rebuild
.NOTES
The first run creates a private Python environment (.venv) and installs the
required libraries; the first transcription also downloads the Whisper model
(about 1.6 GB). Requires Python 3.10+ and FFmpeg on PATH.
Results go to the transcripts folder next to this script.
All arguments are passed to transcribe.py; run with --help for every option.
#>
$ErrorActionPreference = 'Stop'
$python = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
# Written only after the libraries install successfully, so a failed setup is retried.
$installed = Join-Path $PSScriptRoot '.venv\installed.txt'

if (-not (Test-Path -LiteralPath $installed -PathType Leaf)) {
    if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
        # Skip the Microsoft Store shortcut in WindowsApps, which only opens the Store, and
        # take the first real installation when several are on PATH.
        $system = Get-Command python, python3 -CommandType Application -All -ErrorAction SilentlyContinue |
            Where-Object { $_.Source -notlike '*\WindowsApps\*' } | Select-Object -First 1
        if (-not $system) {
            throw 'Python was not found. Install Python 3.10 or newer from python.org and tick "Add python.exe to PATH".'
        }
        Write-Host "First run: setting up the Python environment with $($system.Source)..."
        & $system.Source -m venv (Join-Path $PSScriptRoot '.venv')
        if ($LASTEXITCODE -ne 0) { throw 'Could not create the Python environment.' }
    }
    Write-Host 'Installing the required libraries (a few minutes, first run only)...'
    # pip prints warnings to stderr, which Windows PowerShell 5.1 treats as errors under 'Stop'.
    $ErrorActionPreference = 'Continue'
    & $python -m pip install --quiet --disable-pip-version-check -r (Join-Path $PSScriptRoot 'requirements.txt')
    $pipExit = $LASTEXITCODE
    $ErrorActionPreference = 'Stop'
    if ($pipExit -ne 0) { throw 'Could not install the required libraries. Check the internet connection and run again.' }
    Set-Content -LiteralPath $installed -Value (Get-Date -Format s)
}
if (-not (Get-Command ffmpeg -CommandType Application -ErrorAction SilentlyContinue)) {
    throw 'FFmpeg was not found. Install it with: winget install Gyan.FFmpeg (then reopen PowerShell).'
}

# Model download progress goes to stderr, which Windows PowerShell 5.1 can treat as an error under 'Stop'.
$ErrorActionPreference = 'Continue'
& $python (Join-Path $PSScriptRoot 'transcribe.py') @args
exit $LASTEXITCODE
