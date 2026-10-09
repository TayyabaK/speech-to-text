<#
.SYNOPSIS
Transcribes one Urdu recording into short, timestamped clips for training a text-to-speech voice.
.EXAMPLE
.\Transcribe-ForTTS.ps1 'C:\Users\tayya\Downloads\rida-sohaib.mp3' --start 00:10:00 --duration 00:02:00
.EXAMPLE
.\Transcribe-ForTTS.ps1 'C:\Users\tayya\Downloads\rida-sohaib.mp3' --exclude 'C:\path\removed.txt'
.EXAMPLE
.\Transcribe-ForTTS.ps1 'C:\Users\tayya\Downloads\rida-sohaib.mp3' --export 'transcripts\rida-sohaib_tts.srt'
.NOTES
Writes <name>_tts.srt, <name>_tts.tsv and <name>_tts_review.txt to the transcripts folder next
to this script. Every timestamp is on the original file's timeline. Clips are 2-15 seconds, cut
in pauses; Quran verses, duas and the --exclude ranges are left out and listed in the review file.
After proofreading the SRT, --export cuts the clips from the original audio into the
text-to-speech project (data\raw\wavs and data\raw\metadata.csv).
Uses the same Python environment, model and corrections as Transcribe.ps1.
Requires Python 3.10+ and FFmpeg on PATH.
All arguments are passed to transcribe_tts.py; run with --help for every option.
#>
$ErrorActionPreference = 'Stop'
$python = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
$requirements = Join-Path $PSScriptRoot 'requirements.txt'
# Holds a hash of requirements.txt, so libraries added later are installed into an existing environment.
$installed = Join-Path $PSScriptRoot '.venv\installed.txt'
$wanted = (Get-FileHash -LiteralPath $requirements -Algorithm SHA256).Hash
$current = if (Test-Path -LiteralPath $installed -PathType Leaf) { (Get-Content -LiteralPath $installed -TotalCount 1) } else { '' }

if ($current -ne $wanted) {
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
    & $python -m pip install --quiet --disable-pip-version-check -r $requirements
    $pipExit = $LASTEXITCODE
    $ErrorActionPreference = 'Stop'
    if ($pipExit -ne 0) { throw 'Could not install the required libraries. Check the internet connection and run again.' }
    Set-Content -LiteralPath $installed -Value $wanted
}
if (-not (Get-Command ffmpeg -CommandType Application -ErrorAction SilentlyContinue)) {
    throw 'FFmpeg was not found. Install it with: winget install Gyan.FFmpeg (then reopen PowerShell).'
}

# Model download progress goes to stderr, which Windows PowerShell 5.1 can treat as an error under 'Stop'.
$ErrorActionPreference = 'Continue'
& $python (Join-Path $PSScriptRoot 'transcribe_tts.py') @args
exit $LASTEXITCODE
