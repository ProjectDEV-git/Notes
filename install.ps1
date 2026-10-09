# NoteTaker installer for Windows 10 and 11.
#
# One line in PowerShell sets up everything (Python, FFmpeg, Git, Ollama, the
# notes model, and the `notes` command):
#
#   irm https://raw.githubusercontent.com/ProjectDEV-git/Notes/main/install.ps1 | iex
#
# Or double-click "Install NoteTaker (Windows).cmd" in a downloaded copy.
#
# Every helper app comes from winget, Microsoft's own installer, so nothing is
# downloaded from random websites. Written for Windows PowerShell 5.1, which
# every Windows 10/11 PC has, so no newer syntax is used.

param(
    [switch]$Yes,
    [switch]$NoInstall
)

# Native tools write progress to stderr; with "Stop", PowerShell 5.1 would
# treat that as a failure. Real failures are checked explicitly instead.
$ErrorActionPreference = "Continue"
$Model = "llama3.2:3b"
$RepoUrl = if ($env:NOTETAKER_REPO) { $env:NOTETAKER_REPO } else { "https://github.com/ProjectDEV-git/Notes.git" }

function Say($text) { Write-Host $text }
function Step($text) { Write-Host ""; Write-Host $text -ForegroundColor White }
function Warn($text) { Write-Host $text -ForegroundColor Yellow }
function Ok($text) { Write-Host "  OK  $text" -ForegroundColor Green }

function Confirm-Step($question) {
    if ($Yes) { return $true }
    $reply = Read-Host "$question [Y/n]"
    return ($reply -eq "" -or $reply -match "^[Yy]")
}

function Has($command) {
    return [bool](Get-Command $command -ErrorAction SilentlyContinue)
}

# winget changes PATH in the registry, not in this window. Re-read it so a
# tool installed a moment ago can be used straight away.
function Update-SessionPath {
    $machine = [Environment]::GetEnvironmentVariable("Path", "Machine")
    $user = [Environment]::GetEnvironmentVariable("Path", "User")
    $env:Path = "$machine;$user"
}

function Install-WithWinget($id, $name) {
    Say "  > winget install --id $id"
    winget install --id $id --exact --silent --accept-source-agreements --accept-package-agreements | Out-Host
    Update-SessionPath
}

# Run a Python given as "py -3.12" or "python": the launcher plus its flags.
function Invoke-Python($spec) {
    $parts = $spec.Split(" ")
    $exe = $parts[0]
    $rest = @()
    if ($parts.Length -gt 1) { $rest = $parts[1..($parts.Length - 1)] }
    $pyArgs = $args
    & $exe @rest @pyArgs
}

function Find-Python {
    # "python" on a fresh PC is a Store shortcut that opens the Store instead
    # of running anything, so check it actually works.
    foreach ($candidate in @("py -3.12", "py -3", "python")) {
        if (-not (Has $candidate.Split(" ")[0])) { continue }
        try {
            Invoke-Python $candidate -c "import sys; sys.exit(sys.version_info < (3, 9))" 2>$null
            if ($LASTEXITCODE -eq 0) { return $candidate }
        } catch { }
    }
    return $null
}

# `exit` would close the student's PowerShell window when run via `irm | iex`,
# taking the explanation with it. Stopping with an error keeps it on screen.
function Stop-Install($message) {
    Warn $message
    throw "NoteTaker setup stopped."
}

function Ollama-Up {
    try {
        Invoke-WebRequest -UseBasicParsing -TimeoutSec 3 "http://localhost:11434/api/tags" | Out-Null
        return $true
    } catch { return $false }
}

# --------------------------------------------------------------------------
# Where the app lives. Run from a download, use that folder; run through
# `irm | iex`, fetch a copy into %USERPROFILE%\NoteTaker first.
# --------------------------------------------------------------------------
$AppDir = $null
if ($PSScriptRoot -and (Test-Path (Join-Path $PSScriptRoot "notetaker"))) {
    $AppDir = $PSScriptRoot
}

Say "NoteTaker setup for Windows"
if ($NoInstall) { Warn "-NoInstall: checking only, nothing will be installed" }

if (-not (Has "winget")) {
    Warn "winget (Microsoft's App Installer) is missing or out of date."
    Say  "The Microsoft Store is opening on 'App Installer'. Click Get or Update,"
    Say  "then run this installer again."
    Start-Process "ms-windows-store://pdp/?productid=9NBLGGH4NNS1"
    Stop-Install "Install App Installer, then run this again."
}

# --------------------------------------------------------------------------
# One plain-English question instead of a list of tool names.
# --------------------------------------------------------------------------
$python = Find-Python
$needs = @()
if (-not $python)        { $needs += "Python     runs NoteTaker" }
if (-not (Has "ffmpeg")) { $needs += "FFmpeg     records the sound" }
if (-not (Has "git"))    { $needs += "Git        downloads NoteTaker and its updates" }
if (-not (Has "ollama")) { $needs += "Ollama     the app that writes your notes, offline" }

if ($needs.Count -gt 0 -and -not $NoInstall) {
    Step "NoteTaker needs a few free helper apps"
    foreach ($item in $needs) { Say "  * $item" }
    Say "  * the notes model, a one-time download of about 2 GB"
    Say ""
    Say "This takes 10-20 minutes. Keep the laptop plugged in and online."
    Say "If Windows asks 'Do you want to allow this app to make changes', click Yes."
    if (-not (Confirm-Step "Install all of these now?")) {
        Stop-Install "Nothing was installed. Run this again when you are ready."
    }
    if (-not $python)        { Install-WithWinget "Python.Python.3.12" "Python" ; $python = Find-Python }
    if (-not (Has "ffmpeg")) { Install-WithWinget "Gyan.FFmpeg" "FFmpeg" }
    if (-not (Has "git"))    { Install-WithWinget "Git.Git" "Git" }
    if (-not (Has "ollama")) { Install-WithWinget "Ollama.Ollama" "Ollama" }
}

if (-not $AppDir) {
    $AppDir = if ($env:NOTETAKER_DIR) { $env:NOTETAKER_DIR } else { Join-Path $env:USERPROFILE "NoteTaker" }
    Step "Getting NoteTaker"
    if (Test-Path (Join-Path $AppDir ".git")) {
        Say "  already downloaded to $AppDir, updating it"
        git -C $AppDir pull --ff-only | Out-Host
    } elseif (Test-Path $AppDir) {
        Stop-Install "$AppDir already exists and is not a NoteTaker download. Move it, or set NOTETAKER_DIR to somewhere else."
    } else {
        if (-not (Has "git")) { Stop-Install "Git is needed to download NoteTaker." }
        git clone --quiet $RepoUrl $AppDir
        Ok "downloaded to $AppDir"
    }
}

# --------------------------------------------------------------------------
# Python environment
# --------------------------------------------------------------------------
Step "1/4  Python environment"
$venvPython = Join-Path $AppDir ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    if (-not $python) {
        Stop-Install "Python was not found. Install it with: winget install Python.Python.3.12"
    }
    Invoke-Python $python -m venv (Join-Path $AppDir ".venv")
    Ok "created .venv"
}
& $venvPython -c "import faster_whisper, rich" 2>$null
if ($LASTEXITCODE -eq 0) {
    Ok "Python dependencies already installed"
} else {
    Say "  installing Python dependencies (a few minutes)..."
    & $venvPython -m pip install -q --upgrade pip | Out-Null
    & $venvPython -m pip install -q -r (Join-Path $AppDir "requirements.txt")
    Ok "Python dependencies ready"
}

# --------------------------------------------------------------------------
# Ollama and the notes model
# --------------------------------------------------------------------------
Step "2/4  Ollama (writes the notes)"
if (Has "ollama") {
    if (-not (Ollama-Up)) {
        # The tray app keeps the server running and starts with Windows.
        $app = Join-Path $env:LOCALAPPDATA "Programs\Ollama\ollama app.exe"
        if (Test-Path $app) { Start-Process $app } else { Start-Process "ollama" "serve" -WindowStyle Hidden }
        for ($i = 0; $i -lt 40 -and -not (Ollama-Up); $i++) { Start-Sleep -Milliseconds 500 }
    }
    if (Ollama-Up) {
        Ok "Ollama is running"
        $listed = (ollama list) -join "`n"
        if ($listed -match [regex]::Escape($Model.Split(":")[0])) {
            Ok "$Model already downloaded"
        } elseif (-not $NoInstall) {
            Say "  downloading the notes model (about 2 GB, one time)..."
            ollama pull $Model
        }
    } else {
        Warn "Ollama is installed but not running. Open Ollama from the Start menu."
    }
} else {
    Warn "Ollama is not installed. Recording works; notes need: winget install Ollama.Ollama"
}

# --------------------------------------------------------------------------
# The `notes` command, a Desktop icon, and PATH
# --------------------------------------------------------------------------
Step "3/4  The 'notes' command"
$binDir = Join-Path $env:LOCALAPPDATA "NoteTaker\bin"
New-Item -ItemType Directory -Force -Path $binDir | Out-Null
$launcher = Join-Path $binDir "notes.cmd"
$cmd = "@echo off`r`n" +
       "setlocal`r`n" +
       "set `"PYTHONPATH=$AppDir`"`r`n" +
       "set `"PYTHONUTF8=1`"`r`n" +
       "`"$venvPython`" -m notetaker.shortcuts %*`r`n"
[IO.File]::WriteAllText($launcher, $cmd)
Ok "installed $launcher"

$userPath = [Environment]::GetEnvironmentVariable("Path", "User")
if (-not $userPath) { $userPath = "" }
if (($userPath.Split(";") | Where-Object { $_ -eq $binDir }).Count -eq 0) {
    [Environment]::SetEnvironmentVariable("Path", ($userPath.TrimEnd(";") + ";" + $binDir).TrimStart(";"), "User")
    Ok "added $binDir to your PATH (open a new window to use 'notes')"
}
Update-SessionPath

$madeIcon = $false
if (Confirm-Step "Put a NoteTaker icon on your Desktop?") {
    $shell = New-Object -ComObject WScript.Shell
    $link = $shell.CreateShortcut((Join-Path ([Environment]::GetFolderPath("Desktop")) "NoteTaker.lnk"))
    $link.TargetPath = "$env:WINDIR\System32\cmd.exe"
    $link.Arguments = "/k `"$launcher`""
    $link.WorkingDirectory = $env:USERPROFILE
    $link.Description = "Record a class and write the notes"
    $link.Save()
    Ok "double-click NoteTaker on your Desktop to start"
    $madeIcon = $true
}

# --------------------------------------------------------------------------
# Prove it works, including the microphone privacy switch
# --------------------------------------------------------------------------
Step "4/4  Checking that everything works"
$env:PYTHONPATH = $AppDir
& $venvPython -m notetaker.cli check --listen
if (Confirm-Step "Do you record online classes (Zoom, Teams, YouTube)? Set them up now?") {
    & $venvPython -m notetaker.cli setup-online
}

Say ""
if ($madeIcon) { Say "Done. Double-click NoteTaker on your Desktop, or open a new window and type:" }
else { Say "Done. Open a new PowerShell or Command Prompt window and type:" }
Say ""
Say "  notes"
