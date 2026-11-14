param(
    [switch]$SetupOnly = $false
)

$ErrorActionPreference = "Stop"
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12 -bor [Net.SecurityProtocolType]::Tls13
$ProgressPreference = 'SilentlyContinue'

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $ProjectRoot

$RuntimeDir = Join-Path $ProjectRoot ".runtime"
$PortablePyDir = Join-Path $RuntimeDir "python"
$PortableFfmpegDir = Join-Path $RuntimeDir "ffmpeg"
$VenvDir = Join-Path $ProjectRoot ".venv"

New-Item -ItemType Directory -Force -Path $RuntimeDir | Out-Null

function Write-Step($msg) {
    Write-Host "`n========================================================" -ForegroundColor Cyan
    Write-Host "  $msg" -ForegroundColor Cyan
    Write-Host "========================================================" -ForegroundColor Cyan
}

function Test-ValidPython($exePath) {
    if (-not $exePath -or -not (Test-Path $exePath)) { return $false }
    if ($exePath -like "*WindowsApps*") { return $false }
    try {
        $ver = & $exePath -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null
        if ($LASTEXITCODE -eq 0 -and $ver -match "^3\.(9|10|11|12|13)") {
            return $true
        }
    } catch {}
    return $false
}

function Find-PythonExecutable {
    $localPy = Join-Path $PortablePyDir "python.exe"
    if (Test-ValidPython $localPy) { return $localPy }

    $cmd = Get-Command python -ErrorAction SilentlyContinue
    if ($cmd -and (Test-ValidPython $cmd.Source)) { return $cmd.Source }

    $pyLauncher = Get-Command py -ErrorAction SilentlyContinue
    if ($pyLauncher) {
        try {
            $pyPath = & py -3 -c "import sys; print(sys.executable)" 2>$null
            if ($LASTEXITCODE -eq 0 -and (Test-ValidPython $pyPath.Trim())) {
                return $pyPath.Trim()
            }
        } catch {}
    }

    $commonPaths = @(
        "$env:LocalAppData\Programs\Python\Python311\python.exe",
        "$env:LocalAppData\Programs\Python\Python312\python.exe",
        "$env:LocalAppData\Programs\Python\Python310\python.exe",
        "C:\Python311\python.exe",
        "C:\Python312\python.exe"
    )
    foreach ($p in $commonPaths) {
        if (Test-ValidPython $p) { return $p }
    }
    return $null
}

# 1. VERIFICAR OU INSTALAR PYTHON AUTOMATICAMENTE
Write-Step "1/4 - Verificando ambiente Python..."
$PythonExe = Find-PythonExecutable

if (-not $PythonExe) {
    Write-Host "Python nao encontrado no sistema. Tentando instalacao automatica..." -ForegroundColor Yellow

    $wingetCmd = Get-Command winget -ErrorAction SilentlyContinue
    if ($wingetCmd) {
        Write-Host "Tentando instalar Python 3.11 via Winget..." -ForegroundColor Yellow
        try {
            & winget install -e --id Python.Python.3.11 --scope user --silent --accept-package-agreements --accept-source-agreements
            $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
            $PythonExe = Find-PythonExecutable
        } catch {
            Write-Host "Winget nao concluiu a instalacao. Usando modo Python Portatil..." -ForegroundColor Yellow
        }
    }
}

if (-not $PythonExe) {
    Write-Host "Baixando Python 3.11.9 Portatil (Embeddable AMD64) em .runtime/python..." -ForegroundColor Yellow
    New-Item -ItemType Directory -Force -Path $PortablePyDir | Out-Null

    $pyZipUrl = "https://www.python.org/ftp/python/3.11.9/python-3.11.9-embed-amd64.zip"
    $pyZipPath = Join-Path $RuntimeDir "python-3.11.9-embed-amd64.zip"
    Invoke-WebRequest -Uri $pyZipUrl -OutFile $pyZipPath -UseBasicParsing
    Expand-Archive -Path $pyZipPath -DestinationPath $PortablePyDir -Force
    Remove-Item $pyZipPath -Force -ErrorAction SilentlyContinue

    # Habilitar site-packages no python311._pth
    $pthFile = Join-Path $PortablePyDir "python311._pth"
    if (Test-Path $pthFile) {
        $content = Get-Content $pthFile
        $content = $content -replace "#import site", "import site"
        Set-Content -Path $pthFile -Value $content -Encoding ASCII
    }

    $PythonExe = Join-Path $PortablePyDir "python.exe"

    # Instalar get-pip.py e virtualenv no Python Portatil
    Write-Host "Configurando pip e virtualenv no Python Portatil..." -ForegroundColor Yellow
    $getPipUrl = "https://bootstrap.pypa.io/get-pip.py"
    $getPipPath = Join-Path $PortablePyDir "get-pip.py"
    Invoke-WebRequest -Uri $getPipUrl -OutFile $getPipPath -UseBasicParsing
    & $PythonExe $getPipPath --no-warn-script-location
    & $PythonExe -m pip install --upgrade pip virtualenv --no-warn-script-location
}

Write-Host "Python pronto em: $PythonExe" -ForegroundColor Green

# 2. VERIFICAR OU BAIXAR FFMPEG PORTÁTIL
Write-Step "2/4 - Verificando FFmpeg e FFprobe..."
$localFfmpeg = Join-Path $PortableFfmpegDir "ffmpeg.exe"
$localFfprobe = Join-Path $PortableFfmpegDir "ffprobe.exe"

$sysFfmpeg = Get-Command ffmpeg -ErrorAction SilentlyContinue
$sysFfprobe = Get-Command ffprobe -ErrorAction SilentlyContinue

if ((Test-Path $localFfmpeg) -and (Test-Path $localFfprobe)) {
    $env:PATH = "$PortableFfmpegDir;$env:PATH"
    Write-Host "FFmpeg portatil encontrado em: $PortableFfmpegDir" -ForegroundColor Green
} elseif ($sysFfmpeg -and $sysFfprobe) {
    Write-Host "FFmpeg encontrado no PATH do sistema: $($sysFfmpeg.Source)" -ForegroundColor Green
} else {
    Write-Host "FFmpeg nao encontrado. Baixando ffmpeg-release-essentials.zip automaticamente..." -ForegroundColor Yellow
    New-Item -ItemType Directory -Force -Path $PortableFfmpegDir | Out-Null

    $ffmpegZipPath = Join-Path $RuntimeDir "ffmpeg-release-essentials.zip"
    $ffmpegExtractDir = Join-Path $RuntimeDir "ffmpeg_extracted"
    $ffmpegUrl = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"

    try {
        Invoke-WebRequest -Uri $ffmpegUrl -OutFile $ffmpegZipPath -UseBasicParsing
    } catch {
        Write-Host "Usando mirror secundario do GitHub (BtbN FFmpeg Builds)..." -ForegroundColor Yellow
        $ffmpegUrlFallback = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip"
        Invoke-WebRequest -Uri $ffmpegUrlFallback -OutFile $ffmpegZipPath -UseBasicParsing
    }

    if (Test-Path $ffmpegExtractDir) {
        Remove-Item $ffmpegExtractDir -Recurse -Force -ErrorAction SilentlyContinue
    }
    Expand-Archive -Path $ffmpegZipPath -DestinationPath $ffmpegExtractDir -Force

    $foundFfmpeg = Get-ChildItem -Path $ffmpegExtractDir -Recurse -Filter "ffmpeg.exe" | Select-Object -First 1
    $foundFfprobe = Get-ChildItem -Path $ffmpegExtractDir -Recurse -Filter "ffprobe.exe" | Select-Object -First 1

    if (-not $foundFfmpeg -or -not $foundFfprobe) {
        throw "Falha ao localizar ffmpeg.exe e ffprobe.exe dentro do pacote baixado."
    }

    Copy-Item $foundFfmpeg.FullName -Destination $localFfmpeg -Force
    Copy-Item $foundFfprobe.FullName -Destination $localFfprobe -Force

    Remove-Item $ffmpegZipPath -Force -ErrorAction SilentlyContinue
    Remove-Item $ffmpegExtractDir -Recurse -Force -ErrorAction SilentlyContinue

    $env:PATH = "$PortableFfmpegDir;$env:PATH"
    Write-Host "FFmpeg instalado com sucesso em: $PortableFfmpegDir" -ForegroundColor Green
}

# 3. CRIAR AMBIENTE VIRTUAL (.venv) E INSTALAR DEPENDÊNCIAS
Write-Step "3/4 - Configurando ambiente virtual (.venv) e bibliotecas..."
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"

if (-not (Test-Path $VenvPython)) {
    Write-Host "Criando ambiente virtual em .venv..." -ForegroundColor Yellow
    & $PythonExe -m venv $VenvDir 2>$null
    if (-not (Test-Path $VenvPython)) {
        Write-Host "Usando virtualenv para criar .venv (modo Python Embeddable)..." -ForegroundColor Yellow
        & $PythonExe -m pip install virtualenv --quiet
        & $PythonExe -m virtualenv $VenvDir
    }
}

if (-not (Test-Path $VenvPython)) {
    throw "Nao foi possivel criar o ambiente virtual em .venv."
}

Write-Host "Atualizando pip e verificando pacotes do requirements.txt..." -ForegroundColor Yellow
& $VenvPython -m pip install --upgrade pip --quiet
& $VenvPython -m pip install -r (Join-Path $ProjectRoot "requirements.txt") --quiet

Write-Host "Todas as dependencias estao instaladas e prontas!" -ForegroundColor Green

# 4. INICIAR SERVIDOR E ABRIR INTERFACE GRÁFICA
if ($SetupOnly) {
    Write-Step "4/4 - Modo SetupOnly concluido com sucesso!"
    exit 0
}

Write-Step "4/4 - Iniciando Tradutor & Dublador Automatico de Videos..."
Write-Host "O navegador abrira automaticamente em http://127.0.0.1:8000" -ForegroundColor Green
Write-Host "Pressione CTRL+C nesta janela quando quiser encerrar o aplicativo.`n" -ForegroundColor Gray

& $VenvPython (Join-Path $ProjectRoot "app.py")
