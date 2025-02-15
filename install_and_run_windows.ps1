# Устанавливаем кодировку UTF-8 для PowerShell
$OutputEncoding = [System.Text.Encoding]::UTF8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
[Console]::InputEncoding = [System.Text.Encoding]::UTF8
$PSDefaultParameterValues['*:Encoding'] = 'utf8'

# Функция для установки кодировки консоли
function Set-ConsoleEncoding {
    try {
        # Устанавливаем кодировку консоли в UTF-8
        chcp 65001 | Out-Null
        # Дополнительная настройка для корректного отображения
        $Host.UI.RawUI.WindowTitle = "Sexy Bot Installer"
    } catch {
        Write-Red "Ошибка при установке кодировки консоли:"
        Write-Red $_.Exception.Message
    }
}

# Устанавливаем кодировку консоли
Set-ConsoleEncoding

# Функции для цветного вывода
function Write-Green {
    param([string]$Text)
    Write-Host $Text -ForegroundColor Green
}

function Write-Yellow {
    param([string]$Text)
    Write-Host $Text -ForegroundColor Yellow
}

function Write-Red {
    param([string]$Text)
    Write-Host $Text -ForegroundColor Red
}

Write-Green "=== Установка и настройка Sexy Bot ==="

# Проверка и создание .env файла с API ключом
$envFile = ".env"
if (Test-Path $envFile) {
    Get-Content $envFile -Encoding UTF8 | ForEach-Object {
        if ($_ -match '^GOOGLE_API_KEY=(.*)$') {
            $env:GOOGLE_API_KEY = $matches[1]
        }
    }
}

if (-not $env:GOOGLE_API_KEY) {
    Write-Yellow "Введите ваш Google API ключ:"
    $apiKey = Read-Host
    Set-Content $envFile "GOOGLE_API_KEY=$apiKey" -Encoding UTF8
    $env:GOOGLE_API_KEY = $apiKey
}

# Функция для проверки наличия команды
function Test-Command {
    param([string]$Command)
    return [bool](Get-Command -Name $Command -ErrorAction SilentlyContinue)
}

# Функция для проверки версии Python
function Test-PythonVersion {
    try {
        $pythonVersion = python -c "import sys; print('.'.join(map(str, sys.version_info[:2])))"
        Write-Yellow "Текущая версия Python: $pythonVersion"
        return $pythonVersion.StartsWith("3.10")
    } catch {
        return $false
    }
}

# Функция для удаления Python
function Uninstall-Python {
    Write-Yellow "Удаление текущей версии Python..."
    choco uninstall -y python3
    # Очищаем переменные окружения от старых путей Python
    $paths = [System.Environment]::GetEnvironmentVariable("Path", "Machine").Split(';')
    $newPaths = $paths | Where-Object { $_ -notmatch "Python" }
    $newPathString = $newPaths -join ';'
    [System.Environment]::SetEnvironmentVariable("Path", $newPathString, "Machine")
    $env:Path = $newPathString
}

# Установка Chocolatey если его нет
function Install-Chocolatey {
    if (-not (Test-Command choco)) {
        Write-Green "Установка Chocolatey..."
        Set-ExecutionPolicy Bypass -Scope Process -Force
        [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072
        Invoke-Expression ((New-Object System.Net.WebClient).DownloadString('https://chocolatey.org/install.ps1'))
        $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine")
    }
}

# Установка базовых зависимостей через Chocolatey
function Install-BaseDependencies {
    Write-Green "Установка базовых зависимостей..."
    
    # Проверяем наличие git
    if (-not (Test-Command git)) {
        Write-Yellow "Установка Git..."
        choco install -y git
    }
    
    # Устанавливаем Visual C++ Redistributable
    Write-Yellow "Установка Visual C++ Redistributable..."
    choco install -y vcredist140
    
    # Устанавливаем Visual Studio Build Tools
    Write-Yellow "Установка Visual Studio Build Tools..."
    choco install -y visualstudio2019-workload-vctools
    
    # Обновляем переменные окружения
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine")
}

# Установка Miniconda если его нет
function Install-Miniconda {
    if (-not (Test-Command conda)) {
        Write-Green "Установка Miniconda..."
        $installerPath = "$env:TEMP\Miniconda3.exe"
        $url = "https://repo.anaconda.com/miniconda/Miniconda3-py310_24.1.2-0-Windows-x86_64.exe"
        
        try {
            # Скачивание установщика
            Write-Yellow "Скачивание Miniconda..."
            Invoke-WebRequest -Uri $url -OutFile $installerPath -UseBasicParsing
            
            # Тихая установка
            Write-Yellow "Установка Miniconda..."
            Start-Process -FilePath $installerPath -ArgumentList "/S /RegisterPython=1 /AddToPath=1 /D=C:\Miniconda3" -Wait
            Remove-Item $installerPath
            
            # Добавляем путь к Miniconda в переменные среды
            $condaPath = "C:\Miniconda3"
            $condaScriptsPath = "C:\Miniconda3\Scripts"
            $condaLibPath = "C:\Miniconda3\Library\bin"
            
            # Обновляем Path для текущего процесса
            $env:Path = "$condaPath;$condaScriptsPath;$condaLibPath;" + $env:Path
            
            # Обновляем системный Path
            $systemPath = [System.Environment]::GetEnvironmentVariable("Path", "Machine")
            $systemPath = "$condaPath;$condaScriptsPath;$condaLibPath;" + $systemPath
            [System.Environment]::SetEnvironmentVariable("Path", $systemPath, "Machine")
            
            # Инициализация conda для PowerShell
            Write-Yellow "Инициализация conda..."
            & "$condaScriptsPath\conda.exe" init powershell
            
            # Перезагружаем профиль PowerShell
            if (Test-Path $PROFILE) {
                . $PROFILE
            }
            
            # Проверяем установку
            if (-not (Test-Command conda)) {
                throw "Conda не найдена после установки"
            }
            
        } catch {
            Write-Red "Ошибка при установке Miniconda:"
            Write-Red $_.Exception.Message
            exit 1
        }
    }
}

# Настройка окружения Python
function Setup-Environment {
    Write-Green "Настройка окружения Python..."
    
    try {
        # Проверка существования окружения
        $envExists = & "C:\Miniconda3\Scripts\conda.exe" env list | Select-String "^sexy_bot " -Quiet
        if (-not $envExists) {
            Write-Green "Создание нового окружения..."
            & "C:\Miniconda3\Scripts\conda.exe" create -y -n sexy_bot python=3.10
        } else {
            Write-Yellow "Окружение sexy_bot уже существует, продолжаем..."
        }
        
        # Активация окружения через conda
        Write-Yellow "Активация окружения..."
        $condaPath = "C:\Miniconda3"
        $envPath = Join-Path $condaPath "envs\sexy_bot"
        $envScripts = Join-Path $envPath "Scripts"
        
        # Добавляем пути окружения в текущую сессию
        $env:Path = "$envPath;$envScripts;" + $env:Path
        
        # Установка зависимостей через conda
        Write-Yellow "Установка зависимостей через conda..."
        & "C:\Miniconda3\Scripts\conda.exe" install -y -n sexy_bot -c conda-forge pyaudio portaudio numpy
        
        # Установка остальных зависимостей через pip окружения
        Write-Yellow "Установка Python-пакетов..."
        $pipPath = Join-Path $envScripts "pip.exe"
        & $pipPath install --no-cache-dir google-genai opencv-python pillow mss gTTS pygame edge-tts aiohttp websockets taskgroup exceptiongroup
        
    } catch {
        Write-Red "Ошибка при настройке окружения Python:"
        Write-Red $_.Exception.Message
        exit 1
    }
}

# Основная логика установки
function Main {
    # Проверяем, запущен ли скрипт с правами администратора
    $isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
    if (-not $isAdmin) {
        Write-Red "Этот скрипт требует прав администратора. Пожалуйста, запустите PowerShell от имени администратора."
        exit 1
    }
    
    try {
        # Установка Chocolatey
        Install-Chocolatey
        
        # Установка базовых зависимостей
        Install-BaseDependencies
        
        # Установка Miniconda
        Install-Miniconda
        
        # Настройка окружения Python
        Setup-Environment
        
        Write-Green "Установка завершена!"
        Write-Yellow "Запуск бота..."
        
        # Активация окружения и запуск через полные пути
        $condaPath = "C:\Miniconda3"
        $envPath = Join-Path $condaPath "envs\sexy_bot"
        $pythonPath = Join-Path $envPath "python.exe"
        
        # Запускаем бота используя Python из окружения
        & $pythonPath "sexy_bot.py"
        
    } catch {
        Write-Red "Произошла ошибка при установке:"
        Write-Red $_.Exception.Message
        exit 1
    }
}

# Запуск основной логики
try {
    Main
} catch {
    Write-Red "Произошла критическая ошибка:"
    Write-Red $_.Exception.Message
    Write-Red $_.ScriptStackTrace
    exit 1
} 