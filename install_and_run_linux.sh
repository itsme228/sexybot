#!/bin/bash


# Tested on Kali Linux 2024.4

# Цвета для вывода
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}=== Установка и настройка Sexy Bot ===${NC}"

# Проверка наличия API ключа в .env файле
ENV_FILE=".env"
if [ -f "$ENV_FILE" ]; then
    export $(cat "$ENV_FILE" | xargs)
fi

if [ -z "$GOOGLE_API_KEY" ]; then
    echo -e "${YELLOW}Введите ваш Google API ключ:${NC}"
    read api_key
    echo "GOOGLE_API_KEY=$api_key" > "$ENV_FILE"
    export GOOGLE_API_KEY=$api_key
fi

# Функция для проверки наличия команды
check_command() {
    command -v $1 >/dev/null 2>&1
}

# Установка базовых зависимостей
install_base_deps() {
    echo -e "${GREEN}Установка базовых зависимостей...${NC}"
    if check_command apt; then
        sudo apt update
        # Добавляем обновление libstdc++ и JACK
        sudo apt install -y wget curl git build-essential python3-dev portaudio19-dev python3-pyaudio libstdc++6 jackd2 libjack-jackd2-0 libjack-jackd2-dev
        # Обновляем все пакеты системы для согласованности версий
        sudo apt upgrade -y
    elif check_command pacman; then
        sudo pacman -Sy --noconfirm wget curl git base-devel portaudio python-pyaudio jack2
    elif check_command dnf; then
        sudo dnf install -y wget curl git gcc gcc-c++ make portaudio-devel python3-pyaudio jack-audio-connection-kit
    fi
}

# Установка Miniconda если нет
install_miniconda() {
    if ! check_command conda; then
        echo -e "${GREEN}Установка Miniconda...${NC}"
        wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
        bash miniconda.sh -b -p $HOME/miniconda
        rm miniconda.sh
        export PATH="$HOME/miniconda/bin:$PATH"
        echo 'export PATH="$HOME/miniconda/bin:$PATH"' >> ~/.bashrc
        source ~/.bashrc
        conda init bash
        source ~/.bashrc
    fi
}

# Создание и активация окружения
setup_environment() {
    echo -e "${GREEN}Настройка окружения Python...${NC}"
    
    # Проверяем существование окружения
    if ! conda env list | grep -q "^sexy_bot "; then
        echo -e "${GREEN}Создание нового окружения...${NC}"
        conda create -y -n sexy_bot python=3.10
    else
        echo -e "${YELLOW}Окружение sexy_bot уже существует, пропускаем создание...${NC}"
    fi
    
    source activate sexy_bot
    
    # Устанавливаем PyAudio через conda для избежания проблем с компиляцией
    conda install -y -c conda-forge portaudio pyaudio
    
    # Установка остальных зависимостей через pip
    pip install google-genai opencv-python pillow mss gTTS pygame edge-tts numpy aiohttp websockets taskgroup exceptiongroup
}

# Основная логика установки
main() {
    # Проверка и установка базовых зависимостей
    install_base_deps
    
    # Установка Miniconda
    install_miniconda
    
    # Настройка окружения Python
    setup_environment
    
    # Даем права на выполнение Python-скрипту
    chmod +x sexy_bot.py
    
    echo -e "${GREEN}Установка завершена!${NC}"
    echo -e "${YELLOW}Запуск бота...${NC}"
    
    # Активация окружения и запуск
    source activate sexy_bot
    python sexy_bot.py
}

# Запуск основной логики
main 