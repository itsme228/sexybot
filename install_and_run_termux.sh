
#!/data/data/com.termux/files/usr/bin/bash

# Цвета для вывода
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}=== Установка и настройка Sexy Bot для Termux ===${NC}"

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
    echo -e "${GREEN}Обновление репозиториев и установка базовых зависимостей...${NC}"
    
    # Обновление и установка базовых пакетов
    pkg update -y && pkg upgrade -y
    
    # Установка необходимых пакетов
    pkg install -y python clang python-pip git wget curl pulseaudio 
    
    # Установка дополнительных зависимостей для сборки
    pkg install -y libjpeg-turbo libpng make 
    
    # Запуск PulseAudio сервера для работы со звуком
    if ! pgrep -x "pulseaudio" >/dev/null; then
        pulseaudio --start --load="module-native-protocol-tcp" --exit-idle-time=-1
        echo -e "${GREEN}PulseAudio сервер запущен${NC}"
    fi
}

# Настройка Python-окружения
setup_environment() {
    echo -e "${GREEN}Настройка Python-окружения...${NC}"
    
    # Создание виртуального окружения
    if [ ! -d "venv" ]; then
        python -m venv venv
    fi
    
    # Активация виртуального окружения
    source venv/bin/activate
    
    # Обновление pip
    pip install --upgrade pip
    
    # Установка зависимостей
    LDFLAGS="-L/system/lib64/" CFLAGS="-I/data/data/com.termux/files/usr/include/" pip install pyaudio
    pip install google-genai pillow mss gTTS pygame edge-tts numpy aiohttp websockets taskgroup exceptiongroup
    
    # OpenCV для Termux требует специальной установки
    pkg install -y opencv-python
}

# Настройка разрешений
setup_permissions() {
    echo -e "${GREEN}Настройка разрешений...${NC}"
    
    # Запрос разрешений для Termux
    termux-setup-storage
    
    # Разрешения на выполнение скрипта
    chmod +x sexy_bot.py
}

# Основная логика установки
main() {
    # Проверка и установка базовых зависимостей
    install_base_deps
    
    # Настройка Python-окружения
    setup_environment
    
    # Настройка разрешений
    setup_permissions
    
    echo -e "${GREEN}Установка завершена!${NC}"
    echo -e "${YELLOW}Для запуска бота используйте следующие команды:${NC}"
    echo -e "${GREEN}source venv/bin/activate${NC}"
    echo -e "${GREEN}python sexy_bot.py${NC}"
    
    # Активация окружения и запуск
    source venv/bin/activate
    python sexy_bot.py
}

# Запуск основной логики
main

# Инструкции по использованию
echo -e "\n${YELLOW}=== Важные замечания ===${NC}"
echo -e "1. Убедитесь, что вы предоставили Termux необходимые разрешения"
echo -e "2. Для работы с микрофоном установите Termux:API из F-Droid"
echo -e "3. После установки Termux:API выполните:"
echo -e "${GREEN}pkg install termux-api${NC}"
echo -e "4. Для перезапуска бота используйте:"
echo -e "${GREEN}cd путь/к/папке/с/ботом${NC}"
echo -e "${GREEN}source venv/bin/activate${NC}"
echo -e "${GREEN}python sexy_bot.py${NC}"