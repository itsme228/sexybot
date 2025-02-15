# 🔥 SexyBot - AI Companion

![GitHub](https://img.shields.io/github/license/itsme228/sexybot)
![Python](https://img.shields.io/badge/python-3.10-blue)
![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey)

<div align="center">
  *Продвинутый ИИ-компаньон с интеграцией Lovense*
</div>

## 🌟 Особенности

- 🤖 **Искусственный интеллект**
  - Использование передовой модели Google Gemini Flash 2.0
  - Естественное общение с глубоким пониманием контекста
  - Адаптивное поведение под стиль пользователя
  - Долгосрочная память разговоров

- 🎮 **Интеграция с Lovense**
  - Автоматическое обнаружение устройств в локальной сети
  - Интеллектуальная адаптация под возможности каждой игрушки
  - Синхронизация нескольких устройств
  - Анализ обратной связи для оптимизации паттернов
  - Создание уникальных последовательностей в реальном времени

- 🎤 **Голосовое взаимодействие**
  - Двусторонняя голосовая связь с ботом
  - Реалистичный синтез речи через Edge TTS
  - Автоматическое определение пауз в разговоре
  - Высокое качество распознавания речи

- 📸 **Мультимедийные возможности**
  - Поддержка текстового общения
  - Работа с изображениями через камеру
  - Возможность анализа экрана

- 💾 **Технические преимущества**
  - Сохранение и анализ истории общения
  - Кроссплатформенная поддержка (Windows, Linux, macOS, Android)
  - Автоматическое обновление и самообучение

## 🚀 Установка

### Windows
```powershell
# Клонируем репозиторий
git clone https://github.com/itsme228/sexybot.git
cd sexybot

# Запускаем установщик (от имени администратора)
powershell -ExecutionPolicy Bypass -File install_and_run_windows.ps1
```

### Linux/macOS
```bash
# Клонируем репозиторий
git clone https://github.com/itsme228/sexybot.git
cd sexybot

# Запускаем установщик
chmod +x install_and_run_linux.sh
./install_and_run_linux.sh
```

### Android (Termux)
```bash
# Клонируем репозиторий
git clone https://github.com/itsme228/sexybot.git
cd sexybot

# Запускаем установщик
chmod +x install_and_run_termux.sh
./install_and_run_termux.sh
```

## ⚙️ Настройка

1. Получите API ключ от [Google AI Studio](https://makersuite.google.com/app/apikey)
2. При первом запуске установщик попросит ввести ваш API ключ
3. Убедитесь, что Lovense Connect (ссылки ниже) запущен на вашем устройстве и находится в одной сети

## 🎯 Использование

Установщик автоматически создаст окружение и запустит бота. В дальнейшем используйте те же скрипты для запуска:

- Windows: `install_and_run_windows.ps1`
- Linux/macOS: `install_and_run_linux.sh`
- Android: `install_and_run_termux.sh`

При запуске можно указать режим работы:
```bash
--mode none    # Только голос и текст (режим по умолчанию)
--mode camera  # С использованием камеры
--mode screen  # С захватом экрана
```
## 🛠️ Системные требования

### Базовые требования
- 🎤 Микрофон для голосового общения
- 🎧 Наушники или динамики (рекомендуются наушники)
- 🌐 Стабильное подключение к интернету

### Приложение Lovense Connect (запускать на этом же устройстве или на любом другом в одной сети)
Установите приложение для вашей платформы:
- 🪟 [Windows](https://cdn.lovense.com/files/apps/connect/Lovense_Connect.exe)
- 📱 [Android](https://play.google.com/store/apps/details?id=com.lovense.connect)
- 🍎 [iOS](https://apps.apple.com/us/app/lovense-connect/id1273067916)

### Оборудование
- 🔌 Любое совместимое устройство Lovense
- 💻 Компьютер или смартфон с поддержкой Bluetooth

Все остальные зависимости (включая Python) будут установлены автоматически.

## 📝 Лицензия

MIT License - делайте что хотите, главное упомяните автора 😉

## ⚠️ Дисклеймер

Этот проект предназначен только для совершеннолетних пользователей.
Автор не несет ответственности за последствия использования.

## 🤝 Вклад в проект

Присылайте ваши идеи и улучшения через Pull Request!

1. Форкните репозиторий
2. Создайте ветку для фичи (`git checkout -b feature/CoolFeature`)
3. Закоммитьте изменения (`git commit -am 'Add CoolFeature'`)
4. Пусните ветку (`git push origin feature/CoolFeature`)
5. Создайте Pull Request

## 📞 Контакты

- GitHub: [@itsme228](https://github.com/itsme228)
- Telegram: [@itsme228](https://t.me/itsme228)

---
<div align="center">
  <sub>Built with ❤️ by @itsme228</sub>
</div> 