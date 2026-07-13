## Запуск в браузере

Браузерный режим предназначен для разработки и тестирования интерфейса. Vite запускает UI и подключает Python-аудиодвижок через встроенный bridge.

### Требования

- Windows 10 или Windows 11;
- Python;
- Node.js и npm;
- установленный VB-Cable для полноценной работы с системным звуком.

### 1. Клонирование репозитория

```powershell
git clone https://github.com/Parcart/EarLoop.git
cd EarLoop
```

### 2. Подготовка Python-окружения

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements\requirements.txt
```

### 3. Установка зависимостей интерфейса

```powershell
cd ui
npm install
```

### 4. Запуск

```powershell
npm run dev
```