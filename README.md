# Интеллектуальная система анализа логов

Тема магистерской работы:

**Интеллектуальная система анализа логов контейнерной ИТ-инфраструктуры и поддержки принятия решений на основе машинного обучения и LLM модели.**

Проект состоит из трёх частей:

1. `ML layer` — классификация строк логов на `info`, `warning`, `alert`, `disaster`.
2. `LLM decision support` — объяснение причины и рекомендации действий через OpenRouter API.
3. `Web UI` — Vue dashboard для анализа логов, просмотра ML-поведения и рекомендаций.

## Архитектура

```text
Vue frontend -> FastAPI backend -> ML model
                            └── OpenRouter LLM API
```

Backend:

- загружает обученную модель `models/log_severity.joblib`;
- принимает пачку логов через `/api/analyze`;
- отдаёт распределение классов, confidence, источник решения `ml` или `rule+ml`;
- формирует рекомендации через OpenRouter;
- если `OPENROUTER_API_KEY` не задан, использует локальный playbook.

Frontend:

- красивая страница dashboard;
- поле для логов;
- карточки `info/warning/alert/disaster`;
- таблица поведения ML;
- блок поддержки принятия решений.

## Классы

- `info`
- `warning`
- `alert`
- `disaster`

Данные берутся из LogHub: https://github.com/logpai/loghub

Важно: LogHub в основном содержит сырые системные логи и разметку аномалий, а не готовую разметку `info/warning/alert/disaster`. Поэтому в этом проекте используется weak supervision: начальные метки строятся эвристиками по уровню логирования и словам-маркерам (`INFO`, `WARN`, `ERROR`, `FATAL`, `panic`, `timeout`, `failed` и т.д.), после чего обучается ML-классификатор.

## Локальная установка без Docker

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Загрузка LogHub samples

```powershell
python log_severity_model.py download
```

По умолчанию будут загружены небольшие файлы `*_2k.log` из официального репозитория LogHub в папку `data/raw`.

## Обучение

```powershell
python log_severity_model.py train --data-dir data/raw --model-out models/log_severity.joblib
```

By default, training also adds 1,200 generated examples for each severity class
(`info`, `warning`, `alert`, `disaster`). This improves coverage for container,
Kubernetes, database, network, storage, and runtime failure patterns.

Optional training flags:

```powershell
python log_severity_model.py train --synthetic-per-label 2000
python log_severity_model.py train --no-synthetic
```

После обучения скрипт выведет качество на отложенной выборке и сохранит модель.

## Предсказание

```powershell
python log_severity_model.py predict --model models/log_severity.joblib --text "ERROR nova.compute.manager Instance failed to spawn: timeout"
```

Можно классифицировать файл:

```powershell
python log_severity_model.py predict --model models/log_severity.joblib --input-file data/raw/OpenStack_2k.log
```

## Запуск backend

```powershell
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

API:

- `GET /api/health`
- `GET /api/dataset`
- `POST /api/classify`
- `POST /api/decision`
- `POST /api/analyze`
- `GET /api/storage`
- `GET /api/history`
- `GET /api/history/{analysis_id}`

`/api/classify` returns the fast ML prediction stage. `/api/decision` returns the
second-stage AI/LLM decision support response and stores the completed analysis.
`/api/analyze` is kept as a combined backwards-compatible endpoint.

## SQLite storage

Backend uses a lightweight SQLite database by default:

```text
data/logops.sqlite3
```

It stores analysis runs, every analyzed log line with ML prediction, and dataset
snapshots from `/api/dataset`.

You can override the database path:

```powershell
$env:DB_PATH="data/logops.sqlite3"
```

## Запуск frontend

```powershell
cd frontend
npm install
npm run dev
```

Открыть: http://localhost:5173

## OpenRouter API

Создайте `.env` по примеру `.env.example` или задайте переменные окружения:

```powershell
$env:OPENROUTER_API_KEY="ваш_ключ"
$env:OPENROUTER_MODEL="openai/gpt-4o-mini"
```

Интеграция использует официальный endpoint OpenRouter:

```text
POST https://openrouter.ai/api/v1/chat/completions
Authorization: Bearer <token>
```

Документация: https://openrouter.ai/docs/api-reference/chat-completion

## Docker Compose

```powershell
copy .env.example .env
docker compose up --build
```

После запуска:

- Frontend: http://localhost:8080
- Backend: http://localhost:8000/api/health

## Для научной работы

Эта реализация подходит как базовый ML-уровень каскадной архитектуры:

1. Log parser / normalizer: очистка timestamp, id, ip, path, hex-значений.
2. ML classifier: быстрая классификация тяжести события.
3. LLM/RAG layer: объяснение причин, агрегация связанных событий и рекомендации.

Для строгого эксперимента weak-labeling нужно заменить или дополнить ручной экспертной разметкой части датасета.
