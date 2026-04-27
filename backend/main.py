from __future__ import annotations

import os
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.db import AnalysisStore
from log_severity_model import LABELS, explicit_level_label, load_logs, normalize_log


MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/log_severity.joblib"))
DATA_DIR = Path(os.getenv("DATA_DIR", "data/raw"))
DB_PATH = Path(os.getenv("DB_PATH", "data/logops.sqlite3"))
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

app = FastAPI(
    title="Intelligent Log Analysis API",
    description="ML + LLM decision support for container infrastructure logs.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalyzeRequest(BaseModel):
    logs: str = Field(..., min_length=1)
    use_llm: bool = True


class DecisionRequest(BaseModel):
    logs: str = Field(..., min_length=1)
    use_llm: bool = True


class LogPrediction(BaseModel):
    line_no: int
    message: str
    normalized: str
    severity: str
    source: str
    confidence: float
    probabilities: dict[str, float]


class AnalyzeResponse(BaseModel):
    summary: dict[str, Any]
    predictions: list[LogPrediction]
    decision_support: dict[str, Any]


class ClassifyResponse(BaseModel):
    summary: dict[str, Any]
    predictions: list[LogPrediction]


class DecisionResponse(BaseModel):
    summary: dict[str, Any]
    decision_support: dict[str, Any]


STORE = AnalysisStore(DB_PATH)
STORE.init()


def load_model_payload() -> dict[str, Any]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


MODEL_PAYLOAD = load_model_payload()
MODEL = MODEL_PAYLOAD["model"]


def classify_lines(lines: list[str]) -> list[LogPrediction]:
    normalized = [normalize_log(line) for line in lines]
    model_predictions = MODEL.predict(normalized)
    probabilities = MODEL.predict_proba(normalized)
    classes = MODEL.named_steps["classifier"].classes_

    results: list[LogPrediction] = []
    for index, line in enumerate(lines):
        explicit = explicit_level_label(line)
        severity = explicit or str(model_predictions[index])
        probability_map = {str(label): round(float(score), 4) for label, score in zip(classes, probabilities[index])}
        confidence = 0.99 if explicit else max(probability_map.values())
        results.append(
            LogPrediction(
                line_no=index + 1,
                message=line,
                normalized=normalized[index],
                severity=severity,
                source="rule+ml" if explicit else "ml",
                confidence=round(float(confidence), 4),
                probabilities=probability_map,
            )
        )
    return results


def parse_log_lines(logs: str) -> list[str]:
    lines = [line.strip() for line in logs.splitlines() if line.strip()]
    if not lines:
        raise HTTPException(status_code=400, detail="At least one non-empty log line is required.")
    return lines


def build_summary(predictions: list[LogPrediction]) -> tuple[dict[str, Any], Counter[str]]:
    counts = Counter(item.severity for item in predictions)
    max_severity = max(counts, key=lambda label: LABELS.index(label) if label in LABELS else 0)
    return (
        {
            "total": len(predictions),
            "distribution": {label: counts.get(label, 0) for label in LABELS},
            "max_severity": max_severity,
            "ml_sources": MODEL_PAYLOAD.get("training_sources", []),
        },
        counts,
    )


def save_completed_analysis(
    *,
    predictions: list[LogPrediction],
    summary: dict[str, Any],
    decision_support: dict[str, Any],
    use_llm: bool,
) -> int:
    return STORE.save_analysis(
        total=summary["total"],
        max_severity=summary["max_severity"],
        distribution=summary["distribution"],
        decision_support=decision_support,
        use_llm=use_llm,
        predictions=predictions,
    )


def local_decision_support(predictions: list[LogPrediction]) -> dict[str, Any]:
    counts = Counter(item.severity for item in predictions)
    worst = max(predictions, key=lambda item: LABELS.index(item.severity) if item.severity in LABELS else 0)

    playbooks = {
        "info": [
            "Зафиксировать событие в журнале наблюдения.",
            "Проверить, повторяется ли сообщение в течение последних 10-15 минут.",
        ],
        "warning": [
            "Проверить метрики контейнера: CPU, RAM, disk I/O, network latency.",
            "Сравнить событие с последним deploy, restart или изменением конфигурации.",
            "Настроить наблюдение: предупреждение ещё не авария, но может быть ранним сигналом деградации.",
        ],
        "alert": [
            "Проверить affected service, pod/container id, node и dependency, связанные с логом.",
            "Посмотреть соседние ERROR/WARNING события за окно ±5 минут.",
            "Если проблема повторяется, выполнить rollback последнего изменения или перезапуск конкретного контейнера.",
        ],
        "disaster": [
            "Запустить incident response: назначить ответственного и зафиксировать время начала инцидента.",
            "Изолировать повреждённый контейнер или node, чтобы не распространять отказ.",
            "Проверить backup, persistent volumes, kernel/runtime errors и доступность критичных сервисов.",
            "После стабилизации провести RCA: причина, blast radius, preventive action.",
        ],
    }

    severity_order = ["info", "warning", "alert", "disaster"]
    max_severity = max(counts, key=lambda label: severity_order.index(label) if label in severity_order else 0)
    distribution_text = ", ".join(f"{label}: {counts.get(label, 0)}" for label in LABELS)
    likely_reason = {
        "info": "Критичных признаков не найдено. Логи похожи на штатные системные события.",
        "warning": "Есть признаки деградации или приближения к порогам, но прямого отказа пока нет.",
        "alert": "Есть признаки ошибки сервиса или зависимости: отказ подключения, timeout, exception или rejected request.",
        "disaster": "Есть признаки критического отказа: panic, fatal, corruption, data loss, OOM или недоступность ключевого компонента.",
    }[max_severity]
    analysis = (
        f"ИИ-анализ на основе ML-классификации\n\n"
        f"Уровень риска: {max_severity.upper()}.\n"
        f"Распределение событий: {distribution_text}.\n\n"
        f"Главная гипотеза: {likely_reason}\n"
        f"Наиболее важная строка: #{worst.line_no} [{worst.severity}] {worst.message}\n\n"
        "Рекомендуемые действия:\n"
        + "\n".join(f"{index}. {action}" for index, action in enumerate(playbooks[max_severity], start=1))
    )

    return {
        "provider": "local-ml-analysis",
        "risk_level": max_severity,
        "analysis": analysis,
        "main_hypothesis": f"Наиболее критичное событие классифицировано как {worst.severity}: {worst.message}",
        "recommended_actions": playbooks[max_severity],
        "escalation": max_severity in {"alert", "disaster"},
        "note": "OpenRouter API key не задан или LLM отключена, поэтому использован локальный ML-анализ.",
    }


def openrouter_decision_support(logs: str, predictions: list[LogPrediction]) -> dict[str, Any]:
    api_key = os.getenv("OPENROUTER_API_KEY")
    model = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
    if not api_key:
        return local_decision_support(predictions)

    compact = [
        {
            "line_no": item.line_no,
            "severity": item.severity,
            "confidence": item.confidence,
            "message": item.message[:500],
        }
        for item in predictions[:40]
    ]

    system_prompt = (
        "Ты эксперт AIOps/SRE для контейнерной ИТ-инфраструктуры. "
        "Объясняй кратко, технически и по-русски. "
        "Дай гипотезу причины, приоритет, конкретные действия, команды проверки и условия эскалации. "
        "Не выдумывай факты, которых нет в логах. "
        "Пиши ответ только обычным чистым текстом без Markdown-форматирования. "
        "Не используй заголовки Markdown, жирный текст, маркированные списки, таблицы, блоки кода, обратные кавычки и символы разметки. "
        "Если нужны шаги, пиши обычными строками с нумерацией вида 1. 2. 3."
    )
    user_prompt = (
        "Проанализируй результаты ML-классификации логов и предложи действия.\n\n"
        f"Классификация:\n{compact}\n\n"
        f"Сырые логи:\n{logs[:8000]}"
    )

    try:
        response = requests.post(
            OPENROUTER_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": os.getenv("APP_SITE_URL", "http://localhost:8080"),
                "X-Title": os.getenv("APP_SITE_NAME", "LogOps Decision Support"),
            },
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.2,
                "max_tokens": 900,
            },
            timeout=45,
        )
        response.raise_for_status()
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        return {
            "provider": "openrouter",
            "model": data.get("model", model),
            "risk_level": Counter(item.severity for item in predictions).most_common(1)[0][0],
            "analysis": content,
            "usage": data.get("usage"),
        }
    except Exception as exc:
        fallback = local_decision_support(predictions)
        fallback["provider"] = "local-playbook-after-openrouter-error"
        fallback["openrouter_error"] = str(exc)
        return fallback


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "model_path": str(MODEL_PATH),
        "model_exists": MODEL_PATH.exists(),
        "db_path": str(DB_PATH),
        "db_exists": DB_PATH.exists(),
        "openrouter_enabled": bool(os.getenv("OPENROUTER_API_KEY")),
    }


@app.get("/api/dataset")
def dataset() -> dict[str, Any]:
    frame = load_logs(DATA_DIR)
    payload = {
        "rows": len(frame),
        "sources": sorted(frame["source"].unique().tolist()),
        "class_distribution": dict(Counter(frame["label"])),
        "samples": frame[["source", "line_no", "raw", "label"]].head(12).to_dict(orient="records"),
    }
    snapshot_id = STORE.save_dataset_snapshot(payload)
    payload["snapshot_id"] = snapshot_id
    return payload


@app.get("/api/storage")
def storage() -> dict[str, Any]:
    return STORE.stats()


@app.get("/api/history")
def history(limit: int = 20) -> dict[str, Any]:
    return {"items": STORE.list_history(limit=limit)}


@app.get("/api/history/{analysis_id}")
def history_item(analysis_id: int) -> dict[str, Any]:
    item = STORE.get_analysis(analysis_id)
    if item is None:
        raise HTTPException(status_code=404, detail="Analysis not found.")
    return item


@app.post("/api/classify", response_model=ClassifyResponse)
def classify(request: AnalyzeRequest) -> ClassifyResponse:
    lines = parse_log_lines(request.logs)
    predictions = classify_lines(lines)
    summary, _ = build_summary(predictions)
    return ClassifyResponse(summary=summary, predictions=predictions)


@app.post("/api/decision", response_model=DecisionResponse)
def decision(request: DecisionRequest) -> DecisionResponse:
    lines = parse_log_lines(request.logs)
    predictions = classify_lines(lines)
    summary, _ = build_summary(predictions)
    decision_support = (
        openrouter_decision_support(request.logs, predictions) if request.use_llm else local_decision_support(predictions)
    )
    analysis_id = save_completed_analysis(
        predictions=predictions,
        summary=summary,
        decision_support=decision_support,
        use_llm=request.use_llm,
    )
    summary["analysis_id"] = analysis_id
    return DecisionResponse(summary=summary, decision_support=decision_support)


@app.post("/api/analyze", response_model=AnalyzeResponse)
def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    lines = parse_log_lines(request.logs)
    predictions = classify_lines(lines)
    summary, _ = build_summary(predictions)
    decision_support = (
        openrouter_decision_support(request.logs, predictions) if request.use_llm else local_decision_support(predictions)
    )
    analysis_id = save_completed_analysis(
        predictions=predictions,
        summary=summary,
        decision_support=decision_support,
        use_llm=request.use_llm,
    )
    summary["analysis_id"] = analysis_id

    return AnalyzeResponse(
        summary=summary,
        predictions=predictions,
        decision_support=decision_support,
    )
