from __future__ import annotations

import argparse
import itertools
import re
import sys
import urllib.request
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline


LABELS = ("info", "warning", "alert", "disaster")
DEFAULT_SYNTHETIC_PER_LABEL = 1_200

LOGHUB_SAMPLE_URLS = {
    "Apache_2k.log": "https://raw.githubusercontent.com/logpai/loghub/master/Apache/Apache_2k.log",
    "BGL_2k.log": "https://raw.githubusercontent.com/logpai/loghub/master/BGL/BGL_2k.log",
    "HDFS_2k.log": "https://raw.githubusercontent.com/logpai/loghub/master/HDFS/HDFS_2k.log",
    "Linux_2k.log": "https://raw.githubusercontent.com/logpai/loghub/master/Linux/Linux_2k.log",
    "OpenStack_2k.log": "https://raw.githubusercontent.com/logpai/loghub/master/OpenStack/OpenStack_2k.log",
    "Zookeeper_2k.log": "https://raw.githubusercontent.com/logpai/loghub/master/Zookeeper/Zookeeper_2k.log",
}

NORMALIZE_PATTERNS = [
    (re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b"), " <IP> "),
    (re.compile(r"\b0x[0-9a-fA-F]+\b"), " <HEX> "),
    (re.compile(r"\b[0-9a-fA-F]{8,}\b"), " <ID> "),
    (re.compile(r"\b\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?Z?\b"), " <TIME> "),
    (re.compile(r"\b\d{2}:\d{2}:\d{2}(?:[.,]\d+)?\b"), " <TIME> "),
    (re.compile(r"\b\d+\b"), " <NUM> "),
    (re.compile(r"/(?:[\w.-]+/)+[\w.-]*"), " <PATH> "),
    (re.compile(r"\s+"), " "),
]

DISASTER_RE = re.compile(
    r"\b("
    r"fatal|panic|emerg(?:ency)?|disaster|critical|crit|severe|segfault|"
    r"kernel panic|corrupt(?:ed|ion)?|unrecoverable|abort(?:ed)?|crash(?:ed)?|"
    r"out of memory|oom|system down|data loss"
    r")\b",
    re.IGNORECASE,
)

ALERT_RE = re.compile(
    r"\b("
    r"error|err|exception|fail(?:ed|ure)?|alert|anomal(?:y|ous)|invalid|denied|"
    r"timeout|unable|lost|reject(?:ed)?|refused|unavailable|disconnect(?:ed)?|"
    r"traceback|stacktrace|violation"
    r")\b",
    re.IGNORECASE,
)

WARNING_RE = re.compile(
    r"\b("
    r"warn(?:ing)?|retry(?:ing)?|slow|deprecated|blocked|delay(?:ed)?|"
    r"threshold|overload(?:ed)?|unstable|skipped|missing"
    r")\b",
    re.IGNORECASE,
)

INFO_RE = re.compile(
    r"\b("
    r"info|notice|debug|trace|start(?:ed|ing)?|stop(?:ped|ping)?|complete(?:d)?|"
    r"success(?:ful)?|created|deleted|connected|accepted|received|sent|running"
    r")\b",
    re.IGNORECASE,
)

EXPLICIT_DISASTER_RE = re.compile(r"\b(fatal|panic|emerg(?:ency)?|critical|crit|severe)\b", re.IGNORECASE)
EXPLICIT_ALERT_RE = re.compile(r"\b(error|err|exception|failed|failure|alert)\b", re.IGNORECASE)
EXPLICIT_WARNING_RE = re.compile(r"\b(warn|warning)\b", re.IGNORECASE)
EXPLICIT_INFO_RE = re.compile(r"\b(info|notice|debug|trace)\b", re.IGNORECASE)

SYNTHETIC_COMPONENTS = {
    "services": [
        "api-gateway",
        "auth-service",
        "billing-worker",
        "cache",
        "checkout",
        "container-runtime",
        "database",
        "event-consumer",
        "ingress-controller",
        "inventory-service",
        "job-runner",
        "kubelet",
        "message-broker",
        "metrics-agent",
        "notification-service",
        "order-service",
        "payment-service",
        "scheduler",
        "storage-driver",
        "user-service",
    ],
    "nodes": [
        "node-a1",
        "node-b2",
        "node-c3",
        "worker-01",
        "worker-02",
        "worker-03",
        "zone-east-1",
        "zone-west-2",
    ],
    "containers": [
        "pod/api-7f8d9",
        "pod/auth-56c2",
        "pod/db-0",
        "container/checkout",
        "container/payment",
        "container/worker",
        "task/reconcile",
        "job/nightly-sync",
    ],
    "numbers": ["1", "2", "3", "5", "8", "13", "21", "34", "55", "89"],
    "paths": [
        "/var/log/app/current.log",
        "/etc/service/config.yaml",
        "/data/volume/orders",
        "/mnt/pv/primary",
        "/usr/local/bin/healthcheck",
    ],
}

SYNTHETIC_TEMPLATES = {
    "info": [
        "INFO {service} started on {node} with {num} workers",
        "INFO {service} completed request batch in {num} ms",
        "DEBUG {service} heartbeat accepted from {container}",
        "NOTICE {service} connected to upstream {container}",
        "INFO {service} config loaded from {path}",
        "TRACE {service} sent metrics sample for {container}",
        "INFO {container} ready after successful health check",
        "INFO {service} received graceful shutdown signal on {node}",
        "INFO {service} created session cache entry ttl={num}s",
        "DEBUG {service} polling loop running normally on {node}",
        "INFO {service} deployment rollout completed successfully",
        "NOTICE {service} accepted connection from 10.42.{num}.15",
        "{service} routine health check passed for {container}",
        "{container} processed scheduled maintenance task successfully",
        "{service} background sync completed for shard-{num}",
        "{service} opened keepalive session on {node}",
    ],
    "warning": [
        "WARNING {service} response time exceeded soft threshold {num} ms",
        "WARN {container} retrying request after transient network delay",
        "WARNING {service} memory usage above {num}% on {node}",
        "WARN {service} deprecated config key detected in {path}",
        "WARNING {service} disk usage approaching threshold on {node}",
        "WARN {container} skipped optional cleanup because lock is busy",
        "WARNING {service} queue depth increased to {num} pending messages",
        "WARN {service} probe unstable, next retry scheduled in {num}s",
        "WARNING {service} connection pool nearly exhausted",
        "WARN {container} delayed startup, dependency not ready yet",
        "WARNING {service} observed elevated latency from {container}",
        "WARN {service} rate limit threshold reached for tenant-{num}",
        "{service} latency is increasing but requests are still served",
        "{container} retry scheduled after temporary dependency slowdown",
        "{service} queue depth near operating threshold on {node}",
        "{service} cache hit ratio dropped below expected baseline",
    ],
    "alert": [
        "ERROR {service} failed to connect to database from {node}",
        "ERROR {container} request timeout after {num} seconds",
        "ALERT {service} rejected message because signature is invalid",
        "ERROR {service} permission denied while reading {path}",
        "ERROR {container} failed liveness probe three times",
        "ALERT {service} anomaly detected in traffic from 10.42.{num}.77",
        "ERROR {service} dependency unavailable during transaction commit",
        "ERROR {container} lost connection to message broker",
        "ALERT {service} repeated authentication failures for user-{num}",
        "ERROR {service} exception while processing order id {num}",
        "ERROR {container} refused connection from upstream gateway",
        "ALERT {service} policy violation detected in request payload",
        "{service} timeout while committing transaction for tenant-{num}",
        "{container} lost database connection during request processing",
        "{service} dependency unavailable and request was rejected",
        "{service} invalid token caused authentication failure",
    ],
    "disaster": [
        "FATAL {service} kernel panic reported by {node}",
        "CRITICAL {container} unrecoverable storage corruption on {path}",
        "EMERGENCY {service} system down after control-plane failure",
        "FATAL {service} out of memory killed primary process on {node}",
        "CRITICAL {container} data loss detected in persistent volume",
        "PANIC {service} segmentation fault crashed process pid {num}",
        "SEVERE {service} quorum lost, cluster cannot accept writes",
        "FATAL {container} aborted after repeated crash loop on {node}",
        "CRITICAL {service} backup restore failed and database is unavailable",
        "EMERG {service} node filesystem read-only after disk failure",
        "FATAL {service} unrecoverable transaction log corruption detected",
        "CRITICAL {container} container runtime crashed across node pool",
        "{service} primary database corrupt and cannot recover transaction log",
        "{container} crash loop exhausted and service is unavailable cluster-wide",
        "{service} quorum lost with possible data loss on {path}",
        "{node} reports kernel panic after storage driver corruption",
    ],
}


def normalize_log(line: str) -> str:
    text = line.strip()
    for pattern, replacement in NORMALIZE_PATTERNS:
        text = pattern.sub(replacement, text)
    return text.strip().lower()


def weak_label(line: str) -> str:
    text = line.strip()

    if DISASTER_RE.search(text):
        return "disaster"
    if ALERT_RE.search(text):
        return "alert"
    if WARNING_RE.search(text):
        return "warning"
    if INFO_RE.search(text):
        return "info"

    # BGL logs often start with "-" for normal events and a label for abnormal events.
    first_token = text.split(maxsplit=1)[0] if text else ""
    if first_token and first_token != "-" and re.fullmatch(r"[A-Za-z0-9_.:-]+", first_token):
        return "alert"

    return "info"


def explicit_level_label(line: str) -> str | None:
    if EXPLICIT_DISASTER_RE.search(line):
        return "disaster"
    if EXPLICIT_ALERT_RE.search(line):
        return "alert"
    if EXPLICIT_WARNING_RE.search(line):
        return "warning"
    if EXPLICIT_INFO_RE.search(line):
        return "info"
    return None


def generate_synthetic_logs(per_label: int = DEFAULT_SYNTHETIC_PER_LABEL) -> pd.DataFrame:
    rows = []
    components = SYNTHETIC_COMPONENTS

    for label, templates in SYNTHETIC_TEMPLATES.items():
        values = itertools.product(
            components["services"],
            components["nodes"],
            components["containers"],
            components["numbers"],
            components["paths"],
        )
        for index, (service, node, container, num, path) in enumerate(values, start=1):
            if index > per_label:
                break
            template = templates[(index - 1) % len(templates)]
            raw = template.format(service=service, node=node, container=container, num=num, path=path)
            rows.append(
                {
                    "source": "synthetic_training_examples",
                    "line_no": index,
                    "raw": raw,
                    "text": normalize_log(raw),
                    "label": label,
                }
            )

    return pd.DataFrame(rows)


def load_logs(data_dir: Path, include_synthetic: bool = True, synthetic_per_label: int = DEFAULT_SYNTHETIC_PER_LABEL) -> pd.DataFrame:
    rows = []
    for path in sorted(data_dir.glob("*.log")):
        with path.open("r", encoding="utf-8", errors="ignore") as file:
            for line_no, line in enumerate(file, start=1):
                raw = line.strip()
                if not raw:
                    continue
                rows.append(
                    {
                        "source": path.name,
                        "line_no": line_no,
                        "raw": raw,
                        "text": normalize_log(raw),
                        "label": weak_label(raw),
                    }
                )

    if not rows:
        raise FileNotFoundError(f"No .log files found in {data_dir}")

    frame = pd.DataFrame(rows)
    if include_synthetic:
        frame = pd.concat([frame, generate_synthetic_logs(synthetic_per_label)], ignore_index=True)
    return frame


def build_pipeline() -> Pipeline:
    features = FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=80_000,
                    sublinear_tf=True,
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=2,
                    max_features=80_000,
                    sublinear_tf=True,
                ),
            ),
        ]
    )
    return Pipeline(
        [
            ("features", features),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1_000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )


def download_samples(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, url in LOGHUB_SAMPLE_URLS.items():
        target = output_dir / filename
        if target.exists() and target.stat().st_size > 0:
            print(f"exists: {target}")
            continue

        print(f"download: {url}")
        with urllib.request.urlopen(url, timeout=60) as response:
            content = response.read()
        target.write_bytes(content)
        print(f"saved: {target} ({len(content)} bytes)")


def train(data_dir: Path, model_out: Path, include_synthetic: bool, synthetic_per_label: int) -> None:
    df = load_logs(data_dir, include_synthetic=include_synthetic, synthetic_per_label=synthetic_per_label)
    counts = Counter(df["label"])
    print("Dataset size:", len(df))
    print("Synthetic examples:", synthetic_per_label * len(LABELS) if include_synthetic else 0)
    print("Class distribution:", dict(counts))

    missing = [label for label in LABELS if counts[label] < 2]
    if missing:
        raise ValueError(
            "Not enough examples for classes: "
            + ", ".join(missing)
            + ". Add more logs or adjust weak-labeling rules."
        )

    x_train, x_test, y_train, y_test = train_test_split(
        df["text"],
        df["label"],
        test_size=0.2,
        random_state=42,
        stratify=df["label"],
    )

    model = build_pipeline()
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    print("\nClassification report:")
    print(classification_report(y_test, predictions, labels=list(LABELS), zero_division=0))
    print("Confusion matrix:")
    print(pd.DataFrame(confusion_matrix(y_test, predictions, labels=list(LABELS)), index=LABELS, columns=LABELS))

    model_out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model,
        "labels": LABELS,
        "normalizer": "normalize_log_v1",
        "training_sources": sorted(df["source"].unique().tolist()),
        "class_distribution": dict(counts),
    }
    joblib.dump(payload, model_out)
    print(f"\nModel saved: {model_out}")


def predict(model_path: Path, text: str | None, input_file: Path | None) -> None:
    payload = joblib.load(model_path)
    model = payload["model"]

    if text:
        lines = [text]
    elif input_file:
        with input_file.open("r", encoding="utf-8", errors="ignore") as file:
            lines = [line.strip() for line in file if line.strip()]
    else:
        lines = [line.strip() for line in sys.stdin if line.strip()]

    normalized = [normalize_log(line) for line in lines]
    predicted = model.predict(normalized)
    explicit_labels = [explicit_level_label(line) for line in lines]
    final_labels = [explicit or model_label for explicit, model_label in zip(explicit_labels, predicted)]

    probabilities = None
    if hasattr(model.named_steps["classifier"], "predict_proba"):
        probabilities = model.predict_proba(normalized)
        classes = model.named_steps["classifier"].classes_
    else:
        classes = np.array(payload["labels"])

    for index, (line, label) in enumerate(zip(lines, final_labels)):
        if probabilities is None:
            print(f"{label}\t{line}")
            continue

        model_confidence = float(np.max(probabilities[index]))
        top = sorted(zip(classes, probabilities[index]), key=lambda item: item[1], reverse=True)[:2]
        top_text = ", ".join(f"{name}:{score:.3f}" for name, score in top)
        source = "rule+ml" if explicit_labels[index] else "ml"
        print(f"{label}\tsource={source}\tml_confidence={model_confidence:.3f}\t{top_text}\t{line}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and use a LogHub-based log severity classifier.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    download_parser = subparsers.add_parser("download", help="Download small LogHub sample logs.")
    download_parser.add_argument("--output-dir", type=Path, default=Path("data/raw"))

    train_parser = subparsers.add_parser("train", help="Train the severity classifier.")
    train_parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    train_parser.add_argument("--model-out", type=Path, default=Path("models/log_severity.joblib"))
    train_parser.add_argument("--no-synthetic", action="store_true", help="Train only on raw LogHub files.")
    train_parser.add_argument(
        "--synthetic-per-label",
        type=int,
        default=DEFAULT_SYNTHETIC_PER_LABEL,
        help="Number of generated training examples per severity label.",
    )

    predict_parser = subparsers.add_parser("predict", help="Classify one log line, a file, or stdin.")
    predict_parser.add_argument("--model", type=Path, default=Path("models/log_severity.joblib"))
    predict_parser.add_argument("--text")
    predict_parser.add_argument("--input-file", type=Path)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "download":
        download_samples(args.output_dir)
    elif args.command == "train":
        train(
            args.data_dir,
            args.model_out,
            include_synthetic=not args.no_synthetic,
            synthetic_per_label=args.synthetic_per_label,
        )
    elif args.command == "predict":
        predict(args.model, args.text, args.input_file)
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
