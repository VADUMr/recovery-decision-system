import csv
import json
import os
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from flask import Flask, jsonify, request

app = Flask(__name__)

APP_URL = os.getenv("APP_URL", "http://app-service:8000")
PROMETHEUS_URL = os.getenv("PROMETHEUS_URL", "http://prometheus:9090")
DATA_FILE = Path(os.getenv("DATA_FILE", "/data/pilot_results.csv"))

with open(Path(__file__).with_name("reference_policy.json")) as f:
    REFERENCE_POLICY = json.load(f)


def get_reference_action(scenario_key, severity):
    policy = REFERENCE_POLICY["scenarios"][scenario_key]
    if policy.get("rule_type") == "binary_failure":
        return policy["failure_action"] if severity >= policy.get("failure_value", 1.0) else policy["healthy_action"]
    
    if severity < policy["threshold"]:
        return policy["below_threshold_action"]
    return policy["at_or_above_threshold_action"]


def prometheus_query(query):
    try:
        r = requests.get(f"{PROMETHEUS_URL}/api/v1/query", params={"query": query}, timeout=10)
        r.raise_for_status()
        result = r.json()["data"]["result"]
        return float(result[0]["value"][1]) if result else 0.0
    except Exception:
        return 0.0


def collect_observation():
    return {
        "cpu": prometheus_query("ecdm_cpu_usage_percent"),
        "memory": prometheus_query("ecdm_memory_usage_percent"),
        "allocated_memory_mb": prometheus_query("ecdm_memory_allocated_mb"),
        "error_rate": prometheus_query("ecdm_error_rate_percent"),
        "dependency_available": prometheus_query("ecdm_dependency_available"),
        "latency_ms": prometheus_query(
            "rate(ecdm_request_latency_seconds_sum[1m]) / "
            "clamp_min(rate(ecdm_request_latency_seconds_count[1m]), 0.001)"
        ) * 1000,
    }


def apply_fault(fault_type, severity):
    requests.post(f"{APP_URL}/fault", json={"type": fault_type, "severity": severity}, timeout=10)


def generate_load(duration_seconds=4, requests_per_second=5):
    latencies, ok, fail = [], 0, 0
    started = time.time()
    while time.time() - started < duration_seconds:
        for _ in range(requests_per_second):
            t0 = time.perf_counter()
            try:
                r = requests.get(f"{APP_URL}/work", timeout=3)
                ok += 1 if r.status_code < 400 else 0
                fail += 0 if r.status_code < 400 else 1
            except requests.RequestException:
                fail += 1
            latencies.append((time.perf_counter() - t0) * 1000)
        time.sleep(max(0, 1 - (time.time() - started) % 1))
    total = ok + fail
    return {
        "requests": total,
        "successful_requests": ok,
        "median_client_latency_ms": statistics.median(latencies) if latencies else 0.0,
        "availability": ok / total if total else 0.0,
    }


CSV_FIELDS = [
    "timestamp", "scenario", "severity", "expected_decision",
    "cpu", "memory", "allocated_memory_mb", "latency_ms", "error_rate", "dependency_available",
    "requests", "successful_requests", "median_client_latency_ms",
    "pre_availability", "post_availability",
]


def save_result(row):
    DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    file_exists = DATA_FILE.exists()
    with DATA_FILE.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def execute_scenario_run(scenario_key, severity):
    apply_fault("clear", 0)
    time.sleep(1.5)
    apply_fault(scenario_key, severity)
    time.sleep(1.5)  # даємо час prometheus підхопити метрики

    pre_load = generate_load()
    observation = collect_observation()
    post_load = generate_load()

    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "scenario": scenario_key,
        "severity": severity,
        "expected_decision": get_reference_action(scenario_key, severity),
        **observation,
        **{k: pre_load[k] for k in ("requests", "successful_requests", "median_client_latency_ms")},
        "pre_availability": pre_load["availability"],
        "post_availability": post_load["availability"],
    }
    save_result(row)
    return row


@app.post("/run/cpu")
def run_cpu():
    severity = float((request.get_json(silent=True) or {}).get("severity", 0.9))
    return jsonify(execute_scenario_run("cpu", severity))


@app.post("/run/memory")
def run_memory():
    severity = float((request.get_json(silent=True) or {}).get("severity", 250.0))
    return jsonify(execute_scenario_run("memory", severity))


@app.post("/run/latency")
def run_latency():
    severity = float((request.get_json(silent=True) or {}).get("severity", 800.0))
    return jsonify(execute_scenario_run("latency", severity))


@app.post("/run/errors")
def run_errors():
    severity = float((request.get_json(silent=True) or {}).get("severity", 0.3))
    return jsonify(execute_scenario_run("errors", severity))


@app.post("/run/dependency")
def run_dependency():
    severity = float((request.get_json(silent=True) or {}).get("severity", 1.0))
    return jsonify(execute_scenario_run("dependency", severity))


@app.post("/campaign")
def run_campaign():
    """Масовий прогон розширеної кампанії (~520 рядків)"""
    grid = {
        "cpu": [0.2, 0.4, 0.6, 0.75, 0.9, 1.0],
        "memory": [50.0, 100.0, 150.0, 200.0, 300.0, 400.0],
        "latency": [100.0, 300.0, 500.0, 700.0, 1000.0, 1500.0],
        "errors": [0.0, 0.05, 0.1, 0.2, 0.35, 0.5],
        "dependency": [0.0, 1.0]
    }
    repeats = 20  # 26 унікальних точок * 20 повторів = 520 рядків
    total_runs = 0

    for scenario, severities in grid.items():
        for sev in severities:
            for _ in range(repeats):
                execute_scenario_run(scenario, sev)
                total_runs += 1

    apply_fault("clear", 0)
    return jsonify(status="completed", total_runs=total_runs)


@app.get("/health")
def health():
    return jsonify(status="ok")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8100)