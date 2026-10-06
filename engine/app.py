import csv
import json
import os
import random
import statistics
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import requests
from flask import Flask, jsonify, request

app = Flask(__name__)

APP_URL = os.getenv("APP_URL", "http://app-service:8000")
PROMETHEUS_URL = os.getenv("PROMETHEUS_URL", "http://prometheus:9090")

# --- старий файл лишаємо як є (для історії пілотного етапу) ---
DATA_FILE = Path(os.getenv("DATA_FILE", "/data/pilot_results.csv"))

# --- новий файл: один рядок = одна спроба однієї дії в рамках інциденту ---
INCIDENT_DATA_FILE = Path(os.getenv("INCIDENT_DATA_FILE", "/data/incident_results.csv"))

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
                r = requests.get(f"{APP_URL}/work", timeout=5)
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


# =========================================================================
# ЛЕГАСІ (пілотний етап v1) — лишено недоторканим для історії дослідження,
# новим кодом нижче вже не використовується
# =========================================================================
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


# =========================================================================
# НОВА СХЕМА: incident-based кампанія (кожен інцидент -> 5 дій -> реальний успіх)
# =========================================================================

# Ефект дії: задокументоване припущення симульованого середовища.
# Дія вважається ефективною, якщо її механізм логічно усуває причину
# саме цього типу відмови (обґрунтування — розділ "Модель відмов" роботи).
ACTION_EFFECTIVENESS = {
    "cpu": {
        "Do Nothing": 0.05, "Scale Out": 0.90, "Restart Service": 0.25,
        "Rollback": 0.05, "Restart Dependency": 0.02,
    },
    "memory": {
        "Do Nothing": 0.05, "Restart Service": 0.85, "Scale Out": 0.30,
        "Rollback": 0.10, "Restart Dependency": 0.02,
    },
    "latency": {
        "Do Nothing": 0.10, "Rollback": 0.80, "Restart Service": 0.30,
        "Scale Out": 0.20, "Restart Dependency": 0.05,
    },
    "errors": {
        "Do Nothing": 0.10, "Rollback": 0.85, "Restart Service": 0.45,
        "Scale Out": 0.10, "Restart Dependency": 0.05,
    },
    "dependency": {
        "Do Nothing": 0.05, "Restart Dependency": 0.90, "Restart Service": 0.10,
        "Scale Out": 0.05, "Rollback": 0.05,
    },
}

ALL_ACTIONS = ["Do Nothing", "Scale Out", "Restart Service", "Rollback", "Restart Dependency"]

SEVERITY_RANGES = {
    "cpu": (0.3, 1.0),
    "memory": (50.0, 450.0),
    "latency": (100.0, 1600.0),
    "errors": (0.0, 0.55),
    "dependency": (0.0, 1.0),
}


def sample_severity(scenario_key):
    if scenario_key == "dependency":
        return random.choice([0.0, 1.0])
    low, high = SEVERITY_RANGES[scenario_key]
    return round(random.uniform(low, high), 3)


def is_fault_present(scenario_key, severity):
    policy = REFERENCE_POLICY["scenarios"][scenario_key]
    if policy.get("rule_type") == "binary_failure":
        return severity >= policy.get("failure_value", 1.0)
    return severity >= policy["threshold"]


def action_success_probability(scenario_key, severity, action):
    if not is_fault_present(scenario_key, severity):
        return 0.95  # системі й так нічого не загрожувало
    return ACTION_EFFECTIVENESS[scenario_key][action]


HEALTH_CONFIG = {
    "availability_min": 0.95,
    "latency_margin": 1.5,
    "latency_floor_ms": 500,
    "error_margin_pp": 2.0,
    "error_floor_pct": 5.0,
    "consecutive_healthy_required": 3,
    "max_recovery_checks": 8,
    "check_interval_s": 1.5,
}


def measure_baseline():
    apply_fault("clear", 0)
    time.sleep(1.5)
    load = generate_load(duration_seconds=3)
    obs = collect_observation()
    return {
        "availability": load["availability"],
        "latency_ms": obs["latency_ms"],
        "error_rate": obs["error_rate"],
    }


def is_healthy(observation, load, baseline):
    l_max = max(HEALTH_CONFIG["latency_floor_ms"], baseline["latency_ms"] * HEALTH_CONFIG["latency_margin"])
    e_max = max(HEALTH_CONFIG["error_floor_pct"], baseline["error_rate"] + HEALTH_CONFIG["error_margin_pp"])
    return (
        load["availability"] >= HEALTH_CONFIG["availability_min"]
        and observation["latency_ms"] <= l_max
        and observation["error_rate"] <= e_max
    )


def apply_recovery_action(scenario_key, severity, action):
    """Кидає монетку за ACTION_EFFECTIVENESS і, якщо 'спрацювало',
    реально знімає інжектовану відмову — тому X_post вимірюється по-справжньому."""
    success_roll = random.random() < action_success_probability(scenario_key, severity, action)
    if action != "Do Nothing" and success_roll:
        apply_fault("clear", 0)
    return success_roll


def wait_for_recovery(scenario_key, severity, action, baseline):
    t_start = time.time()
    consecutive = 0
    obs, load = collect_observation(), {"availability": 0.0}
    for _ in range(HEALTH_CONFIG["max_recovery_checks"]):
        load = generate_load(duration_seconds=2)
        obs = collect_observation()
        if is_healthy(obs, load, baseline):
            consecutive += 1
            if consecutive >= HEALTH_CONFIG["consecutive_healthy_required"]:
                return True, int((time.time() - t_start) * 1000), obs, load
        else:
            consecutive = 0
        time.sleep(HEALTH_CONFIG["check_interval_s"])
    return False, int((time.time() - t_start) * 1000), obs, load


INCIDENT_CSV_FIELDS = [
    "timestamp", "incident_group_id", "scenario", "severity", "replicate",
    "candidate_action", "expected_decision",
    "baseline_latency_ms", "baseline_error_rate", "baseline_availability",
    "pre_cpu", "pre_memory", "pre_latency_ms", "pre_error_rate", "pre_dependency_available",
    "pre_requests", "pre_successful_requests", "pre_median_client_latency_ms",
    "post_cpu", "post_memory", "post_latency_ms", "post_error_rate", "post_dependency_available",
    "post_availability", "recovery_success", "recovery_time_ms",
]


def save_incident_row(row):
    INCIDENT_DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    file_exists = INCIDENT_DATA_FILE.exists()
    with INCIDENT_DATA_FILE.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=INCIDENT_CSV_FIELDS, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def execute_incident(scenario_key, severity, replicate):
    incident_group_id = f"IG_{uuid.uuid4().hex[:10]}"
    baseline = measure_baseline()
    expected = get_reference_action(scenario_key, severity)

    for action in ALL_ACTIONS:
        apply_fault("clear", 0)
        time.sleep(1.0)
        apply_fault(scenario_key, severity)
        time.sleep(1.5)

        pre_load = generate_load(duration_seconds=3)
        pre_obs = collect_observation()

        apply_recovery_action(scenario_key, severity, action)
        success, recovery_time_ms, post_obs, post_load = wait_for_recovery(
            scenario_key, severity, action, baseline
        )

        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "incident_group_id": incident_group_id,
            "scenario": scenario_key, "severity": severity, "replicate": replicate,
            "candidate_action": action, "expected_decision": expected,
            "baseline_latency_ms": baseline["latency_ms"],
            "baseline_error_rate": baseline["error_rate"],
            "baseline_availability": baseline["availability"],
            "pre_cpu": pre_obs["cpu"], "pre_memory": pre_obs["memory"],
            "pre_latency_ms": pre_obs["latency_ms"], "pre_error_rate": pre_obs["error_rate"],
            "pre_dependency_available": pre_obs["dependency_available"],
            "pre_requests": pre_load["requests"], "pre_successful_requests": pre_load["successful_requests"],
            "pre_median_client_latency_ms": pre_load["median_client_latency_ms"],
            "post_cpu": post_obs["cpu"], "post_memory": post_obs["memory"],
            "post_latency_ms": post_obs["latency_ms"], "post_error_rate": post_obs["error_rate"],
            "post_dependency_available": post_obs["dependency_available"],
            "post_availability": post_load["availability"],
            "recovery_success": int(success), "recovery_time_ms": recovery_time_ms,
        }
        save_incident_row(row)

    apply_fault("clear", 0)
    return incident_group_id


@app.post("/run/cpu")
def run_cpu():
    severity = float((request.get_json(silent=True) or {}).get("severity", 0.9))
    return jsonify(incident_group_id=execute_incident("cpu", severity, replicate=0))


@app.post("/run/memory")
def run_memory():
    severity = float((request.get_json(silent=True) or {}).get("severity", 250.0))
    return jsonify(incident_group_id=execute_incident("memory", severity, replicate=0))


@app.post("/run/latency")
def run_latency():
    severity = float((request.get_json(silent=True) or {}).get("severity", 800.0))
    return jsonify(incident_group_id=execute_incident("latency", severity, replicate=0))


@app.post("/run/errors")
def run_errors():
    severity = float((request.get_json(silent=True) or {}).get("severity", 0.3))
    return jsonify(incident_group_id=execute_incident("errors", severity, replicate=0))


@app.post("/run/dependency")
def run_dependency():
    severity = float((request.get_json(silent=True) or {}).get("severity", 1.0))
    return jsonify(incident_group_id=execute_incident("dependency", severity, replicate=0))


@app.post("/campaign")
def run_campaign():
    """Кампанія з безперервним severity: для кожного сценарію генерується
    n_incidents_per_scenario інцидентів з випадковою severity в межах
    SEVERITY_RANGES, кожен інцидент пробує всі 5 дій."""
    payload = request.get_json(silent=True) or {}
    n_incidents_per_scenario = int(payload.get("n_incidents_per_scenario", 10))

    incident_ids = []
    for scenario in SEVERITY_RANGES:
        for i in range(n_incidents_per_scenario):
            severity = sample_severity(scenario)
            incident_ids.append(execute_incident(scenario, severity, replicate=i))

    apply_fault("clear", 0)
    return jsonify(status="completed", total_incidents=len(incident_ids))


@app.get("/health")
def health():
    return jsonify(status="ok")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8100)