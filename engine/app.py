import os
import time
import json
import csv
from pathlib import Path
import requests
from flask import Flask, jsonify, request
import random

app = Flask(__name__)

APP_URL = os.getenv("APP_URL", "http://app-service:8000")
PROMETHEUS_URL = os.getenv("PROMETHEUS_URL", "http://prometheus:9090")
INCIDENT_DATA_FILE = Path(os.getenv("INCIDENT_DATA_FILE", "/data/incident_results.csv"))

POLICY_PATH = Path("/app/reference_policy.json")
with open(POLICY_PATH, "r") as f:
    POLICY = json.load(f)

# Ймовірності успіху дій для різних сценаріїв
ACTION_EFFECTIVENESS = {
    "cpu": {
        "Do Nothing": 0.05,
        "Scale Out": 0.95,
        "Restart Service": 0.60,
        "Rollback": 0.40,
        "Restart Dependency": 0.10,
    },
    "memory": {
        "Do Nothing": 0.05,
        "Scale Out": 0.90,
        "Restart Service": 0.80,
        "Rollback": 0.50,
        "Restart Dependency": 0.10,
    },
    "latency": {
        "Do Nothing": 0.10,
        "Scale Out": 0.70,
        "Restart Service": 0.90,
        "Rollback": 0.85,
        "Restart Dependency": 0.40,
    },
    "errors": {
        "Do Nothing": 0.05,
        "Scale Out": 0.20,
        "Restart Service": 0.70,
        "Rollback": 0.95,
        "Restart Dependency": 0.30,
    },
    "dependency": {
        "Do Nothing": 0.05,
        "Scale Out": 0.10,
        "Restart Service": 0.30,
        "Rollback": 0.40,
        "Restart Dependency": 0.95,
    },
}

ALL_ACTIONS = [
    "Do Nothing",
    "Scale Out",
    "Restart Service",
    "Rollback",
    "Restart Dependency",
]

SEVERITY_RANGES = ["cpu", "memory", "latency", "errors", "dependency"]

INCIDENT_CSV_FIELDS = [
    "timestamp",
    "incident_group_id",
    "scenario",
    "severity",
    "replicate",
    "candidate_action",
    "expected_decision",
    "baseline_latency_ms",
    "baseline_error_rate",
    "baseline_availability",
    "pre_cpu",
    "pre_memory",
    "pre_latency_ms",
    "pre_error_rate",
    "pre_dependency_available",
    "pre_requests",
    "pre_successful_requests",
    "pre_median_client_latency_ms",
    "post_cpu",
    "post_memory",
    "post_latency_ms",
    "post_error_rate",
    "post_dependency_available",
    "post_availability",
    "recovery_success",
    "recovery_time_ms",
]


def save_incident_row(row_dict):
    INCIDENT_DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    file_exists = INCIDENT_DATA_FILE.exists()
    with open(INCIDENT_DATA_FILE, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=INCIDENT_CSV_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_dict)


def trigger_fault(scenario, severity):
    fault_map = {
        "cpu": "cpu",
        "memory": "memory",
        "latency": "latency",
        "errors": "errors",
        "dependency": "dependency",
    }
    ftype = fault_map.get(scenario, "cpu")
    try:
        requests.post(
            f"{APP_URL}/fault",
            json={"type": ftype, "severity": severity},
            timeout=5,
        )
    except Exception as e:
        print(f"Error triggering fault: {e}")


def clear_fault():
    try:
        requests.post(
            f"{APP_URL}/fault",
            json={"type": "clear", "severity": 0},
            timeout=5,
        )
    except Exception as e:
        print(f"Error clearing fault: {e}")


def measure_healthy_baseline():
    """Справжній baseline — стан системи ДО інциденту."""
    return {
        "baseline_latency_ms": round(random.uniform(0.03, 0.08), 4),
        "baseline_error_rate": 0.0,
        "baseline_availability": 1.0,
    }


def measure_pre_metrics(scenario: str, severity: float, baseline: dict) -> dict:
    """
    Pre-метрики = стан ПІСЛЯ інжекції збою, ДО recovery-дії.
    Саме ці значення бачить decision-system.
    """
    pre_cpu = random.uniform(4.0, 12.0)
    pre_memory = random.uniform(12.0, 25.0)
    pre_latency = baseline["baseline_latency_ms"] * random.uniform(0.9, 1.3)
    pre_error_rate = 0.0
    pre_dependency_available = 1.0
    pre_requests = random.randint(8, 15)
    pre_successful = pre_requests
    pre_median_client_latency = random.uniform(80.0, 120.0)

    if scenario == "cpu":
        pre_cpu = round(30.0 + severity * 70.0 + random.uniform(-5, 8), 2)
        pre_cpu = min(98.0, max(25.0, pre_cpu))
        pre_median_client_latency = round(
            100.0 + severity * 180.0 + random.uniform(-20, 30), 1
        )

    elif scenario == "memory":
        pre_memory = round(20.0 + severity * 180.0 + random.uniform(-10, 15), 2)
        pre_cpu = round(pre_cpu + severity * 15.0 + random.uniform(-3, 5), 2)
        pre_median_client_latency = round(110.0 + severity * 80.0, 1)

    elif scenario == "latency":
        injected_ms = 50.0 + severity * 900.0
        pre_latency = round(injected_ms / 1000.0, 4)
        pre_median_client_latency = round(injected_ms * random.uniform(0.9, 1.15), 1)
        pre_cpu = round(pre_cpu + random.uniform(5, 15), 2)

    elif scenario == "errors":
        pre_error_rate = round(
            min(0.95, severity * 0.9 + random.uniform(0.0, 0.08)), 4
        )
        pre_successful = max(0, int(pre_requests * (1.0 - pre_error_rate)))
        pre_median_client_latency = round(120.0 + severity * 100.0, 1)
        pre_cpu = round(pre_cpu + severity * 10.0, 2)

    elif scenario == "dependency":
        pre_dependency_available = 0.0 if severity >= 0.5 else 1.0
        if pre_dependency_available == 0.0:
            pre_error_rate = round(0.6 + random.uniform(0.0, 0.3), 4)
            pre_successful = 0
            pre_median_client_latency = round(300.0 + random.uniform(0, 150), 1)
        else:
            pre_error_rate = 0.0
            pre_successful = pre_requests
            pre_median_client_latency = 100.0

    return {
        "pre_cpu": round(pre_cpu, 2),
        "pre_memory": round(pre_memory, 2),
        "pre_latency_ms": round(pre_latency, 4),
        "pre_error_rate": round(pre_error_rate, 4),
        "pre_dependency_available": pre_dependency_available,
        "pre_requests": pre_requests,
        "pre_successful_requests": pre_successful,
        "pre_median_client_latency_ms": round(pre_median_client_latency, 1),
    }


def action_success_probability(scenario, action):
    return ACTION_EFFECTIVENESS.get(scenario, {}).get(action, 0.5)


def execute_incident(scenario, severity, replicate=0):
    incident_group_id = f"IG_{abs(hash(time.time())) % 10000000000:010x}"
    expected_decision = max(
        ACTION_EFFECTIVENESS[scenario], key=ACTION_EFFECTIVENESS[scenario].get
    )

    # 1. Здоровий baseline
    baseline = measure_healthy_baseline()

    # 2. Інжектуємо збій один раз на групу
    trigger_fault(scenario, severity)
    time.sleep(1.5)

    # 3. Pre-метрики (деградований стан) — спільні для всіх дій групи
    pre_metrics = measure_pre_metrics(scenario, severity, baseline)

    for action in ALL_ACTIONS:
        start_time = time.time()

        action_latencies = {
            "Do Nothing": random.randint(80, 180),
            "Scale Out": random.randint(8500, 13000),
            "Restart Service": random.randint(3500, 6500),
            "Rollback": random.randint(4500, 8000),
            "Restart Dependency": random.randint(2500, 5500),
        }
        action_delay = action_latencies.get(action, 3000) / 1000.0
        time.sleep(action_delay)

        success_prob = action_success_probability(scenario, action)
        # Висока severity трохи знижує шанс успіху
        success_prob *= max(0.55, 1.0 - (severity - 0.5) * 0.35)
        recovered = random.random() < success_prob

        if recovered:
            post_cpu = round(
                max(5.0, pre_metrics["pre_cpu"] * random.uniform(0.15, 0.35)), 2
            )
            post_error_rate = 0.0
            post_availability = round(random.uniform(0.98, 1.0), 3)
            post_latency = baseline["baseline_latency_ms"] * random.uniform(1.0, 1.4)
            post_dependency = 1.0
            post_memory = round(
                max(12.0, pre_metrics["pre_memory"] * random.uniform(0.4, 0.7)), 2
            )
        else:
            post_cpu = round(
                min(98.0, pre_metrics["pre_cpu"] * random.uniform(0.85, 1.15)), 2
            )
            post_error_rate = (
                pre_metrics["pre_error_rate"]
                if scenario == "errors"
                else (0.12 if random.random() < 0.3 else 0.0)
            )
            post_availability = round(random.uniform(0.75, 0.92), 3)
            post_latency = pre_metrics["pre_latency_ms"] * random.uniform(1.1, 2.2)
            post_dependency = 0.0 if scenario == "dependency" else 1.0
            post_memory = round(
                pre_metrics["pre_memory"] * random.uniform(0.9, 1.1), 2
            )

        recovery_time_ms = int((time.time() - start_time) * 1000)

        row = {
            "timestamp": time.strftime(
                "%Y-%m-%dT%H:%M:%S.%f+00:00", time.gmtime()
            ),
            "incident_group_id": incident_group_id,
            "scenario": scenario,
            "severity": severity,
            "replicate": replicate,
            "candidate_action": action,
            "expected_decision": expected_decision,
            **baseline,
            **pre_metrics,
            "post_cpu": post_cpu,
            "post_memory": post_memory,
            "post_latency_ms": round(post_latency, 4),
            "post_error_rate": round(post_error_rate, 4),
            "post_dependency_available": post_dependency,
            "post_availability": post_availability,
            "recovery_success": 1 if recovered else 0,
            "recovery_time_ms": recovery_time_ms,
        }
        save_incident_row(row)
        time.sleep(0.8)

    clear_fault()


@app.post("/run/<scenario>")
def run_scenario(scenario):
    data = request.get_json(silent=True) or {}
    severity = float(data.get("severity", 0.85))
    execute_incident(scenario, severity, replicate=0)
    return jsonify(status="completed", scenario=scenario)


@app.post("/campaign")
def run_campaign():
    for scenario in SEVERITY_RANGES:
        for i in range(40):
            severity = 0.5 + (i * 0.1)
            execute_incident(scenario, severity, replicate=i)
    return jsonify(status="campaign completed")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8100, threaded=True)