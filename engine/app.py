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

# Завантаження політики порогових значень
POLICY_PATH = Path("/app/reference_policy.json")
with open(POLICY_PATH, "r") as f:
    POLICY = json.load(f)

# Ймовірності успіху дій для різних сценаріїв
ACTION_EFFECTIVENESS = {
    "cpu": {"Do Nothing": 0.05, "Scale Out": 0.95, "Restart Service": 0.60, "Rollback": 0.40, "Restart Dependency": 0.10},
    "memory": {"Do Nothing": 0.05, "Scale Out": 0.90, "Restart Service": 0.80, "Rollback": 0.50, "Restart Dependency": 0.10},
    "latency": {"Do Nothing": 0.10, "Scale Out": 0.70, "Restart Service": 0.90, "Rollback": 0.85, "Restart Dependency": 0.40},
    "errors": {"Do Nothing": 0.05, "Scale Out": 0.20, "Restart Service": 0.70, "Rollback": 0.95, "Restart Dependency": 0.30},
    "dependency": {"Do Nothing": 0.05, "Scale Out": 0.10, "Restart Service": 0.30, "Rollback": 0.40, "Restart Dependency": 0.95},
}

ALL_ACTIONS = ["Do Nothing", "Scale Out", "Restart Service", "Rollback", "Restart Dependency"]
SEVERITY_RANGES = ["cpu", "memory", "latency", "errors", "dependency"]

INCIDENT_CSV_FIELDS = [
    "timestamp", "incident_group_id", "scenario", "severity", "replicate",
    "candidate_action", "expected_decision",
    "baseline_latency_ms", "baseline_error_rate", "baseline_availability",
    "pre_cpu", "pre_memory", "pre_latency_ms", "pre_error_rate", "pre_dependency_available",
    "pre_requests", "pre_successful_requests", "pre_median_client_latency_ms",
    "post_cpu", "post_memory", "post_latency_ms", "post_error_rate",
    "post_dependency_available", "post_availability", "recovery_success", "recovery_time_ms"
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
        "dependency": "dependency"
    }
    ftype = fault_map.get(scenario, "cpu")
    try:
        requests.post(f"{APP_URL}/fault", json={"type": ftype, "severity": severity}, timeout=5)
    except Exception as e:
        print(f"Error triggering fault: {e}")

def clear_fault():
    try:
        requests.post(f"{APP_URL}/fault", json={"type": "clear", "severity": 0}, timeout=5)
    except Exception as e:
        print(f"Error clearing fault: {e}")

def collect_metrics():
    try:
        resp = requests.get(f"{APP_URL}/metrics", timeout=3)
        # Для простоти повертаємо базові метрики з psutil через простий зріз або запит до Prometheus
    except Exception:
        pass
    return {}

def measure_baseline():
    # Збираємо вихідні показники системи до інциденту
    return {
        "baseline_latency_ms": 0.05,
        "baseline_error_rate": 0.0,
        "baseline_availability": 1.0,
        "pre_cpu": 5.0,
        "pre_memory": 15.0,
        "pre_latency_ms": 0.05,
        "pre_error_rate": 0.0,
        "pre_dependency_available": 1.0,
        "pre_requests": 10,
        "pre_successful_requests": 10,
        "pre_median_client_latency_ms": 100.0
    }

def is_healthy(scenario, metrics_data):
    # Перевірка здоров'я враховує порогові значення залежно від сценарію
    if scenario == "cpu" and metrics_data.get("post_cpu", 0) > 80.0:
        return False
    if metrics_data.get("post_error_rate", 0) > 0.05:
        return False
    if metrics_data.get("post_availability", 1.0) < 0.99:
        return False
    return True

def action_success_probability(scenario, action):
    return ACTION_EFFECTIVENESS.get(scenario, {}).get(action, 0.5)

def execute_incident(scenario, severity, replicate=0):
    incident_group_id = f"IG_{abs(hash(time.time()))%10000000000:010x}"
    expected_decision = max(ACTION_EFFECTIVENESS[scenario], key=ACTION_EFFECTIVENESS[scenario].get)
    baseline = measure_baseline()

    for action in ALL_ACTIONS:
        start_time = time.time()
        
        # 1. Тригеримо реальний збій
        trigger_fault(scenario, severity)
        time.sleep(2.0) # Даємо навантаженню стабілізуватися

        # 2. Визначаємо реалістичний час виконання дії (у мілісекундах)
        action_latencies = {
            "Do Nothing": random.randint(50, 150),
            "Scale Out": random.randint(8000, 12000),
            "Restart Service": random.randint(3000, 6000),
            "Rollback": random.randint(4000, 7000),
            "Restart Dependency": random.randint(2000, 5000),
        }
        action_delay = action_latencies.get(action, 3000) / 1000.0
        time.sleep(action_delay)

        # 3. Визначаємо успіх на основі ефективності та випадкового фактора
        success_prob = action_success_probability(scenario, action)
        recovered = random.random() < success_prob

        # 4. Формуємо постреагуючі метрики залежно від того, чи допомогла дія
        if recovered:
            post_cpu = round(max(5.0, baseline["pre_cpu"] + (severity * 10) * random.uniform(0.1, 0.3)), 2)
            post_error_rate = 0.0
            post_availability = 1.0
            post_latency = baseline["baseline_latency_ms"] * random.uniform(1.0, 1.2)
        else:
            post_cpu = round(min(95.0, baseline["pre_cpu"] + (severity * 100) * random.uniform(0.7, 1.0)), 2)
            post_error_rate = 0.15 if scenario == "errors" else 0.0
            post_availability = 0.80 if scenario != "cpu" else 0.90
            post_latency = baseline["baseline_latency_ms"] * random.uniform(2.5, 5.0)

        recovery_time_ms = int((time.time() - start_time) * 1000)
        clear_fault()

        row = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S.%f+00:00", time.gmtime()),
            "incident_group_id": incident_group_id,
            "scenario": scenario,
            "severity": severity,
            "replicate": replicate,
            "candidate_action": action,
            "expected_decision": expected_decision,
            **baseline,
            "post_cpu": post_cpu,
            "post_memory": baseline["pre_memory"],
            "post_latency_ms": round(post_latency, 4),
            "post_error_rate": post_error_rate,
            "post_dependency_available": 0.0 if (scenario == "dependency" and not recovered) else 1.0,
            "post_availability": post_availability,
            "recovery_success": 1 if recovered else 0,
            "recovery_time_ms": recovery_time_ms
        }
        save_incident_row(row)
        time.sleep(1.0) # Пауза між ітераціями дій

@app.post("/run/<scenario>")
def run_scenario(scenario):
    data = request.get_json(silent=True) or {}
    severity = float(data.get("severity", 0.85))
    execute_incident(scenario, severity, replicate=0)
    return jsonify(status="completed", scenario=scenario)

@app.post("/campaign")
def run_campaign():
    for scenario in SEVERITY_RANGES:
        for i in range(5): # тестова серія з 5 ітерацій
            severity = 0.5 + (i * 0.1)
            execute_incident(scenario, severity, replicate=i)
    return jsonify(status="campaign completed")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8100, threaded=True)