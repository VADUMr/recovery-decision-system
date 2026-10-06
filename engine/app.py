import os
import time
import json
import csv
import uuid
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from flask import Flask, jsonify, request
import random

app = Flask(__name__)

APP_URL = os.getenv("APP_URL", "http://app-service:8000")
PROMETHEUS_URL = os.getenv("PROMETHEUS_URL", "http://prometheus:9090")
INCIDENT_DATA_FILE = Path(os.getenv("INCIDENT_DATA_FILE", "/data/incident_results.csv"))

POLICY_PATH = Path("/app/reference_policy.json")
with open(POLICY_PATH, "r", encoding="utf-8") as f:
    POLICY = json.load(f)

ALL_ACTIONS = [
    "Do Nothing",
    "Scale Out",
    "Restart Service",
    "Rollback",
    "Restart Dependency",
]

# Фізично осмислені діапазони severity для кожного сценарію
SEVERITY_RANGES = {
    "cpu": (0.30, 1.00),       # коефіцієнт навантаження CPU [0..1]
    "memory": (50.0, 450.0),   # МБ ballast
    "latency": (100.0, 1600.0),# мс штучної затримки
    "errors": (0.05, 0.90),    # ймовірність помилки
    "dependency": (0.0, 1.0),  # 0 = ok, >=0.5 = failure
}

# Скільки HTTP-запитів до /work на одне спостереження
LOAD_REQUESTS = int(os.getenv("LOAD_REQUESTS", "20"))
LOAD_WORKERS = int(os.getenv("LOAD_WORKERS", "8"))
# Скільки послідовних «healthy» замірів потрібно для success
STABLE_HEALTHY_SAMPLES = int(os.getenv("STABLE_HEALTHY_SAMPLES", "2"))

INCIDENT_CSV_FIELDS = [
    "timestamp", "incident_group_id", "scenario", "severity", "replicate",
    "candidate_action", "expected_decision",
    "baseline_latency_ms", "baseline_error_rate", "baseline_availability",
    "pre_cpu", "pre_memory", "pre_latency_ms", "pre_error_rate",
    "pre_dependency_available", "pre_requests", "pre_successful_requests",
    "pre_median_client_latency_ms",
    "post_cpu", "post_memory", "post_latency_ms", "post_error_rate",
    "post_dependency_available", "post_availability",
    "recovery_success", "recovery_time_ms",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def save_incident_row(row_dict: dict) -> None:
    INCIDENT_DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    file_exists = INCIDENT_DATA_FILE.exists()
    with open(INCIDENT_DATA_FILE, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=INCIDENT_CSV_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_dict)


def sample_severity(scenario: str) -> float:
    lo, hi = SEVERITY_RANGES[scenario]
    return round(random.uniform(lo, hi), 4)


def get_reference_action(scenario: str, severity: float) -> str:
    """Порогова / бінарна логіка з reference_policy.json."""
    cfg = POLICY["scenarios"].get(scenario, {})
    rule = cfg.get("rule_type", "threshold")

    if rule == "binary_failure":
        failure_value = float(cfg.get("failure_value", 1.0))
        if severity >= failure_value * 0.5:  # >= 0.5 → failure
            return cfg.get("failure_action", "Restart Dependency")
        return cfg.get("healthy_action", "Do Nothing")

    # threshold
    threshold = float(cfg.get("threshold", 0.5))
    if severity >= threshold:
        return cfg.get("at_or_above_threshold_action", "Do Nothing")
    return cfg.get("below_threshold_action", "Do Nothing")


# ---------------------------------------------------------------------------
# Fault control
# ---------------------------------------------------------------------------

def trigger_fault(scenario: str, severity: float) -> None:
    fault_map = {
        "cpu": "cpu",
        "memory": "memory",
        "latency": "latency",
        "errors": "errors",
        "dependency": "dependency",
    }
    ftype = fault_map.get(scenario, "cpu")
    # Для dependency: severity >= 0.5 → failure=1
    payload_severity = severity
    if scenario == "dependency":
        payload_severity = 1.0 if severity >= 0.5 else 0.0
    try:
        requests.post(
            f"{APP_URL}/fault",
            json={"type": ftype, "severity": payload_severity},
            timeout=5,
        )
    except Exception as e:
        print(f"[engine] Error triggering fault: {e}")


def clear_fault() -> None:
    try:
        requests.post(
            f"{APP_URL}/fault",
            json={"type": "clear", "severity": 0},
            timeout=5,
        )
    except Exception as e:
        print(f"[engine] Error clearing fault: {e}")


def apply_recovery_action(scenario: str, action: str, severity: float) -> None:
    """
    Ефект recovery action на поточний fault-стан.
    У спрощеному стенді (без реального k8s scale-out) моделюємо
    семантику дій через clear / partial-clear / leave-as-is.
    """
    if action == "Do Nothing":
        return  # fault залишається

    # Дії, які «лікують» відповідний сценарій — повний clear
    curative = {
        "cpu": {"Scale Out", "Restart Service"},
        "memory": {"Scale Out", "Restart Service"},
        "latency": {"Restart Service", "Rollback"},
        "errors": {"Rollback", "Restart Service"},
        "dependency": {"Restart Dependency"},
    }
    if action in curative.get(scenario, set()):
        clear_fault()
        return

    # Частково доречні дії — зменшуємо severity (але не прибираємо повністю)
    partial = {
        "cpu": {"Rollback"},
        "memory": {"Rollback"},
        "latency": {"Scale Out"},
        "errors": {"Scale Out"},
        "dependency": {"Restart Service", "Rollback"},
    }
    if action in partial.get(scenario, set()):
        reduced = max(0.0, severity * 0.35)
        trigger_fault(scenario, reduced)
        return

    # Решта — practically no effect (залишаємо fault)
    return


# ---------------------------------------------------------------------------
# Real measurement
# ---------------------------------------------------------------------------

def _single_work_request() -> dict:
    """Один HTTP GET /work → {ok, latency_ms, status_code}."""
    t0 = time.perf_counter()
    try:
        r = requests.get(f"{APP_URL}/work", timeout=15)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        ok = r.status_code == 200
        return {"ok": ok, "latency_ms": latency_ms, "status_code": r.status_code}
    except Exception:
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return {"ok": False, "latency_ms": latency_ms, "status_code": 0}


def generate_load(n_requests: int = LOAD_REQUESTS) -> dict:
    """Паралельні запити до /work → агреговані client-side метрики."""
    results = []
    with ThreadPoolExecutor(max_workers=LOAD_WORKERS) as pool:
        futures = [pool.submit(_single_work_request) for _ in range(n_requests)]
        for fut in as_completed(futures):
            results.append(fut.result())

    n = len(results) or 1
    successes = sum(1 for r in results if r["ok"])
    latencies = sorted(r["latency_ms"] for r in results)
    median_lat = latencies[len(latencies) // 2] if latencies else 0.0
    error_rate = 1.0 - (successes / n)
    availability = successes / n

    return {
        "requests": n,
        "successful_requests": successes,
        "error_rate": round(error_rate, 4),
        "availability": round(availability, 4),
        "median_client_latency_ms": round(median_lat, 2),
        "mean_client_latency_ms": round(sum(latencies) / n, 2) if latencies else 0.0,
    }


def query_prometheus(promql: str, default: float = 0.0) -> float:
    """Миттєве значення з Prometheus (query)."""
    try:
        r = requests.get(
            f"{PROMETHEUS_URL}/api/v1/query",
            params={"query": promql},
            timeout=3,
        )
        data = r.json()
        result = data.get("data", {}).get("result", [])
        if result:
            return float(result[0]["value"][1])
    except Exception as e:
        print(f"[engine] Prometheus query failed ({promql}): {e}")
    return default


def collect_resource_metrics() -> dict:
    """
    CPU / memory з Prometheus (якщо доступний) або fallback через /metrics scrape.
    Також dependency availability з gauge.
    """
    # Prometheus metrics exported by app-service
    cpu = query_prometheus("ecdm_cpu_usage_percent", default=-1.0)
    mem_mb = query_prometheus("ecdm_memory_allocated_mb", default=-1.0)
    dep = query_prometheus("ecdm_dependency_available", default=-1.0)

    # Fallback: parse text exposition from /metrics
    if cpu < 0 or mem_mb < 0:
        try:
            # Trigger gauge update
            requests.get(f"{APP_URL}/metrics", timeout=3)
            text = requests.get(f"{APP_URL}/metrics", timeout=3).text
            for line in text.splitlines():
                if line.startswith("ecdm_cpu_usage_percent "):
                    cpu = float(line.split()[-1])
                elif line.startswith("ecdm_memory_allocated_mb "):
                    mem_mb = float(line.split()[-1])
                elif line.startswith("ecdm_dependency_available "):
                    dep = float(line.split()[-1])
        except Exception as e:
            print(f"[engine] /metrics fallback failed: {e}")

    if cpu < 0:
        cpu = 0.0
    if mem_mb < 0:
        mem_mb = 0.0
    if dep < 0:
        dep = 1.0

    return {
        "cpu": round(cpu, 2),
        "memory_mb": round(mem_mb, 2),
        "dependency_available": 1.0 if dep >= 0.5 else 0.0,
    }


def collect_observation() -> dict:
    """
    Повне спостереження стану системи:
      - resource metrics (CPU, memory, dependency)
      - client-side load test (/work)
    """
    # Оновлюємо gauges
    try:
        requests.get(f"{APP_URL}/metrics", timeout=2)
    except Exception:
        pass

    resources = collect_resource_metrics()
    load = generate_load()

    return {
        "cpu": resources["cpu"],
        "memory": resources["memory_mb"],
        "latency_ms": load["median_client_latency_ms"] / 1000.0,  # зберігаємо в секундах як раніше
        "error_rate": load["error_rate"],
        "dependency_available": resources["dependency_available"],
        "requests": load["requests"],
        "successful_requests": load["successful_requests"],
        "median_client_latency_ms": load["median_client_latency_ms"],
        "availability": load["availability"],
    }


def is_healthy(obs: dict, scenario: str) -> bool:
    """
    Критерії здорового стану (узгоджені з експериментом).
    Усі умови мають виконуватися одночасно.
    """
    if obs["availability"] < 0.95:
        return False
    if obs["error_rate"] > 0.05:
        return False
    if obs["median_client_latency_ms"] > 500.0:  # жорсткий SLA для стенду
        return False
    if scenario == "cpu" and obs["cpu"] > 85.0:
        return False
    if scenario == "memory" and obs["memory"] > 400.0:
        return False
    if scenario == "dependency" and obs["dependency_available"] < 0.5:
        return False
    return True


def wait_stable_health(scenario: str, samples: int = STABLE_HEALTHY_SAMPLES,
                       interval: float = 1.0) -> tuple[bool, dict]:
    """
    Перевіряє, чи система стабільно здорова протягом `samples` замірів.
    Повертає (success, last_observation).
    """
    last = None
    for _ in range(samples):
        last = collect_observation()
        if not is_healthy(last, scenario):
            return False, last
        time.sleep(interval)
    return True, last


# ---------------------------------------------------------------------------
# Incident execution
# ---------------------------------------------------------------------------

def execute_incident(scenario: str, severity: float, replicate: int = 0) -> None:
    incident_group_id = f"IG_{uuid.uuid4().hex[:10]}"
    expected_decision = get_reference_action(scenario, severity)

    print(f"[engine] === Incident {incident_group_id} | {scenario} sev={severity} "
          f"| expected={expected_decision} ===")

    # --- 1. Healthy baseline ---
    clear_fault()
    time.sleep(1.0)
    baseline_obs = collect_observation()
    baseline = {
        "baseline_latency_ms": round(baseline_obs["median_client_latency_ms"] / 1000.0, 4),
        "baseline_error_rate": baseline_obs["error_rate"],
        "baseline_availability": baseline_obs["availability"],
    }

    # --- 2. Inject fault (once for the whole group) ---
    trigger_fault(scenario, severity)
    time.sleep(2.0)  # даємо fault проявитися

    # --- 3. Pre-metrics (shared across all actions of this group) ---
    pre_obs = collect_observation()
    pre_metrics = {
        "pre_cpu": pre_obs["cpu"],
        "pre_memory": pre_obs["memory"],
        "pre_latency_ms": round(pre_obs["latency_ms"], 4),
        "pre_error_rate": pre_obs["error_rate"],
        "pre_dependency_available": pre_obs["dependency_available"],
        "pre_requests": pre_obs["requests"],
        "pre_successful_requests": pre_obs["successful_requests"],
        "pre_median_client_latency_ms": pre_obs["median_client_latency_ms"],
    }
    print(f"[engine]   pre: cpu={pre_metrics['pre_cpu']} "
          f"err={pre_metrics['pre_error_rate']} "
          f"lat_ms={pre_metrics['pre_median_client_latency_ms']} "
          f"dep={pre_metrics['pre_dependency_available']}")

    for action in ALL_ACTIONS:
        # Переінжектимо fault перед кожною дією (щоб умови були однаковими)
        clear_fault()
        time.sleep(0.5)
        trigger_fault(scenario, severity)
        time.sleep(1.5)

        start_time = time.time()

        # --- 4. Apply recovery action ---
        apply_recovery_action(scenario, action, severity)
        time.sleep(1.0)  # даємо дії подіяти

        # --- 5. Post observation + stable health check ---
        recovered, post_obs = wait_stable_health(scenario)
        recovery_time_ms = int((time.time() - start_time) * 1000)

        row = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S.%f+00:00", time.gmtime()),
            "incident_group_id": incident_group_id,
            "scenario": scenario,
            "severity": severity,
            "replicate": replicate,
            "candidate_action": action,
            "expected_decision": expected_decision,
            **baseline,
            **pre_metrics,
            "post_cpu": post_obs["cpu"],
            "post_memory": post_obs["memory"],
            "post_latency_ms": round(post_obs["latency_ms"], 4),
            "post_error_rate": post_obs["error_rate"],
            "post_dependency_available": post_obs["dependency_available"],
            "post_availability": post_obs["availability"],
            "recovery_success": 1 if recovered else 0,
            "recovery_time_ms": recovery_time_ms,
        }
        save_incident_row(row)
        print(f"[engine]   action={action:20s} success={recovered} "
              f"post_avail={post_obs['availability']} t={recovery_time_ms}ms")

        time.sleep(0.5)

    clear_fault()
    print(f"[engine] === Done {incident_group_id} ===\n")


# ---------------------------------------------------------------------------
# HTTP API
# ---------------------------------------------------------------------------

@app.post("/run/<scenario>")
def run_scenario(scenario: str):
    if scenario not in SEVERITY_RANGES:
        return jsonify(error=f"Unknown scenario: {scenario}"), 400
    data = request.get_json(silent=True) or {}
    severity = float(data.get("severity", sample_severity(scenario)))
    replicate = int(data.get("replicate", 0))
    execute_incident(scenario, severity, replicate=replicate)
    return jsonify(status="completed", scenario=scenario, severity=severity)


@app.post("/campaign")
def run_campaign():
    """
    Повна кампанія.
    Query/body params:
      n_per_scenario  — скільки інцидентів на сценарій (default 5 для smoke-test)
    """
    data = request.get_json(silent=True) or {}
    n_per = int(data.get("n_per_scenario", 5))

    summary = {}
    for scenario in SEVERITY_RANGES:
        summary[scenario] = []
        for i in range(n_per):
            severity = sample_severity(scenario)
            execute_incident(scenario, severity, replicate=i)
            summary[scenario].append(severity)

    return jsonify(status="campaign completed", n_per_scenario=n_per, severities=summary)


@app.get("/health")
def health():
    return jsonify(status="ok", policy=POLICY.get("policy_name"))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8100, threaded=True)