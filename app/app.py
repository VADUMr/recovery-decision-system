import random
import time
import threading
from flask import Flask, jsonify, request
from prometheus_client import Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST

app = Flask(__name__)

# --- стан несправностей (потокобезпечно) ---
state_lock = threading.Lock()
fault_state = {
    "cpu_load": 0.0,          # 0..1, скільки часу "зайняти" CPU на запит
    "memory_mb": 0.0,         # скільки МБ утримувати у пам'яті
    "latency_ms": 0.0,        # додаткова затримка на запит
    "error_probability": 0.0, # ймовірність повернути 500
    "dependency_failure": False,
}
_memory_ballast = []  # тут "тримаємо" пам'ять, якщо memory_mb > 0

# --- Prometheus-метрики ---
CPU_GAUGE = Gauge("ecdm_cpu_usage_percent", "Simulated CPU usage percent")
MEMORY_GAUGE = Gauge("ecdm_memory_usage_percent", "Simulated memory usage percent")
MEMORY_MB_GAUGE = Gauge("ecdm_memory_allocated_mb", "Allocated memory MB")
ERROR_RATE_GAUGE = Gauge("ecdm_error_rate_percent", "Error rate percent")
DEPENDENCY_GAUGE = Gauge("ecdm_dependency_available", "1 if dependency reachable else 0")
LATENCY_HIST = Histogram(
    "ecdm_request_latency_seconds",
    "Request latency",
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10),
)


@app.post("/fault")
def set_fault():
    payload = request.get_json(force=True)
    fault_type = payload.get("type")
    severity = float(payload.get("severity", 0))

    with state_lock:
        if fault_type == "clear":
            fault_state.update(
                cpu_load=0.0, memory_mb=0.0, latency_ms=0.0,
                error_probability=0.0, dependency_failure=False,
            )
            _memory_ballast.clear()
        elif fault_type == "cpu":
            fault_state["cpu_load"] = severity
        elif fault_type == "memory":
            fault_state["memory_mb"] = severity
            _memory_ballast.clear()
            _memory_ballast.append(bytearray(int(severity * 1024 * 1024)))
        elif fault_type == "latency":
            fault_state["latency_ms"] = severity
        elif fault_type == "errors":
            fault_state["error_probability"] = severity
        elif fault_type == "dependency":
            fault_state["dependency_failure"] = bool(severity)

    return jsonify(status="ok", fault_state=fault_state)


@app.get("/work")
@LATENCY_HIST.time()  # <--- Додано декоратор для автоматичного заміру часу виконання запиту
def work():
    with state_lock:
        cpu_load = fault_state["cpu_load"]
        latency_ms = fault_state["latency_ms"]
        error_probability = fault_state["error_probability"]
        dependency_failure = fault_state["dependency_failure"]

    # імітація навантаження CPU
    if cpu_load > 0:
        busy_until = time.perf_counter() + cpu_load * 0.05
        while time.perf_counter() < busy_until:
            pass

    # імітація затримки
    if latency_ms > 0:
        time.sleep(latency_ms / 1000.0)

    # звернення до залежності (якщо не "збита" штучно)
    if dependency_failure:
        return jsonify(error="dependency unavailable"), 503

    # випадкова помилка
    if random.random() < error_probability:
        return jsonify(error="simulated failure"), 500

    return jsonify(status="ok")


@app.get("/metrics")
def metrics():
    with state_lock:
        cpu_load = fault_state["cpu_load"]
        memory_mb = fault_state["memory_mb"]
        error_probability = fault_state["error_probability"]
        dependency_failure = fault_state["dependency_failure"]

    CPU_GAUGE.set(min(cpu_load * 100, 100))
    MEMORY_GAUGE.set(min((memory_mb / 512) * 100, 100))  # умовний max 512MB
    MEMORY_MB_GAUGE.set(memory_mb)
    ERROR_RATE_GAUGE.set(error_probability * 100)
    DEPENDENCY_GAUGE.set(0 if dependency_failure else 1)

    return generate_latest(), 200, {"Content-Type": CONTENT_TYPE_LATEST}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)