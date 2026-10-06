import random
import time
import threading
from flask import Flask, jsonify, request
from prometheus_client import Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST
import psutil

app = Flask(__name__)

# --- стан несправностей (потокобезпечно) ---
state_lock = threading.Lock()
fault_state = {
    "cpu_load": 0.0,          # 0..1, цільова частка ядер, зайнятих busy-потоками
    "memory_mb": 0.0,         # скільки МБ утримувати у пам'яті
    "latency_ms": 0.0,        # додаткова затримка на запит
    "error_probability": 0.0, # ймовірність повернути 500
    "dependency_failure": False,
}
_memory_ballast = []  # тут "тримаємо" пам'ять, якщо memory_mb > 0

# --- реальне навантаження CPU через фонові потоки (не копія параметра) ---
_cpu_stop_flag = threading.Event()
_cpu_workers = []


def _cpu_burn():
    while not _cpu_stop_flag.is_set():
        x = 0.0001
        for _ in range(50000):
            x = x * 1.0000001


def set_cpu_load(severity):
    global _cpu_workers
    stop_cpu_load()
    if severity <= 0:
        return
    n_cores = psutil.cpu_count(logical=True) or 4
    n_workers = max(1, round(severity * n_cores))
    _cpu_stop_flag.clear()
    _cpu_workers = [threading.Thread(target=_cpu_burn, daemon=True) for _ in range(n_workers)]
    for w in _cpu_workers:
        w.start()


def stop_cpu_load():
    _cpu_stop_flag.set()
    for w in _cpu_workers:
        w.join(timeout=0.1)
    _cpu_workers.clear()


# --- Prometheus-метрики ---
CPU_GAUGE = Gauge("ecdm_cpu_usage_percent", "Real measured CPU usage percent (psutil)")
MEMORY_GAUGE = Gauge("ecdm_memory_usage_percent", "Real measured system memory usage percent (psutil)")
MEMORY_MB_GAUGE = Gauge("ecdm_memory_allocated_mb", "Real measured process RSS memory MB (psutil)")
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
            stop_cpu_load()
        elif fault_type == "cpu":
            fault_state["cpu_load"] = severity
            set_cpu_load(severity)
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
@LATENCY_HIST.time()
def work():
    with state_lock:
        latency_ms = fault_state["latency_ms"]
        error_probability = fault_state["error_probability"]
        dependency_failure = fault_state["dependency_failure"]

    # реальне навантаження CPU тепер створюють фонові потоки (_cpu_burn),
    # а не цей запит — тому тут більше немає ручного busy-wait циклу

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
        error_probability = fault_state["error_probability"]
        dependency_failure = fault_state["dependency_failure"]

    # РЕАЛЬНІ виміри ОС (psutil), а не копія вхідного параметра severity —
    # звідси природна "негладкість" значень (напр. 82.37%, а не рівно 80.0)
    CPU_GAUGE.set(psutil.cpu_percent(interval=0.3))
    MEMORY_GAUGE.set(psutil.virtual_memory().percent)
    MEMORY_MB_GAUGE.set(psutil.Process().memory_info().rss / (1024 * 1024))
    ERROR_RATE_GAUGE.set(error_probability * 100)
    DEPENDENCY_GAUGE.set(0 if dependency_failure else 1)

    return generate_latest(), 200, {"Content-Type": CONTENT_TYPE_LATEST}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)