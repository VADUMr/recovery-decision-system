import random
import time
import threading
from flask import Flask, jsonify, request
from prometheus_client import Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST
import psutil

app = Flask(__name__)

# --- Стан несправностей (потокобезпечно) ---
state_lock = threading.Lock()
fault_state = {
    "cpu_load": 0.0,
    "memory_mb": 0.0,
    "latency_ms": 0.0,
    "error_probability": 0.0,
    "dependency_failure": False,
}
_memory_ballast = []

# --- CPU: один duty-cycle потік замість N потоків (GIL робить "N потоків" марними) ---
CPU_CYCLE_SECONDS = 1.0
_cpu_stop_flag = threading.Event()
_cpu_thread = None


def _cpu_duty_cycle(severity):
    while not _cpu_stop_flag.is_set():
        busy_until = time.perf_counter() + severity * CPU_CYCLE_SECONDS
        while time.perf_counter() < busy_until and not _cpu_stop_flag.is_set():
            x = 0.0001
            for _ in range(2000):
                x = x * 1.0000001
        idle_seconds = (1 - severity) * CPU_CYCLE_SECONDS
        if idle_seconds > 0:
            _cpu_stop_flag.wait(timeout=idle_seconds)


def set_cpu_load(severity):
    global _cpu_thread
    stop_cpu_load()
    if severity <= 0:
        return
    _cpu_stop_flag.clear()
    _cpu_thread = threading.Thread(target=_cpu_duty_cycle, args=(severity,), daemon=True)
    _cpu_thread.start()


def stop_cpu_load():
    global _cpu_thread
    _cpu_stop_flag.set()
    if _cpu_thread is not None:
        _cpu_thread.join(timeout=1.0)
    _cpu_thread = None


# --- Prometheus-метрики ---
CPU_GAUGE = Gauge("ecdm_cpu_usage_percent", "Real measured CPU usage percent of this process (psutil)")
MEMORY_GAUGE = Gauge("ecdm_memory_usage_percent", "Real measured system memory usage percent (psutil)")
MEMORY_MB_GAUGE = Gauge("ecdm_memory_allocated_mb", "Real measured process RSS memory MB (psutil)")
ERROR_RATE_GAUGE = Gauge("ecdm_error_rate_percent", "Error rate percent")
DEPENDENCY_GAUGE = Gauge("ecdm_dependency_available", "1 if dependency reachable else 0")
LATENCY_HIST = Histogram(
    "ecdm_request_latency_seconds",
    "Request latency",
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10),
)

_PROCESS = psutil.Process()
_PROCESS.cpu_percent() # Праймінг-виклик для ініціалізації заміру


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

    if latency_ms > 0:
        time.sleep(latency_ms / 1000.0)

    if dependency_failure:
        return jsonify(error="dependency unavailable"), 503

    if random.random() < error_probability:
        return jsonify(error="simulated failure"), 500

    return jsonify(status="ok")


@app.get("/metrics")
def metrics():
    with state_lock:
        error_probability = fault_state["error_probability"]
        dependency_failure = fault_state["dependency_failure"]

    CPU_GAUGE.set(_PROCESS.cpu_percent())
    MEMORY_GAUGE.set(psutil.virtual_memory().percent)
    MEMORY_MB_GAUGE.set(_PROCESS.memory_info().rss / (1024 * 1024))
    ERROR_RATE_GAUGE.set(error_probability * 100)
    DEPENDENCY_GAUGE.set(0 if dependency_failure else 1)

    return generate_latest(), 200, {"Content-Type": CONTENT_TYPE_LATEST}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, threaded=True)