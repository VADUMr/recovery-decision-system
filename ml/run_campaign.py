import requests

print(">>> Ну шо погнали")
try:
    # Встановлюємо великий timeout (наприклад, 40 хвилин = 2400 секунд)
    response = requests.post("http://localhost:8100/campaign", timeout=2400)
    print(">>> Кампанія успішно завершена!")
    print(response.json())
except requests.exceptions.Timeout:
    print(">>> Помилка: перевищено час очікування (але кампанія може ще виконуватися в контейнері).")
except Exception as e:
    print(f">>> Сталася помилка: {e}")