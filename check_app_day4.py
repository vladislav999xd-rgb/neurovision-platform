"""Проверка действия дня 4 в интерфейсе и звука всех кнопок."""

from copy import deepcopy
from pathlib import Path
import sys

from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parent

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

app_source = (ROOT / "app.py").read_text(encoding="utf-8")
assert "st.rerun()" not in app_source


def find_button(app, text):
    return next(button for button in app.button if text in button.label)


# Новая кнопка меняет талон 5 один раз и не меняет при повторном клике.
app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
assert len(app.exception) == 0
ambient_url = app.get("audio")[0].proto.url

find_button(app, "Вернуть в очередь").click()
app.run()
assert len(app.exception) == 0
state = app.session_state["state"]
ticket_5 = next(ticket for ticket in state["records"] if ticket["id"] == 5)
assert ticket_5["status"] == "waiting"
assert ticket_5["queue_order"] == 7
assert len(state["history"]) == 1
assert state["history"][0]["action"] == "requeue"
first_quack_url = app.get("audio")[1].proto.url

before_repeat = deepcopy(state)
find_button(app, "Вернуть в очередь").click()
app.run()
assert len(app.exception) == 0
assert app.session_state["state"] == before_repeat
assert app.get("audio")[0].proto.url == ambient_url
assert app.get("audio")[1].proto.url != first_quack_url
print("Кнопка возврата и повторный отказ: пройдено")


# Каждая кнопка получает новый звук, фон остаётся тем же и зациклен.
app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
assert len(app.exception) == 0
ambient = app.get("audio")[0]
assert ambient.proto.autoplay is True
assert ambient.proto.loop is True
ambient_url = ambient.proto.url
previous_quack_url = None

button_labels = [
    "Создать талон",
    "Вызвать следующего",
    "Перенести талон",
    "Вернуть в очередь",
    "Сбросить",
    "Завершить обращение",
    "Отменить талон",
    "Сохранить",
    "Восстановить",
    "Выгрузить",
    "Следующая шутка",
    "Вернуть исходные данные",
]

for label in button_labels:
    find_button(app, label).click()
    app.run()
    assert len(app.exception) == 0
    audio = app.get("audio")
    assert len(audio) == 2
    assert audio[0].proto.url == ambient_url
    assert audio[0].proto.loop is True
    if previous_quack_url is not None:
        assert audio[1].proto.url != previous_quack_url
    previous_quack_url = audio[1].proto.url

print("Звук всех двенадцати кнопок: пройдено")
print("Интерфейс четвёртого дня проверен.")
