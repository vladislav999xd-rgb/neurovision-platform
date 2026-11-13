"""Проверка медицинского dashboard и звука каждой кнопки."""

from pathlib import Path
import sys

from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def find_button(app, text):
    return next(button for button in app.button if text in button.label)


def audio_urls(app):
    elements = app.get("audio")
    assert len(elements) == 2
    assert elements[0].proto.autoplay is True
    assert elements[0].proto.loop is True
    assert elements[1].proto.autoplay is True
    return elements[0].proto.url, elements[1].proto.url


app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
assert len(app.exception) == 0

page_text = " ".join(markdown.value for markdown in app.markdown)
assert "Забота начинается" in page_text
assert "Состояние кабинетов" in page_text
assert "Три шутки клиники" in page_text

initial_audio = app.get("audio")
assert len(initial_audio) == 1
assert initial_audio[0].proto.autoplay is True
assert initial_audio[0].proto.loop is True
ambient_url = initial_audio[0].proto.url

previous_quack = None
buttons = [
    "Следующая шутка",
    "Вернуть исходные данные",
    "Вызвать следующего",
    "Сбросить",
    "Завершить обращение",
    "Отменить талон",
    "Создать талон",
]

for label in buttons:
    find_button(app, label).click()
    app.run()
    assert len(app.exception) == 0
    current_ambient, current_quack = audio_urls(app)
    assert current_ambient == ambient_url
    if previous_quack is not None:
        assert current_quack != previous_quack
    previous_quack = current_quack

print("Медицинский dashboard: пройдено")
print("Фоновый звук стабилен и зациклен: пройдено")
print("Каждая из семи кнопок получила новое кряканье: пройдено")
