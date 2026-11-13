"""Восемь независимых проверок регистратуры третьего дня."""

from copy import deepcopy
from pathlib import Path
import sys
import tempfile

from io_adapter import load_state, save_state
from student_logic import (
    call_next_logged,
    cancel_ticket_logged,
    find_ticket,
    finish_ticket_logged,
    transfer_ticket_logged,
    validate_project_state,
)


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


ROOT = Path(__file__).resolve().parent
QUEUE_FIXTURE = ROOT / "fixtures" / "queue.json"
STOCK_FIXTURE = ROOT / "fixtures" / "stock.json"


def fresh_state():
    """Каждая проверка получает отдельный исходный снимок."""
    loaded = load_state(QUEUE_FIXTURE, "queue")
    assert loaded["ok"] is True
    assert validate_project_state(loaded["state"]) == []
    return loaded["state"]


# 1. Успешный вызов создаёт ровно одно событие call.
state = fresh_state()
result = call_next_logged(state, "101")
assert result["ok"] is True
assert result["record_id"] == 1
assert find_ticket(state["records"], 1)["status"] == "in_service"
assert len(state["history"]) == 1
event = state["history"][0]
assert event["event_id"] == 1
assert event["action"] == "call"
assert event["record_id"] == 1
assert event["field"] == "status"
assert event["before"] == "waiting"
assert event["after"] == "in_service"
print("1. Вызов в кабинет 101 и событие call: пройдено")

# 2. Отказ занятого кабинета не меняет ни одну часть state.
state = fresh_state()
before = deepcopy(state)
result = call_next_logged(state, "102")
assert result["ok"] is False
assert state == before
assert state["history"] == []
print("2. Отказ вызова в занятый кабинет: пройдено")

# 3. Отмена waiting-талона создаёт cancel, запись остаётся в списке.
state = fresh_state()
result = cancel_ticket_logged(state, 2)
assert result["ok"] is True
assert find_ticket(state["records"], 2)["status"] == "canceled"
assert len(state["records"]) == 6
assert len(state["history"]) == 1
event = state["history"][0]
assert event["action"] == "cancel"
assert event["record_id"] == 2
assert event["before"] == "waiting"
assert event["after"] == "canceled"
print("3. Отмена талона 2 и событие cancel: пройдено")

# 4. Уже завершённый талон нельзя завершить повторно.
state = fresh_state()
before = deepcopy(state)
result = finish_ticket_logged(state, 4)
assert result["ok"] is False
assert state == before
assert state["history"] == []
print("4. Отказ повторного завершения талона 4: пройдено")

# 5. Перенос хранит отдельные снимки кабинета и порядка.
state = fresh_state()
result = transfer_ticket_logged(state, 2, "103")
assert result["ok"] is True
ticket_2 = find_ticket(state["records"], 2)
assert ticket_2["id"] == 2
assert ticket_2["status"] == "waiting"
assert ticket_2["cabinet"] == "103"
assert ticket_2["queue_order"] == 7
assert len(state["history"]) == 1
event = state["history"][0]
assert event["action"] == "move"
assert event["before"] == {"cabinet": "101", "queue_order": 2}
assert event["after"] == {"cabinet": "103", "queue_order": 7}
print("5. Перенос талона 2 и событие move: пройдено")

# 6. Повторный вызов занятого кабинета не добавляет событие.
state = fresh_state()
first = call_next_logged(state, "101")
history_after_first = deepcopy(state["history"])
second = call_next_logged(state, "101")
assert first["ok"] is True
assert second["ok"] is False
assert find_ticket(state["records"], 2)["status"] == "waiting"
assert state["history"] == history_after_first
assert len(state["history"]) == 1
print("6. Повторный вызов без ложного события: пройдено")

# 7. Сохранение, восстановление и следующее действие продолжают историю.
state = fresh_state()
assert call_next_logged(state, "101")["ok"] is True
with tempfile.TemporaryDirectory() as temp_directory:
    path = Path(temp_directory) / "team_state.json"
    before_save = deepcopy(state)
    saved = save_state(path, "queue", state)
    assert saved["ok"] is True
    assert state == before_save

    loaded = load_state(path, "queue")
    assert loaded["ok"] is True
    restored = loaded["state"]
    assert restored == before_save
    assert validate_project_state(restored) == []

    finished = finish_ticket_logged(restored, 1)
    assert finished["ok"] is True
    assert find_ticket(restored["records"], 1)["status"] == "done"
    assert len(restored["history"]) == 2
    assert restored["history"][1]["event_id"] == 2
    assert restored["history"][1]["action"] == "finish"
    assert restored["history"][1]["before"] == "in_service"
    assert restored["history"][1]["after"] == "done"
print("7. Сохранение, восстановление и завершение: пройдено")

# 8. Файл другого проекта отклоняется до замены текущего state.
state = fresh_state()
before = deepcopy(state)
loaded = load_state(STOCK_FIXTURE, "queue")
assert loaded["ok"] is False
assert loaded["state"] is None
assert "другому проекту" in loaded["message"]
assert state == before
print("8. Отказ загрузки stock.json как queue: пройдено")

print("Все проверки третьего дня пройдены.")

