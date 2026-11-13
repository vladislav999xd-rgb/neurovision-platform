"""Восемь независимых проверок регистратуры четвёртого дня."""

from copy import deepcopy
import sys

from student_logic import (
    call_next_logged,
    find_ticket,
    finish_ticket_logged,
    make_initial_extra,
    make_initial_records,
    requeue_ticket,
)


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def fresh_state():
    """Каждая независимая проверка начинает с исходных данных."""
    return {
        "records": make_initial_records(),
        "history": [],
        "extra": make_initial_extra(),
    }


# 1. Исходный canceled № 5 возвращается с тем же id и новым порядком 7.
state = fresh_state()
result = requeue_ticket(state, 5)
ticket = find_ticket(state["records"], 5)
assert result == {
    "ok": True,
    "message": "Талон возвращён в конец очереди.",
    "record_id": 5,
}
assert ticket["id"] == 5
assert ticket["cabinet"] == "101"
assert ticket["status"] == "waiting"
assert ticket["queue_order"] == 7
assert len(state["history"]) == 1
event = state["history"][0]
assert event["action"] == "requeue"
assert event["record_id"] == 5
assert event["field"] == "status"
assert event["before"] == "canceled"
assert event["after"] == "waiting"
assert event["related_ids"] == [5]
assert event["details"] == {"old_queue_order": 5, "new_queue_order": 7}
print("1. Возврат отменённого талона 5: пройдено")

# 2. Waiting возвращать нельзя, весь state остаётся прежним.
state = fresh_state()
before = deepcopy(state)
result = requeue_ticket(state, 1)
assert result["ok"] is False
assert state == before
print("2. Отказ возврата waiting-талона 1: пройдено")

# 3. In-service возвращать нельзя.
state = fresh_state()
before = deepcopy(state)
result = requeue_ticket(state, 3)
assert result["ok"] is False
assert state == before
print("3. Отказ возврата талона 3 на приёме: пройдено")

# 4. Done возвращать нельзя.
state = fresh_state()
before = deepcopy(state)
result = requeue_ticket(state, 4)
assert result["ok"] is False
assert state == before
print("4. Отказ возврата завершённого талона 4: пройдено")

# 5. Повторный клик не создаёт порядок 8 и второе событие.
state = fresh_state()
first = requeue_ticket(state, 5)
after_first = deepcopy(state)
second = requeue_ticket(state, 5)
assert first["ok"] is True
assert second["ok"] is False
assert state == after_first
assert find_ticket(state["records"], 5)["queue_order"] == 7
assert len(state["history"]) == 1
print("5. Повторный возврат без второго изменения: пройдено")

# 6. Неизвестный номер не добавляет записи и событие.
state = fresh_state()
before = deepcopy(state)
result = requeue_ticket(state, 99)
assert result["ok"] is False
assert state == before
print("6. Неизвестный талон 99: пройдено")

# 7. Другой активный талон P105 в том же кабинете блокирует возврат.
state = fresh_state()
state["records"].append(
    {
        "id": 7,
        "visitor_code": "P105",
        "cabinet": "101",
        "status": "waiting",
        "queue_order": 7,
    }
)
before = deepcopy(state)
result = requeue_ticket(state, 5)
assert result["ok"] is False
assert "активный талон" in result["message"]
assert state == before
assert find_ticket(state["records"], 5)["status"] == "canceled"
print("7. Активный дубль P105 блокирует возврат: пройдено")

# 8. Возвращённый талон идёт после прежних ожидающих 1 и 2.
state = fresh_state()
assert requeue_ticket(state, 5)["ok"] is True
first = call_next_logged(state, "101")
assert first["record_id"] == 1
assert finish_ticket_logged(state, 1)["ok"] is True
second = call_next_logged(state, "101")
assert second["record_id"] == 2
assert finish_ticket_logged(state, 2)["ok"] is True
third = call_next_logged(state, "101")
assert third["record_id"] == 5
assert find_ticket(state["records"], 5)["id"] == 5
print("8. Очередь вызова 1 → 2 → 5: пройдено")

print("Все проверки четвёртого дня пройдены.")
