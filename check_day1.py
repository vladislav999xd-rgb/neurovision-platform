"""Восемь независимых проверок и один полный сценарий первого дня."""

import sys

from student_logic import (
    call_next,
    cancel_ticket,
    find_ticket,
    finish_ticket,
    make_initial_extra,
    make_initial_records,
    register_ticket,
    select_tickets,
)


# Терминал Windows может оставить для Python кодировку cp1251 даже после
# команды chcp 65001. Явно включаем UTF-8, чтобы русские буквы не пропадали.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

CABINETS = make_initial_extra()["cabinets"]


def copy_tickets(tickets):
    """Сделать снимок списка, чтобы проверить отсутствие изменений."""
    copied = []
    for ticket in tickets:
        copied.append(ticket.copy())
    return copied


# 1. Успешная регистрация.
tickets = make_initial_records()
result = register_ticket(tickets, CABINETS, " p107 ", "101")
assert result == {"ok": True, "message": "Талон создан.", "record_id": 7}
assert len(tickets) == 7
assert tickets[6] == {
    "id": 7,
    "visitor_code": "P107",
    "cabinet": "101",
    "status": "waiting",
    "queue_order": 7,
}
print("1. Успешная регистрация: пройдено")

# 2. Отказ при пустом коде без изменения данных.
tickets = make_initial_records()
before = copy_tickets(tickets)
result = register_ticket(tickets, CABINETS, "   ", "101")
assert result["ok"] is False
assert result["record_id"] is None
assert tickets == before
print("2. Пустой код: пройдено")

# 3. Вызов первого ожидающего в кабинет 101.
tickets = make_initial_records()
before = copy_tickets(tickets)
result = call_next(tickets, CABINETS, "101")
before[0]["status"] = "in_service"
assert result["ok"] is True
assert result["record_id"] == 1
assert tickets == before
assert find_ticket(tickets, 2)["status"] == "waiting"
print("3. Вызов талона 1: пройдено")

# 4. Отказ при вызове в занятый кабинет 102.
tickets = make_initial_records()
before = copy_tickets(tickets)
result = call_next(tickets, CABINETS, "102")
assert result["ok"] is False
assert result["record_id"] is None
assert tickets == before
print("4. Занятый кабинет: пройдено")

# 5. Успешное завершение талона 3.
tickets = make_initial_records()
before = copy_tickets(tickets)
result = finish_ticket(tickets, 3)
before[2]["status"] = "done"
assert result["ok"] is True
assert result["record_id"] == 3
assert tickets == before
print("5. Завершение талона 3: пройдено")

# 6. Отказ при повторном завершении талона 4.
tickets = make_initial_records()
before = copy_tickets(tickets)
result = finish_ticket(tickets, 4)
assert result["ok"] is False
assert result["record_id"] is None
assert tickets == before
print("6. Повторное завершение: пройдено")

# 7. Отмена ожидающего талона 2.
tickets = make_initial_records()
before = copy_tickets(tickets)
result = cancel_ticket(tickets, 2)
before[1]["status"] = "canceled"
assert result["ok"] is True
assert result["record_id"] == 2
assert tickets == before
assert find_ticket(tickets, 2)["queue_order"] == 2
print("7. Отмена талона 2: пройдено")

# 8. Одновременная работа трёх фильтров без изменения данных.
tickets = make_initial_records()
before = copy_tickets(tickets)
selected = select_tickets(tickets, "101", "waiting", "p10")
selected_ids = []
for ticket in selected:
    selected_ids.append(ticket["id"])
assert selected_ids == [1, 2]
assert tickets == before
print("8. Фильтрация: пройдено")

# Полный сценарий: новый талон проходит кабинет 102 до завершения.
tickets = make_initial_records()
assert register_ticket(tickets, CABINETS, "P107", "102")["ok"] is True
assert finish_ticket(tickets, 3)["ok"] is True
called = call_next(tickets, CABINETS, "102")
assert called["ok"] is True
assert called["record_id"] == 7
assert finish_ticket(tickets, 7)["ok"] is True
assert find_ticket(tickets, 3)["status"] == "done"
assert find_ticket(tickets, 7)["status"] == "done"
print("Полный сценарий P107 в кабинете 102: пройдено")

print("Все проверки первого дня пройдены.")

