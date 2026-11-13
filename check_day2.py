"""Десять проверок командной работы второго дня."""

import sys

from student_logic import (
    call_next,
    find_first_waiting,
    find_ticket,
    finish_ticket,
    make_initial_extra,
    make_initial_records,
    next_queue_order,
    register_ticket,
    transfer_ticket,
)


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def copy_tickets(tickets):
    """Сделать независимый снимок списка простых словарей."""
    return [ticket.copy() for ticket in tickets]


def fresh_data():
    """Вернуть свежие records и extra для независимой проверки."""
    tickets = make_initial_records()
    cabinets = make_initial_extra()["cabinets"]
    return tickets, cabinets


# 1. Минимальный waiting в кабинете 101, список не меняется.
tickets, cabinets = fresh_data()
before = copy_tickets(tickets)
found = find_first_waiting(tickets, "101")
assert found is tickets[0]
assert found["id"] == 1
assert tickets == before
print("1. Первый ожидающий в кабинете 101: пройдено")

# 2. Перенос id1 в 103 ставит его после id6.
tickets, cabinets = fresh_data()
result = transfer_ticket(tickets, cabinets, 1, "103")
moved = find_ticket(tickets, 1)
assert result["ok"] is True
assert result["record_id"] == 1
assert moved == {
    "id": 1,
    "visitor_code": "P101",
    "cabinet": "103",
    "status": "waiting",
    "queue_order": 7,
}
assert find_ticket(tickets, 6)["queue_order"] == 6
print("2. Перенос id1 в кабинет 103: пройдено")

# 3. В очередь занятого кабинета вставать разрешено.
tickets, cabinets = fresh_data()
result = transfer_ticket(tickets, cabinets, 1, "102")
assert result["ok"] is True
assert find_ticket(tickets, 1)["cabinet"] == "102"
assert find_ticket(tickets, 1)["status"] == "waiting"
assert find_ticket(tickets, 3)["status"] == "in_service"
print("3. Перенос в занятый обслуживанием кабинет: пройдено")

# 4. Талон in_service переносить нельзя.
tickets, cabinets = fresh_data()
before = copy_tickets(tickets)
result = transfer_ticket(tickets, cabinets, 3, "103")
assert result["ok"] is False
assert result["record_id"] is None
assert tickets == before
print("4. Отказ переноса талона на приёме: пройдено")

# 5. Перенос в прежний кабинет ничего не меняет.
tickets, cabinets = fresh_data()
before = copy_tickets(tickets)
result = transfer_ticket(tickets, cabinets, 1, "101")
assert result["ok"] is False
assert tickets == before
assert next_queue_order(tickets) == 7
print("5. Отказ переноса в тот же кабинет: пройдено")

# 6. Неизвестная цель отклоняется без изменения списка.
tickets, cabinets = fresh_data()
before = copy_tickets(tickets)
result = transfer_ticket(tickets, cabinets, 1, "999")
assert result["ok"] is False
assert tickets == before
print("6. Отказ для неизвестного кабинета: пройдено")

# 7. Дубликат активного посетителя в целевом кабинете запрещён.
tickets, cabinets = fresh_data()
registered = register_ticket(tickets, cabinets, "P101", "103")
assert registered["ok"] is True
assert registered["record_id"] == 7
before_transfer = copy_tickets(tickets)
result = transfer_ticket(tickets, cabinets, 1, "103")
assert result["ok"] is False
assert tickets == before_transfer
assert find_ticket(tickets, 1)["cabinet"] == "101"
print("7. Конфликт активного талона посетителя: пройдено")

# 8. Вызов определяется queue_order, а не моментом переноса.
tickets, cabinets = fresh_data()
assert transfer_ticket(tickets, cabinets, 1, "103")["ok"] is True
first_call = call_next(tickets, cabinets, "103")
assert first_call["ok"] is True
assert first_call["record_id"] == 6
assert finish_ticket(tickets, 6)["ok"] is True
second_call = call_next(tickets, cabinets, "103")
assert second_call["ok"] is True
assert second_call["record_id"] == 1
print("8. Перенос, вызов id6, затем вызов id1: пройдено")

# 9. Физический порядок словарей не влияет на поиск минимума.
tickets, cabinets = fresh_data()
tickets[:] = [tickets[5], tickets[2], tickets[1], tickets[4], tickets[0], tickets[3]]
before = copy_tickets(tickets)
found = find_first_waiting(tickets, "101")
assert found["id"] == 1
assert tickets == before
print("9. Поиск после перестановки словарей: пройдено")

# 10. Граничный случай пустого списка.
tickets = []
assert find_first_waiting(tickets, "101") is None
assert next_queue_order(tickets) == 1
assert tickets == []
print("10. Пустой список: пройдено")

print("Все проверки второго дня пройдены.")

