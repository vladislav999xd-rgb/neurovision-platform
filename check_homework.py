"""Четыре независимые проверки домашней функции room_message."""

import sys

from student_logic import make_initial_records, room_message


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def copy_tickets(tickets):
    """Сделать снимок данных для проверки отсутствия изменений."""
    copied = []
    for ticket in tickets:
        copied.append(ticket.copy())
    return copied


tickets = make_initial_records()
before = copy_tickets(tickets)
answer = room_message(tickets, "101")
assert answer == "Кабинет 101: ожидают 2; свободен"
assert tickets == before
print("1. Кабинет 101: пройдено")

tickets = make_initial_records()
before = copy_tickets(tickets)
answer = room_message(tickets, "102")
assert answer == "Кабинет 102: ожидают 0; занят"
assert tickets == before
print("2. Кабинет 102: пройдено")

tickets = make_initial_records()
before = copy_tickets(tickets)
answer = room_message(tickets, "999")
assert answer == "Неизвестный кабинет"
assert tickets == before
print("3. Неизвестный кабинет: пройдено")

tickets = make_initial_records()
before = copy_tickets(tickets)
answer = room_message(tickets, "103")
assert answer == "Кабинет 103: ожидают 1; свободен"
assert tickets == before
print("4. Собственный пример для кабинета 103: пройдено")

print("Все проверки домашней работы пройдены.")
