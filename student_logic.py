"""Предметная логика учебной электронной регистратуры.

Файл не зависит от Streamlit: функции получают обычные списки и словари.
Второй день добавляет общий справочник кабинетов, переиспользуемые проверки,
поиск минимального порядка очереди и перенос ожидающего талона.
Четвёртый день добавляет возврат отменённого талона в конец очереди.
"""

from history_tools import append_event


def make_initial_records():
    """Вернуть новый независимый список из шести исходных талонов."""
    tickets = [
        {
            "id": 1,
            "visitor_code": "P101",
            "cabinet": "101",
            "status": "waiting",
            "queue_order": 1,
        },
        {
            "id": 2,
            "visitor_code": "P102",
            "cabinet": "101",
            "status": "waiting",
            "queue_order": 2,
        },
        {
            "id": 3,
            "visitor_code": "P103",
            "cabinet": "102",
            "status": "in_service",
            "queue_order": 3,
        },
        {
            "id": 4,
            "visitor_code": "P104",
            "cabinet": "103",
            "status": "done",
            "queue_order": 4,
        },
        {
            "id": 5,
            "visitor_code": "P105",
            "cabinet": "101",
            "status": "canceled",
            "queue_order": 5,
        },
        {
            "id": 6,
            "visitor_code": "P106",
            "cabinet": "103",
            "status": "waiting",
            "queue_order": 6,
        },
    ]
    return tickets


def make_initial_extra():
    """Вернуть новый словарь дополнительных данных второго дня."""
    return {"cabinets": ["101", "102", "103"]}


def find_ticket(tickets, ticket_id):
    """Найти исходный словарь талона по id или вернуть None."""
    for ticket in tickets:
        if ticket["id"] == ticket_id:
            return ticket
    return None


def normalize_visitor_code(value):
    """Убрать внешние пробелы и привести код посетителя к верхнему регистру."""
    return value.strip().upper()


def validate_registration(
    tickets,
    cabinets,
    visitor_code,
    cabinet,
    excluded_ticket_id=None,
):
    """Вернуть все ошибки регистрации или переноса, не меняя tickets.

    excluded_ticket_id равен None при новой регистрации. При переносе сюда
    передаётся id переносимого талона, чтобы он не считался своим же дублем.
    """
    errors = []
    normalized_code = normalize_visitor_code(visitor_code)
    normalized_cabinet = cabinet.strip()

    code_is_valid = (
        len(normalized_code) == 4
        and normalized_code[0] == "P"
        and normalized_code[1:].isdigit()
    )
    if not code_is_valid:
        errors.append("Код посетителя должен состоять из P и трёх цифр.")

    if normalized_cabinet not in cabinets:
        errors.append("Неизвестный кабинет.")

    for ticket in tickets:
        if ticket["id"] == excluded_ticket_id:
            continue
        same_visitor = ticket["visitor_code"] == normalized_code
        same_cabinet = ticket["cabinet"] == normalized_cabinet
        active_ticket = ticket["status"] in ["waiting", "in_service"]
        if same_visitor and same_cabinet and active_ticket:
            errors.append(
                "У посетителя уже есть активный талон в этом кабинете."
            )
            break

    return errors


def next_queue_order(tickets):
    """Вернуть следующий глобальный порядок очереди, не меняя список."""
    largest_order = 0
    for ticket in tickets:
        if ticket["queue_order"] > largest_order:
            largest_order = ticket["queue_order"]
    return largest_order + 1


def find_first_waiting(tickets, cabinet):
    """Найти waiting-талон кабинета с минимальным queue_order.

    Возвращается исходный словарь из tickets. Список не сортируется и порядок
    словарей в нём не меняется.
    """
    first_waiting = None
    for ticket in tickets:
        waiting_here = ticket["cabinet"] == cabinet
        waiting_here = waiting_here and ticket["status"] == "waiting"
        if not waiting_here:
            continue
        if first_waiting is None:
            first_waiting = ticket
        elif ticket["queue_order"] < first_waiting["queue_order"]:
            first_waiting = ticket
    return first_waiting


def register_ticket(tickets, cabinets, visitor_code, cabinet):
    """Проверить данные общей функцией и добавить ожидающий талон."""
    normalized_code = normalize_visitor_code(visitor_code)
    normalized_cabinet = cabinet.strip()
    errors = validate_registration(
        tickets,
        cabinets,
        normalized_code,
        normalized_cabinet,
        excluded_ticket_id=None,
    )
    if errors:
        return {
            "ok": False,
            "message": " ".join(errors),
            "record_id": None,
        }

    new_id = 1
    for ticket in tickets:
        if ticket["id"] >= new_id:
            new_id = ticket["id"] + 1

    new_ticket = {
        "id": new_id,
        "visitor_code": normalized_code,
        "cabinet": normalized_cabinet,
        "status": "waiting",
        "queue_order": next_queue_order(tickets),
    }
    tickets.append(new_ticket)
    return {
        "ok": True,
        "message": "Талон создан.",
        "record_id": new_id,
    }


def call_next(tickets, cabinets, cabinet):
    """Вызвать первого ожидающего, если кабинет известен и свободен."""
    normalized_cabinet = cabinet.strip()
    if normalized_cabinet not in cabinets:
        return {
            "ok": False,
            "message": "Неизвестный кабинет.",
            "record_id": None,
        }

    for ticket in tickets:
        cabinet_is_busy = ticket["cabinet"] == normalized_cabinet
        cabinet_is_busy = cabinet_is_busy and ticket["status"] == "in_service"
        if cabinet_is_busy:
            return {
                "ok": False,
                "message": "Кабинет занят.",
                "record_id": None,
            }

    first_waiting = find_first_waiting(tickets, normalized_cabinet)
    if first_waiting is None:
        return {
            "ok": False,
            "message": "В кабинете нет ожидающих.",
            "record_id": None,
        }

    first_waiting["status"] = "in_service"
    return {
        "ok": True,
        "message": "Посетитель вызван.",
        "record_id": first_waiting["id"],
    }


def transfer_ticket(tickets, cabinets, ticket_id, target_cabinet):
    """Перенести waiting-талон в конец очереди другого кабинета.

    Все проверки выполняются до изменения данных. При успехе меняются только
    cabinet и queue_order; номер, код посетителя и статус сохраняются.
    """
    ticket = find_ticket(tickets, ticket_id)
    if ticket is None:
        return {
            "ok": False,
            "message": "Талон не найден.",
            "record_id": None,
        }

    normalized_target = target_cabinet.strip()
    errors = []

    if ticket["status"] != "waiting":
        errors.append("Перенести можно только ожидающий талон.")

    if normalized_target == ticket["cabinet"]:
        errors.append("Новый кабинет должен отличаться от текущего.")

    registration_errors = validate_registration(
        tickets,
        cabinets,
        ticket["visitor_code"],
        normalized_target,
        excluded_ticket_id=ticket["id"],
    )
    errors.extend(registration_errors)

    if errors:
        return {
            "ok": False,
            "message": " ".join(errors),
            "record_id": None,
        }

    new_order = next_queue_order(tickets)
    ticket["cabinet"] = normalized_target
    ticket["queue_order"] = new_order
    return {
        "ok": True,
        "message": "Талон перенесён в кабинет " + normalized_target + ".",
        "record_id": ticket["id"],
    }


def finish_ticket(tickets, ticket_id):
    """Завершить талон, только если посетитель уже вызван."""
    ticket = find_ticket(tickets, ticket_id)
    if ticket is None:
        return {
            "ok": False,
            "message": "Талон не найден.",
            "record_id": None,
        }
    if ticket["status"] != "in_service":
        return {
            "ok": False,
            "message": "Завершить можно только вызванный талон.",
            "record_id": None,
        }

    ticket["status"] = "done"
    return {
        "ok": True,
        "message": "Обращение завершено.",
        "record_id": ticket_id,
    }


def cancel_ticket(tickets, ticket_id):
    """Отменить талон, только пока он ожидает вызова."""
    ticket = find_ticket(tickets, ticket_id)
    if ticket is None:
        return {
            "ok": False,
            "message": "Талон не найден.",
            "record_id": None,
        }
    if ticket["status"] != "waiting":
        return {
            "ok": False,
            "message": "Отменить можно только ожидающий талон.",
            "record_id": None,
        }

    ticket["status"] = "canceled"
    return {
        "ok": True,
        "message": "Талон отменён.",
        "record_id": ticket_id,
    }


def select_tickets(tickets, cabinet, status, fragment):
    """Вернуть новый список по всем непустым условиям без изменения tickets."""
    cabinet = cabinet.strip()
    status = status.strip()
    fragment = fragment.strip().lower()
    selected = []

    for ticket in tickets:
        if cabinet != "" and ticket["cabinet"] != cabinet:
            continue
        if status != "" and ticket["status"] != status:
            continue
        if fragment not in ticket["visitor_code"].lower():
            continue
        selected.append(ticket)

    return selected


def room_message(tickets, cabinet):
    """Вернуть строку о числе ожидающих и занятости кабинета."""
    cabinet = cabinet.strip()
    if cabinet not in ["101", "102", "103"]:
        return "Неизвестный кабинет"

    waiting_count = 0
    occupied = False
    for ticket in tickets:
        if ticket["cabinet"] != cabinet:
            continue
        if ticket["status"] == "waiting":
            waiting_count += 1
        if ticket["status"] == "in_service":
            occupied = True

    room_state = "занят" if occupied else "свободен"
    return (
        "Кабинет "
        + cabinet
        + ": ожидают "
        + str(waiting_count)
        + "; "
        + room_state
    )


def call_next_logged(state, cabinet):
    """Вызвать следующего и записать call только при успехе."""
    tickets = state["records"]
    cabinets = state["extra"]["cabinets"]
    candidate = find_first_waiting(tickets, cabinet.strip())
    before_status = None
    if candidate is not None:
        before_status = candidate["status"]

    result = call_next(tickets, cabinets, cabinet)
    if not result["ok"]:
        return result

    called_ticket = find_ticket(tickets, result["record_id"])
    append_event(
        state,
        action="call",
        record_id=called_ticket["id"],
        field="status",
        before=before_status,
        after=called_ticket["status"],
        related_ids=[],
        details={"cabinet": called_ticket["cabinet"]},
    )
    return result


def cancel_ticket_logged(state, ticket_id):
    """Отменить waiting-талон и записать cancel только при успехе."""
    ticket = find_ticket(state["records"], ticket_id)
    before_status = None
    if ticket is not None:
        before_status = ticket["status"]

    result = cancel_ticket(state["records"], ticket_id)
    if not result["ok"]:
        return result

    changed_ticket = find_ticket(state["records"], result["record_id"])
    append_event(
        state,
        action="cancel",
        record_id=changed_ticket["id"],
        field="status",
        before=before_status,
        after=changed_ticket["status"],
        related_ids=[],
        details={"cabinet": changed_ticket["cabinet"]},
    )
    return result


def finish_ticket_logged(state, ticket_id):
    """Завершить вызванный талон и записать finish только при успехе."""
    ticket = find_ticket(state["records"], ticket_id)
    before_status = None
    if ticket is not None:
        before_status = ticket["status"]

    result = finish_ticket(state["records"], ticket_id)
    if not result["ok"]:
        return result

    changed_ticket = find_ticket(state["records"], result["record_id"])
    append_event(
        state,
        action="finish",
        record_id=changed_ticket["id"],
        field="status",
        before=before_status,
        after=changed_ticket["status"],
        related_ids=[],
        details={"cabinet": changed_ticket["cabinet"]},
    )
    return result


def transfer_ticket_logged(state, ticket_id, target_cabinet):
    """Перенести талон и записать прежние и новые кабинет с порядком."""
    ticket = find_ticket(state["records"], ticket_id)
    before = None
    if ticket is not None:
        before = {
            "cabinet": ticket["cabinet"],
            "queue_order": ticket["queue_order"],
        }

    result = transfer_ticket(
        state["records"],
        state["extra"]["cabinets"],
        ticket_id,
        target_cabinet,
    )
    if not result["ok"]:
        return result

    changed_ticket = find_ticket(state["records"], result["record_id"])
    after = {
        "cabinet": changed_ticket["cabinet"],
        "queue_order": changed_ticket["queue_order"],
    }
    append_event(
        state,
        action="move",
        record_id=changed_ticket["id"],
        field="cabinet_and_queue_order",
        before=before,
        after=after,
        related_ids=[],
        details={
            "visitor_code": changed_ticket["visitor_code"],
            "status": changed_ticket["status"],
        },
    )
    return result


def requeue_ticket(state, ticket_id):
    """Вернуть отменённый талон в конец очереди того же кабинета.

    Все проверки выполняются до изменения данных. При успехе сохраняются id,
    visitor_code и cabinet, статус становится waiting, а queue_order получает
    следующее глобальное значение. В журнал добавляется ровно одно событие.
    """
    tickets = state["records"]
    cabinets = state["extra"]["cabinets"]
    ticket = find_ticket(tickets, ticket_id)

    if ticket is None:
        return {
            "ok": False,
            "message": "Талон не найден.",
            "record_id": None,
        }

    if ticket["status"] != "canceled":
        return {
            "ok": False,
            "message": "Вернуть в очередь можно только отменённый талон.",
            "record_id": None,
        }

    errors = validate_registration(
        tickets,
        cabinets,
        ticket["visitor_code"],
        ticket["cabinet"],
        excluded_ticket_id=ticket["id"],
    )
    if errors:
        return {
            "ok": False,
            "message": " ".join(errors),
            "record_id": None,
        }

    old_queue_order = ticket["queue_order"]
    new_queue_order = next_queue_order(tickets)
    ticket["status"] = "waiting"
    ticket["queue_order"] = new_queue_order

    append_event(
        state,
        action="requeue",
        record_id=ticket["id"],
        field="status",
        before="canceled",
        after="waiting",
        related_ids=[ticket["id"]],
        details={
            "old_queue_order": old_queue_order,
            "new_queue_order": new_queue_order,
        },
    )
    return {
        "ok": True,
        "message": "Талон возвращён в конец очереди.",
        "record_id": ticket["id"],
    }


def validate_project_state(candidate):
    """Проверить загруженное состояние регистратуры и вернуть список ошибок."""
    errors = []
    if not isinstance(candidate, dict):
        return ["Состояние должно быть словарём."]

    tickets = candidate.get("records")
    history = candidate.get("history")
    extra = candidate.get("extra")
    if not isinstance(tickets, list):
        errors.append("records должен быть списком.")
    if not isinstance(history, list):
        errors.append("history должен быть списком.")
    if not isinstance(extra, dict):
        errors.append("extra должен быть словарём.")
    if errors:
        return errors

    cabinets = extra.get("cabinets")
    cabinets_are_valid = (
        isinstance(cabinets, list)
        and len(cabinets) > 0
        and all(isinstance(cabinet, str) for cabinet in cabinets)
        and len(cabinets) == len(set(cabinets))
    )
    if not cabinets_are_valid:
        errors.append("В extra нужен непустой список уникальных кабинетов.")
        cabinets = []

    valid_statuses = ["waiting", "in_service", "done", "canceled"]
    ticket_ids = set()
    queue_orders = set()
    busy_cabinets = set()

    for position, ticket in enumerate(tickets, start=1):
        if not isinstance(ticket, dict):
            errors.append("Талон в позиции " + str(position) + " должен быть словарём.")
            continue

        ticket_id = ticket.get("id")
        if type(ticket_id) is not int or ticket_id <= 0:
            errors.append("У каждого талона нужен положительный целый id.")
        elif ticket_id in ticket_ids:
            errors.append("Номера талонов не должны повторяться.")
        else:
            ticket_ids.add(ticket_id)

        queue_order = ticket.get("queue_order")
        if type(queue_order) is not int or queue_order <= 0:
            errors.append("У каждого талона нужен положительный queue_order.")
        elif queue_order in queue_orders:
            errors.append("Порядок очереди не должен повторяться.")
        else:
            queue_orders.add(queue_order)

        cabinet = ticket.get("cabinet")
        if cabinet not in cabinets:
            errors.append("У талона указан неизвестный кабинет.")

        status = ticket.get("status")
        if status not in valid_statuses:
            errors.append("У талона указан неизвестный статус.")
        elif status == "in_service":
            if cabinet in busy_cabinets:
                errors.append(
                    "В одном кабинете не может быть два талона in_service."
                )
            else:
                busy_cabinets.add(cabinet)

    event_ids = set()
    valid_actions = ["call", "cancel", "finish", "move", "requeue"]
    for position, event in enumerate(history, start=1):
        if not isinstance(event, dict):
            errors.append("Событие в позиции " + str(position) + " должно быть словарём.")
            continue

        event_id = event.get("event_id")
        if type(event_id) is not int or event_id <= 0:
            errors.append("У каждого события нужен положительный event_id.")
        elif event_id in event_ids:
            errors.append("Номера событий не должны повторяться.")
        else:
            event_ids.add(event_id)

        if event.get("action") not in valid_actions:
            errors.append("У события указано неизвестное действие.")

        if event.get("record_id") not in ticket_ids:
            errors.append("Событие ссылается на неизвестный талон.")

        related_ids = event.get("related_ids")
        if not isinstance(related_ids, list):
            errors.append("related_ids события должен быть списком.")
        else:
            for related_id in related_ids:
                if related_id not in ticket_ids:
                    errors.append("Событие содержит неизвестный related_id.")

    return errors


def build_work_document(state):
    """Собрать понятный текст с текущими талонами и журналом."""
    lines = ["Электронная регистратура", "", "Талоны:"]
    for ticket in state["records"]:
        lines.append(
            "№"
            + str(ticket["id"])
            + " | "
            + ticket["visitor_code"]
            + " | кабинет "
            + ticket["cabinet"]
            + " | "
            + ticket["status"]
            + " | порядок "
            + str(ticket["queue_order"])
        )

    lines.append("")
    lines.append("История действий:")
    if len(state["history"]) == 0:
        lines.append("История пуста.")
    else:
        for event in state["history"]:
            lines.append(
                "Событие №"
                + str(event["event_id"])
                + " | "
                + event["action"]
                + " | талон "
                + str(event["record_id"])
                + " | "
                + event["field"]
                + " | было: "
                + str(event["before"])
                + " | стало: "
                + str(event["after"])
            )
    return "\n".join(lines)

