"""Готовый файловый адаптер дня 3. Предметные правила здесь не задаются."""
import json
from pathlib import Path

FORMAT_VERSION = 1


def _answer(ok, message, state):
    return {"ok": ok, "message": message, "state": state}


def validate_container(state):
    """Проверить только общую оболочку, не поля талонов или анкет."""
    if not isinstance(state, dict):
        return "Состояние должно быть словарём."
    if not isinstance(state.get("records"), list):
        return "В состоянии нужен список records."
    if not isinstance(state.get("history"), list):
        return "В состоянии нужен список history."
    if not isinstance(state.get("extra"), dict):
        return "В состоянии нужен словарь extra."
    return ""


def save_state(path, project, state):
    """Вернуть ok/message/state. Числа и bool сохраняют свой тип."""
    error = validate_container(state)
    if error:
        return _answer(False, error, None)
    payload = {"format_version": FORMAT_VERSION,
               "project": project, "state": state}
    try:
        text = json.dumps(payload, ensure_ascii=False, indent=2,
                          allow_nan=False)
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".tmp")
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(target)
    except (OSError, TypeError, ValueError) as error:
        return _answer(False, "Не удалось сохранить: " + str(error), None)
    return _answer(True, "Состояние сохранено.", None)


def load_state(path, expected_project):
    """Загрузить отдельно. Текущую программу эта функция не меняет."""
    try:
        text = Path(path).read_text(encoding="utf-8")
        payload = json.loads(text)
    except (OSError, UnicodeError, ValueError) as error:
        return _answer(False, "Не удалось прочитать: " + str(error), None)
    if not isinstance(payload, dict):
        return _answer(False, "Файл не содержит словарь состояния.", None)
    if payload.get("format_version") != FORMAT_VERSION:
        return _answer(False, "Неизвестная версия файла.", None)
    if payload.get("project") != expected_project:
        return _answer(False, "Этот файл относится к другому проекту.", None)
    state = payload.get("state")
    error = validate_container(state)
    if error:
        return _answer(False, error, None)
    return _answer(True, "Файл прочитан. Проверьте предметные данные.", state)


def export_document(path, text):
    """Сохранить готовый рабочий текст; содержание формирует команда."""
    if not isinstance(text, str):
        return _answer(False, "Для документа нужен текст.", None)
    try:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    except OSError as error:
        return _answer(False, "Не удалось выгрузить: " + str(error), None)
    return _answer(True, "Рабочий документ выгружен.", None)

