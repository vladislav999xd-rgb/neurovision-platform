"""Современный интерфейс учебной электронной регистратуры."""

from base64 import b64encode
from html import escape
from pathlib import Path

import pandas as pd
import streamlit as st

from history_tools import events_for_record
from io_adapter import export_document, load_state, save_state
from student_logic import (
    build_work_document,
    call_next_logged,
    cancel_ticket_logged,
    find_ticket,
    finish_ticket_logged,
    make_initial_extra,
    make_initial_records,
    register_ticket,
    requeue_ticket,
    room_message,
    select_tickets,
    transfer_ticket_logged,
    validate_project_state,
)


st.set_page_config(
    page_title="Электронная регистратура",
    page_icon="✚",
    layout="wide",
    initial_sidebar_state="collapsed",
)


ROOT = Path(__file__).resolve().parent
ASSET_DIR = ROOT / "assets"
QUACK_FILE = ASSET_DIR / "quack.mp3"
AMBIENT_FILE = ASSET_DIR / "ambient-loop.mp3"
HERO_FILE = ASSET_DIR / "clinic_hero.jpg"
STATE_KEY = "state"
PROJECT_CODE = "queue"
STATE_FILE = ROOT / "registratura_state.json"
DOCUMENT_FILE = ROOT / "work_document.txt"

STATUS_LABELS = {
    "waiting": "Ожидает",
    "in_service": "На приёме",
    "done": "Завершён",
    "canceled": "Отменён",
}

STATUS_COLORS = {
    "Ожидает": ("#F5B94F", "rgba(245, 185, 79, 0.16)"),
    "На приёме": ("#39B8FF", "rgba(57, 184, 255, 0.16)"),
    "Завершён": ("#2DD4A0", "rgba(45, 212, 160, 0.16)"),
    "Отменён": ("#FF6B86", "rgba(255, 107, 134, 0.16)"),
}

CLINIC_JOKES = [
    "Пациент: «Доктор, у меня всё двоится». Доктор: «Садитесь оба».",
    "Очередь исчезла из регистратуры. Оказалось, её отфильтровали по status=done.",
    "Кибердоктор закончил диагностику: ошибка 404 — болезнь не найдена.",
]


def show_cabinet_option(value):
    """Показать пустой фильтр понятным текстом."""
    if value == "":
        return "Все кабинеты"
    return "Кабинет " + value


def show_status_option(value):
    """Показать пустой фильтр понятным текстом."""
    if value == "":
        return "Все статусы"
    return STATUS_LABELS[value]


def show_ticket_option(ticket_id):
    """Подписать талон в выпадающем списке действий."""
    ticket = find_ticket(st.session_state[STATE_KEY]["records"], ticket_id)
    if ticket is None:
        return "Талон #" + str(ticket_id)
    return (
        "Талон #"
        + str(ticket_id)
        + " · "
        + ticket["visitor_code"]
        + " · кабинет "
        + ticket["cabinet"]
    )


def make_display_rows(tickets):
    """Подготовить строки таблицы, не изменяя исходные записи."""
    rows = []
    for ticket in tickets:
        rows.append(
            {
                "Талон": ticket["id"],
                "Код посетителя": ticket["visitor_code"],
                "Кабинет": ticket["cabinet"],
                "Статус": STATUS_LABELS[ticket["status"]],
                "Порядок в очереди": ticket["queue_order"],
            }
        )
    return rows


def make_history_rows(events):
    """Подготовить события для понятной таблицы интерфейса."""
    action_labels = {
        "call": "Вызов",
        "cancel": "Отмена",
        "finish": "Завершение",
        "move": "Перенос",
        "requeue": "Возврат в очередь",
    }
    rows = []
    for event in events:
        rows.append(
            {
                "Событие": event["event_id"],
                "Действие": action_labels.get(event["action"], event["action"]),
                "Талон": event["record_id"],
                "Поле": event["field"],
                "Было": str(event["before"]),
                "Стало": str(event["after"]),
            }
        )
    return rows


def count_status(records, status):
    """Посчитать записи с указанным статусом."""
    return sum(1 for ticket in records if ticket["status"] == status)


@st.cache_data(show_spinner=False)
def read_asset(path_string):
    """Прочитать неизменяемый локальный ресурс один раз."""
    path = Path(path_string)
    if not path.exists():
        return b""
    return path.read_bytes()


def make_quack_sound():
    """Вернуть присланный пользователем звук нажатия."""
    return read_asset(str(QUACK_FILE))


def make_ambient_sound():
    """Вернуть постоянный фоновый звук сайта."""
    return read_asset(str(AMBIENT_FILE))


def image_data_uri(path):
    """Подготовить локальную картинку для hero без внешних ссылок."""
    content = read_asset(str(path))
    if not content:
        return ""
    mime = "image/jpeg" if path.suffix.lower() in [".jpg", ".jpeg"] else "image/png"
    return "data:" + mime + ";base64," + b64encode(content).decode("ascii")


def request_quack():
    """Запросить новое воспроизведение кряканья после любого нажатия."""
    serial = st.session_state.get("quack_serial", 0) + 1
    st.session_state["quack_serial"] = serial
    st.session_state["quack_pending"] = serial


def set_flash(result):
    """Сохранить результат операции до следующей перерисовки."""
    st.session_state["flash"] = (
        "success" if result["ok"] else "error",
        result["message"],
    )


def cabinet_load(records, cabinet):
    """Собрать показатели одного кабинета."""
    waiting = sum(
        1
        for ticket in records
        if ticket["cabinet"] == cabinet and ticket["status"] == "waiting"
    )
    busy = any(
        ticket["cabinet"] == cabinet and ticket["status"] == "in_service"
        for ticket in records
    )
    active = waiting + int(busy)
    return {"waiting": waiting, "busy": busy, "active": active}


def recommend_cabinet(records):
    """Предложить свободный кабинет с самой короткой очередью."""
    cabinets = st.session_state[STATE_KEY]["extra"]["cabinets"]
    loads = {cabinet: cabinet_load(records, cabinet) for cabinet in cabinets}
    return min(
        cabinets,
        key=lambda cabinet: (
            loads[cabinet]["busy"],
            loads[cabinet]["waiting"],
            cabinet,
        ),
    )


def metric_card(icon, value, label, tone):
    """Отрисовать одну карточку статистики."""
    return f"""
    <div class="metric-card metric-{tone}">
        <div class="metric-icon">{icon}</div>
        <div>
            <div class="metric-value">{value}</div>
            <div class="metric-label">{label}</div>
        </div>
    </div>
    """


def status_cell_style(value):
    """Цвет статуса внутри таблицы."""
    foreground, background = STATUS_COLORS.get(value, ("#F7FBFF", "transparent"))
    return (
        f"color: {foreground}; background-color: {background}; "
        "font-weight: 700; border-radius: 8px;"
    )


def reset_filters():
    """Очистить три фильтра до построения виджетов."""
    request_quack()
    st.session_state["filter_cabinet"] = ""
    st.session_state["filter_status"] = ""
    st.session_state["filter_fragment"] = ""


def handle_registration():
    """Создать талон до единственной штатной перерисовки Streamlit."""
    request_quack()
    state = st.session_state[STATE_KEY]
    result = register_ticket(
        state["records"],
        state["extra"]["cabinets"],
        st.session_state.get("register_visitor_code", ""),
        st.session_state["register_cabinet"],
    )
    set_flash(result)


def handle_call_next():
    """Вызвать следующего посетителя в callback кнопки."""
    request_quack()
    result = call_next_logged(
        st.session_state[STATE_KEY],
        st.session_state["call_cabinet"],
    )
    set_flash(result)


def handle_transfer():
    """Перенести выбранный талон без дополнительного st.rerun."""
    request_quack()
    result = transfer_ticket_logged(
        st.session_state[STATE_KEY],
        st.session_state["transfer_ticket_id"],
        st.session_state["transfer_target_cabinet"],
    )
    set_flash(result)


def handle_requeue():
    """Вернуть отменённый талон без дополнительного st.rerun."""
    request_quack()
    result = requeue_ticket(
        st.session_state[STATE_KEY],
        st.session_state["requeue_ticket_id"],
    )
    set_flash(result)


def handle_finish():
    """Завершить выбранное обращение."""
    request_quack()
    result = finish_ticket_logged(
        st.session_state[STATE_KEY],
        st.session_state["action_ticket"],
    )
    set_flash(result)


def handle_cancel():
    """Отменить выбранный ожидающий талон."""
    request_quack()
    result = cancel_ticket_logged(
        st.session_state[STATE_KEY],
        st.session_state["action_ticket"],
    )
    set_flash(result)


def handle_save_state():
    """Сохранить состояние и показать результат на том же проходе."""
    request_quack()
    result = save_state(STATE_FILE, PROJECT_CODE, st.session_state[STATE_KEY])
    set_flash(result)


def handle_load_state():
    """Проверить файл и заменить state до построения страницы."""
    request_quack()
    loaded = load_state(STATE_FILE, PROJECT_CODE)
    if not loaded["ok"]:
        set_flash(loaded)
        return

    errors = validate_project_state(loaded["state"])
    if errors:
        set_flash({"ok": False, "message": " ".join(errors)})
        return

    st.session_state[STATE_KEY] = loaded["state"]
    set_flash({"ok": True, "message": "Работа восстановлена."})


def handle_export_document():
    """Сформировать и выгрузить рабочий документ."""
    request_quack()
    state = st.session_state[STATE_KEY]
    result = export_document(DOCUMENT_FILE, build_work_document(state))
    set_flash(result)


def handle_next_joke():
    """Переключить шутку в callback."""
    request_quack()
    st.session_state["joke_index"] = (
        st.session_state.get("joke_index", 0) + 1
    ) % len(CLINIC_JOKES)


def handle_reset_demo():
    """Восстановить исходный учебный state до построения страницы."""
    request_quack()
    st.session_state[STATE_KEY] = {
        "records": make_initial_records(),
        "history": [],
        "extra": make_initial_extra(),
    }
    st.session_state["flash"] = (
        "success",
        "Демонстрационные данные восстановлены.",
    )


CSS = """
<style>
:root {
    --bg: #071426;
    --sidebar: #081B32;
    --card: #102642;
    --card-2: #142E4D;
    --cyan: #18D9E6;
    --blue: #39B8FF;
    --purple: #A35CF6;
    --success: #2DD4A0;
    --warning: #F5B94F;
    --error: #FF6B86;
    --text: #F7FBFF;
    --muted: #A8BCD1;
    --line: rgba(116, 196, 235, 0.24);
}

html { scroll-behavior: smooth; }
body, [data-testid="stAppViewContainer"], .stApp {
    background:
        radial-gradient(circle at 77% 1%, rgba(57,184,255,.12), transparent 28rem),
        radial-gradient(circle at 45% 28%, rgba(163,92,246,.08), transparent 25rem),
        var(--bg);
    color: var(--text);
    font-family: Inter, "Segoe UI", sans-serif;
}

[data-testid="stHeader"] { background: transparent; height: 0; }
[data-testid="stToolbar"], [data-testid="stDecoration"], #MainMenu, footer { display: none !important; }
.block-container {
    max-width: 1780px;
    padding: 1.05rem 1.2rem 4rem 258px;
}

h1, h2, h3, p { color: var(--text); }
.anchor { position: relative; top: -1rem; visibility: hidden; }

.medical-sidebar {
    position: fixed;
    inset: 0 auto 0 0;
    width: 236px;
    z-index: 999;
    padding: 30px 14px 20px;
    display: flex;
    flex-direction: column;
    background:
        radial-gradient(circle at 30% 64%, rgba(24,217,230,.09), transparent 16rem),
        linear-gradient(180deg, #07172b, var(--sidebar));
    border-right: 1px solid rgba(76,189,236,.24);
    box-shadow: 18px 0 60px rgba(0,0,0,.2);
}
.brand { display:flex; gap:14px; align-items:center; padding: 0 12px 25px; }
.brand-cross {
    width:43px; height:43px; display:grid; place-items:center;
    border-radius:12px; color:var(--cyan); font-size:34px; font-weight:900;
    background:rgba(24,217,230,.09); border:1px solid rgba(24,217,230,.45);
    box-shadow:0 0 22px rgba(24,217,230,.26), inset 0 0 18px rgba(24,217,230,.08);
}
.brand-title {font-size:16px; font-weight:800; line-height:1.18;}
.brand-subtitle {font-size:11px; color:var(--muted); margin-top:5px;}
.nav-list { display:flex; flex-direction:column; gap:7px; }
.nav-link {
    display:flex; align-items:center; gap:12px; min-height:44px;
    padding:0 14px; color:#D8E9F6 !important; text-decoration:none !important;
    border:1px solid transparent; border-radius:11px; font-size:13px; font-weight:600;
    transition:.22s ease;
}
.nav-link:hover, .nav-link.active {
    color:white !important; background:linear-gradient(90deg,rgba(24,217,230,.22),rgba(57,184,255,.12));
    border-color:rgba(24,217,230,.42); transform:translateX(2px);
    box-shadow:0 0 20px rgba(24,217,230,.10);
}
.nav-icon {width:22px; text-align:center; color:#8CDBFF; font-size:18px;}
.sidebar-bottom { margin-top:auto; }
.pulse-line {
    height:70px; opacity:.8; margin:0 3px 16px;
    background:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 220 70'%3E%3Cpath d='M0 40h42l11-2 9-15 10 33 12-48 10 32h126' fill='none' stroke='%2318D9E6' stroke-width='3'/%3E%3C/svg%3E") center/cover no-repeat;
    filter:drop-shadow(0 0 9px rgba(24,217,230,.65));
}
.study-card {
    padding:14px; border:1px solid rgba(57,184,255,.35); border-radius:12px;
    background:linear-gradient(145deg,rgba(20,55,90,.8),rgba(10,35,61,.9));
    color:var(--muted); font-size:11px; line-height:1.45;
}
.study-card strong {display:block; color:white; font-size:12px; margin-bottom:7px;}

.medical-hero {
    position:relative; overflow:hidden; min-height:294px; display:flex; align-items:center;
    padding:32px 38px; border:1px solid rgba(80,193,239,.26); border-radius:18px;
    background-color:#0B213C; background-size:cover; background-position:center right;
    box-shadow:0 25px 70px rgba(0,0,0,.24), inset 0 1px rgba(255,255,255,.05);
}
.medical-hero::before {
    content:""; position:absolute; inset:0;
    background:linear-gradient(90deg,#0A1E38 0%,rgba(10,30,56,.97) 34%,rgba(10,30,56,.60) 60%,rgba(7,20,38,.08) 100%);
}
.medical-hero::after {
    content:""; position:absolute; inset:auto 0 0; height:2px;
    background:linear-gradient(90deg,var(--cyan),var(--blue),transparent 78%);
    box-shadow:0 0 18px var(--cyan);
}
.hero-copy {position:relative; z-index:2; width:min(620px,60%);}
.online-badge {
    display:inline-flex; align-items:center; gap:8px; padding:6px 12px; border-radius:999px;
    border:1px solid rgba(24,217,230,.7); background:rgba(7,31,52,.62);
    color:#DDFEFF; font-size:12px; font-weight:700;
}
.online-dot {width:8px; height:8px; background:var(--cyan); border-radius:50%; box-shadow:0 0 12px var(--cyan);}
.hero-title {font-size:clamp(34px,3.4vw,55px); line-height:.98; letter-spacing:-1.8px; font-weight:850; margin:16px 0 13px;}
.hero-title span {color:var(--cyan); text-shadow:0 0 25px rgba(24,217,230,.24);}
.hero-description {max-width:610px; color:#D1E3F2; font-size:14px; line-height:1.45; margin:0 0 15px;}
.hero-badges {display:flex; flex-wrap:wrap; gap:9px;}
.feature-badge {
    padding:8px 13px; border-radius:9px; border:1px solid rgba(123,198,235,.3);
    background:rgba(16,43,75,.76); color:#EAF7FF; font-size:12px; font-weight:650;
    backdrop-filter:blur(9px);
}
.feature-badge.purple {border-color:rgba(163,92,246,.5); background:rgba(102,54,150,.28);}

.metric-card {
    min-height:92px; display:flex; align-items:center; gap:15px; margin:14px 0 0;
    padding:16px 18px; border-radius:14px; border:1px solid var(--line);
    background:linear-gradient(145deg,rgba(20,49,82,.95),rgba(11,32,56,.96));
    box-shadow:0 12px 28px rgba(0,0,0,.15), inset 0 1px rgba(255,255,255,.04);
    transition:transform .2s ease,border-color .2s ease;
}
.metric-card:hover {transform:translateY(-3px); border-color:rgba(24,217,230,.5);}
.metric-icon {width:48px;height:48px;border-radius:50%;display:grid;place-items:center;font-size:22px;background:rgba(57,184,255,.14);color:var(--blue);box-shadow:0 0 22px rgba(57,184,255,.12);}
.metric-value {font-size:26px;line-height:1;font-weight:850;color:white;}
.metric-label {font-size:11px;color:#D3E1ED;margin-top:7px;white-space:nowrap;}
.metric-waiting .metric-icon{color:var(--warning);background:rgba(245,185,79,.14)}
.metric-service .metric-icon{color:var(--cyan);background:rgba(24,217,230,.14)}
.metric-done .metric-icon{color:var(--success);background:rgba(45,212,160,.14)}
.metric-canceled .metric-icon{color:var(--error);background:rgba(255,107,134,.14)}

[data-testid="stVerticalBlockBorderWrapper"] {
    border-color:var(--line) !important; border-radius:15px !important;
    background:linear-gradient(145deg,rgba(16,38,66,.94),rgba(10,29,51,.96));
    box-shadow:0 16px 36px rgba(0,0,0,.14),inset 0 1px rgba(255,255,255,.035);
}
.section-title {display:flex;align-items:center;gap:10px;font-size:18px;font-weight:800;color:white;margin:0 0 2px;}
.section-icon {color:var(--cyan);font-size:22px;text-shadow:0 0 13px rgba(24,217,230,.5);}
.section-copy {font-size:12px;color:var(--muted);margin:0 0 12px;}

label, [data-testid="stWidgetLabel"] p {color:#EAF5FD !important;font-size:12px !important;font-weight:650 !important;}
[data-baseweb="input"] > div, [data-baseweb="select"] > div {
    background:#0D2542 !important; border-color:rgba(122,190,226,.40) !important; border-radius:9px !important;
}
[data-baseweb="input"] input {color:white !important;}
[data-baseweb="select"] span {color:#F7FBFF !important;}
[data-testid="stButton"] button, [data-testid="stFormSubmitButton"] button {
    width:100%; min-height:42px; border-radius:9px; border:1px solid rgba(24,217,230,.75);
    color:white; font-weight:750; background:linear-gradient(100deg,#0ABBC9,#168ED7 58%,#8B4BE8);
    box-shadow:0 7px 22px rgba(20,180,213,.18); transition:.2s ease;
}
[data-testid="stButton"] button:hover, [data-testid="stFormSubmitButton"] button:hover {
    border-color:#8DFAFF; color:white; transform:translateY(-2px); box-shadow:0 11px 28px rgba(24,217,230,.24);
}
[data-testid="stButton"] button:active, [data-testid="stFormSubmitButton"] button:active {transform:translateY(0) scale(.99);}

.room-grid {display:grid;grid-template-columns:repeat(3,1fr);gap:9px;margin-top:7px;}
.room-card {padding:13px 12px;border-radius:11px;border:1px solid rgba(24,217,230,.4);background:linear-gradient(145deg,rgba(21,68,85,.55),rgba(12,44,67,.75));}
.room-card.busy{border-color:rgba(163,92,246,.55);background:linear-gradient(145deg,rgba(74,48,112,.55),rgba(31,43,76,.76));}
.room-number{font-size:20px;font-weight:850;color:white;margin-bottom:8px;}
.room-state{display:inline-flex;gap:6px;align-items:center;padding:4px 8px;border-radius:999px;font-size:10px;font-weight:750;color:#B9FFE8;background:rgba(45,212,160,.18);}
.room-card.busy .room-state{color:#FFD6E0;background:rgba(255,107,134,.18);}
.state-dot{width:7px;height:7px;border-radius:50%;background:var(--success);box-shadow:0 0 9px var(--success);}
.room-card.busy .state-dot{background:var(--error);box-shadow:0 0 9px var(--error);}
.room-waiting{margin:11px 0 8px;color:#D9E8F3;font-size:10px;}
.load-track{height:7px;border-radius:999px;background:rgba(144,195,221,.22);overflow:hidden;}
.load-fill{height:100%;border-radius:999px;background:linear-gradient(90deg,var(--success),var(--cyan));box-shadow:0 0 10px rgba(24,217,230,.5);}
.room-card.busy .load-fill{background:linear-gradient(90deg,#A35CF6,#FF6B86);}
.recommendation{margin-top:12px;padding:9px 11px;border-radius:9px;background:rgba(24,217,230,.08);border:1px solid rgba(24,217,230,.22);font-size:11px;color:#CFFBFF;}

[data-testid="stDataFrame"] {border:1px solid var(--line);border-radius:11px;overflow:hidden;background:#0C213B;}
[data-testid="stDataFrame"] * {font-family:Inter,"Segoe UI",sans-serif !important;}
.action-note {padding:10px 12px;border-left:2px solid var(--cyan);background:rgba(24,217,230,.06);border-radius:0 8px 8px 0;color:var(--muted);font-size:11px;line-height:1.45;}
.selected-card {padding:10px 12px;margin-top:5px;border:1px solid rgba(116,196,235,.25);border-radius:9px;background:rgba(7,24,44,.52);color:#D9EBF7;font-size:11px;}

.joke-panel {display:flex;gap:14px;align-items:center;padding:15px 18px;border:1px solid rgba(163,92,246,.30);border-radius:13px;background:linear-gradient(90deg,rgba(91,55,135,.20),rgba(16,48,77,.65));}
.joke-orb {flex:none;width:42px;height:42px;border-radius:50%;display:grid;place-items:center;background:rgba(163,92,246,.2);font-size:22px;box-shadow:0 0 22px rgba(163,92,246,.18);}
.joke-label {font-size:10px;text-transform:uppercase;letter-spacing:1.4px;color:#CDAEFF;font-weight:800;}
.joke-text {margin-top:3px;color:#F4F8FD;font-size:12px;line-height:1.4;}

[data-testid="stAlert"] {border-radius:10px;border:1px solid var(--line);}
[data-testid="stAudio"] {position:fixed!important;width:1px!important;height:1px!important;opacity:.001!important;overflow:hidden!important;pointer-events:none!important;left:-10px!important;bottom:-10px!important;}
.sound-status {position:fixed;right:18px;bottom:14px;z-index:700;padding:7px 11px;border:1px solid rgba(24,217,230,.26);border-radius:999px;background:rgba(7,25,45,.88);backdrop-filter:blur(8px);font-size:10px;color:#BDEDF2;box-shadow:0 8px 24px rgba(0,0,0,.22);}
.sound-status::before{content:"";display:inline-block;width:7px;height:7px;margin-right:7px;border-radius:50%;background:var(--cyan);box-shadow:0 0 9px var(--cyan);}
.page-footer {text-align:center;color:#6F89A3;font-size:10px;margin:25px 0 0;}

@media (max-width: 1100px) {
    .block-container {padding-left:226px;}
    .medical-sidebar {width:206px;}
    .metric-card {padding:14px 11px;gap:10px;}
    .metric-icon {width:39px;height:39px;}
    .metric-label {white-space:normal;}
    .hero-copy {width:72%;}
}
@media (max-width: 820px) {
    .block-container {padding: .8rem .75rem 4rem;}
    .medical-sidebar {position:relative;width:auto;height:auto;margin:-.8rem -.75rem 12px;padding:12px;display:block;border-right:0;border-bottom:1px solid var(--line);}
    .brand {padding:0 5px 10px;}
    .brand-cross{width:35px;height:35px;font-size:28px;}
    .nav-list{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:5px;}
    .nav-link{min-height:36px;padding:0 8px;font-size:10px;}
    .nav-icon{font-size:14px;width:16px;}
    .sidebar-bottom{display:none;}
    .medical-hero{min-height:330px;padding:25px 22px;background-position:65% center;}
    .medical-hero::before{background:linear-gradient(90deg,rgba(7,22,42,.98),rgba(7,22,42,.82));}
    .hero-copy{width:100%;}
    .hero-title{font-size:38px;}
    .room-grid{grid-template-columns:1fr;}
}
@media (max-width: 520px) {
    .nav-list{grid-template-columns:repeat(2,minmax(0,1fr));}
    .medical-hero{min-height:380px;}
    .hero-title{font-size:32px;letter-spacing:-1px;}
    .hero-description{font-size:12px;}
    .sound-status{display:none;}
}
@media (prefers-reduced-motion: reduce) {
    html{scroll-behavior:auto;}
    *{animation:none!important;transition:none!important;}
}
</style>
"""


st.markdown(CSS, unsafe_allow_html=True)

# Streamlit перерисовывает Python-приложение после виджетов. Запоминаем
# положение прокрутки в браузере и возвращаем его после обновления DOM.
st.iframe(
    """
    <script>
    (() => {
        const host = window.parent;
        const doc = host.document;
        const key = "registratura-scroll-position";

        function scrollTarget() {
            return doc.querySelector('[data-testid="stMain"]') || host;
        }
        function currentPosition(target) {
            return target === host ? host.scrollY : target.scrollTop;
        }
        function savePosition() {
            const target = scrollTarget();
            host.sessionStorage.setItem(key, String(currentPosition(target)));
        }
        function restorePosition() {
            const saved = host.sessionStorage.getItem(key);
            if (saved === null) return;
            const target = scrollTarget();
            const top = Number(saved);
            if (target === host) host.scrollTo(0, top);
            else target.scrollTop = top;
        }

        const old = host.__registraturaScrollKeeper;
        const target = scrollTarget();
        if (!old || old.target !== target) {
            if (old) old.target.removeEventListener('scroll', old.listener);
            target.addEventListener('scroll', savePosition, {passive: true});
            host.__registraturaScrollKeeper = {target, listener: savePosition};
        }

        requestAnimationFrame(restorePosition);
        setTimeout(restorePosition, 120);
        setTimeout(restorePosition, 350);
    })();
    </script>
    """,
    width=1,
    height=1,
    tab_index=-1,
)

if STATE_KEY not in st.session_state:
    previous_records = st.session_state.pop("records", None)
    if previous_records is None:
        previous_records = make_initial_records()
    st.session_state[STATE_KEY] = {
        "records": previous_records,
        "history": [],
        "extra": make_initial_extra(),
    }
if "joke_index" not in st.session_state:
    st.session_state["joke_index"] = 0

state = st.session_state[STATE_KEY]
records = state["records"]
cabinets = state["extra"]["cabinets"]

# Постоянный аудиопоток создаётся в одном и том же месте на каждом rerun.
ambient_sound = make_ambient_sound()
if ambient_sound:
    st.audio(ambient_sound, format="audio/mp3", autoplay=True, loop=True)

st.markdown(
    """
    <aside class="medical-sidebar">
        <div class="brand">
            <div class="brand-cross">✚</div>
            <div><div class="brand-title">Электронная<br>регистратура</div><div class="brand-subtitle">Учебная клиника</div></div>
        </div>
        <nav class="nav-list">
            <a class="nav-link active" href="#home" target="_self"><span class="nav-icon">⌂</span>Главная</a>
            <a class="nav-link" href="#registration" target="_self"><span class="nav-icon">▱</span>Регистрация талона</a>
            <a class="nav-link" href="#queue" target="_self"><span class="nav-icon">♧</span>Управление очередью</a>
            <a class="nav-link" href="#tickets" target="_self"><span class="nav-icon">⌕</span>Талоны и поиск</a>
            <a class="nav-link" href="#cabinets" target="_self"><span class="nav-icon">▣</span>Кабинеты</a>
            <a class="nav-link" href="#statistics" target="_self"><span class="nav-icon">▥</span>Статистика</a>
            <a class="nav-link" href="#history" target="_self"><span class="nav-icon">◴</span>История и файлы</a>
        </nav>
        <div class="sidebar-bottom">
            <div class="pulse-line"></div>
            <div class="study-card"><strong>ⓘ &nbsp;Учебный проект</strong>Данные демонстрационные и предназначены для учебных целей.</div>
        </div>
    </aside>
    """,
    unsafe_allow_html=True,
)

hero_image = image_data_uri(HERO_FILE)
hero_style = ""
if hero_image:
    hero_style = f"background-image: url('{hero_image}')"

st.markdown('<div id="home" class="anchor"></div>', unsafe_allow_html=True)
st.markdown(
    f"""
    <section class="medical-hero" style="{hero_style}">
        <div class="hero-copy">
            <div class="online-badge"><span class="online-dot"></span>Медицина онлайн</div>
            <div class="hero-title">Забота начинается<br>с <span>удобной записи</span></div>
            <p class="hero-description">Современная электронная регистратура учебной клиники: выдавайте талоны, управляйте очередью и сопровождайте посетителя до завершения приёма.</p>
            <div class="hero-badges">
                <span class="feature-badge purple">✦ &nbsp;День 4</span>
                <span class="feature-badge">♧ &nbsp;Умная очередь</span>
                <span class="feature-badge">▱ &nbsp;3 кабинета</span>
            </div>
        </div>
    </section>
    """,
    unsafe_allow_html=True,
)

flash = st.session_state.pop("flash", None)
if flash is not None:
    flash_type, flash_message = flash
    if flash_type == "success":
        st.toast(flash_message, icon="✅")
    else:
        st.toast(flash_message, icon="⚠️")

st.markdown('<div id="statistics" class="anchor"></div>', unsafe_allow_html=True)
metric_columns = st.columns(5)
metric_values = [
    ("▱", len(records), "Всего талонов", "total"),
    ("◷", count_status(records, "waiting"), "Ожидают вызова", "waiting"),
    ("♧", count_status(records, "in_service"), "Сейчас на приёме", "service"),
    ("✓", count_status(records, "done"), "Приём завершён", "done"),
    ("×", count_status(records, "canceled"), "Отменён", "canceled"),
]
for column, (icon, value, label, tone) in zip(metric_columns, metric_values):
    with column:
        st.markdown(metric_card(icon, value, label, tone), unsafe_allow_html=True)

st.write("")
registration_column, call_column, cabinet_column = st.columns([1.05, 1.05, 1.35])

with registration_column:
    st.markdown('<div id="registration" class="anchor"></div>', unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown('<div class="section-title"><span class="section-icon">▱</span>Создать талон</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-copy">Зарегистрируйте посетителя и выберите кабинет.</div>', unsafe_allow_html=True)
        with st.form("registration_form", clear_on_submit=True):
            visitor_code = st.text_input(
                "Код посетителя",
                placeholder="Например, P107",
                help="Одна буква P и три цифры",
                key="register_visitor_code",
            )
            register_cabinet = st.selectbox("Кабинет", cabinets, key="register_cabinet")
            st.form_submit_button(
                "＋  Создать талон",
                on_click=handle_registration,
            )

with call_column:
    st.markdown('<div id="queue" class="anchor"></div>', unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown('<div class="section-title"><span class="section-icon">◁</span>Вызвать следующего</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-copy">Первым будет вызван ожидающий с наименьшим порядком.</div>', unsafe_allow_html=True)
        call_cabinet = st.selectbox("Кабинет", cabinets, key="call_cabinet")
        st.info(room_message(records, call_cabinet))
        st.button(
            "▷  Вызвать следующего",
            key="call_next_button",
            on_click=handle_call_next,
        )

with cabinet_column:
    st.markdown('<div id="cabinets" class="anchor"></div>', unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown('<div class="section-title"><span class="section-icon">▣</span>Состояние кабинетов</div>', unsafe_allow_html=True)
        room_cards = []
        for room in cabinets:
            load = cabinet_load(records, room)
            state_class = "busy" if load["busy"] else ""
            state_text = "Занят" if load["busy"] else "Свободен"
            width = min(100, 22 + load["active"] * 22)
            room_cards.append(
                f'<div class="room-card {state_class}">'
                f'<div class="room-number">▣ &nbsp;{room}</div>'
                f'<div class="room-state"><span class="state-dot"></span>{state_text}</div>'
                f'<div class="room-waiting">♧ &nbsp;Ожидают: {load["waiting"]}</div>'
                f'<div class="load-track"><div class="load-fill" style="width:{width}%"></div></div>'
                '</div>'
            )
        recommendation = recommend_cabinet(records)
        st.markdown(
            '<div class="room-grid">'
            + "".join(room_cards)
            + '</div><div class="recommendation">✦ Для нового талона рекомендуем кабинет '
            + recommendation
            + ": здесь сейчас самая лёгкая нагрузка.</div>",
            unsafe_allow_html=True,
        )

st.write("")
transfer_column, requeue_column = st.columns(2)

with transfer_column:
    with st.container(border=True):
        st.markdown('<div class="section-title"><span class="section-icon">⇄</span>Перенос талона</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-copy">День 2 · ожидающий переходит в конец очереди другого кабинета.</div>', unsafe_allow_html=True)
        waiting_ids = [
            ticket["id"] for ticket in records if ticket["status"] == "waiting"
        ]
        if waiting_ids:
            transfer_id = st.selectbox(
                "Ожидающий талон",
                waiting_ids,
                format_func=lambda ticket_id: "Талон #" + str(ticket_id),
                key="transfer_ticket_id",
            )
            transfer_target = st.selectbox(
                "Новый кабинет",
                cabinets,
                key="transfer_target_cabinet",
            )
            transfer_source = find_ticket(records, transfer_id)
            st.caption(
                "Сейчас: кабинет "
                + transfer_source["cabinet"]
                + " · порядок "
                + str(transfer_source["queue_order"])
            )
            st.button(
                "⇄  Перенести талон",
                key="transfer_button",
                on_click=handle_transfer,
            )
        else:
            st.info("Сейчас нет ожидающих талонов для переноса.")

with requeue_column:
    with st.container(border=True):
        st.markdown('<div class="section-title"><span class="section-icon">↩</span>Вернуть в очередь</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-copy">День 4 · возвращается только отменённый талон, его номер сохраняется.</div>', unsafe_allow_html=True)
        requeue_ids = [ticket["id"] for ticket in records]
        requeue_labels = {
            ticket["id"]: (
                "Талон #"
                + str(ticket["id"])
                + " · "
                + ticket["visitor_code"]
                + " · "
                + STATUS_LABELS[ticket["status"]]
            )
            for ticket in records
        }
        requeue_id = st.selectbox(
            "Талон для возврата",
            requeue_ids,
            index=requeue_ids.index(5) if 5 in requeue_ids else 0,
            format_func=requeue_labels.get,
            key="requeue_ticket_id",
        )
        requeue_source = find_ticket(records, requeue_id)
        st.caption(
            "Кабинет "
            + requeue_source["cabinet"]
            + " · текущий порядок "
            + str(requeue_source["queue_order"])
        )
        st.button(
            "↩  Вернуть в очередь",
            key="requeue_button",
            on_click=handle_requeue,
        )

st.write("")
st.markdown('<div id="tickets" class="anchor"></div>', unsafe_allow_html=True)
with st.container(border=True):
    st.markdown('<div class="section-title"><span class="section-icon">☷</span>Талоны и статусы</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-copy">Фильтры помогают искать записи и не изменяют исходные данные. Чем меньше порядок, тем раньше талон вызовут в своём кабинете.</div>', unsafe_allow_html=True)

    filter_col_1, filter_col_2, filter_col_3, filter_col_4 = st.columns([1, 1, 1.25, .62])
    with filter_col_1:
        filter_cabinet = st.selectbox(
            "Кабинет",
            [""] + cabinets,
            format_func=show_cabinet_option,
            key="filter_cabinet",
        )
    with filter_col_2:
        filter_status = st.selectbox(
            "Статус",
            [""] + list(STATUS_LABELS),
            format_func=show_status_option,
            key="filter_status",
        )
    with filter_col_3:
        filter_fragment = st.text_input(
            "Поиск по коду",
            placeholder="Например, P10",
            key="filter_fragment",
        )
    with filter_col_4:
        st.markdown('<div style="height:27px"></div>', unsafe_allow_html=True)
        st.button("↻  Сбросить", key="reset_filters_button", on_click=reset_filters)

    filtered_records = select_tickets(
        records,
        filter_cabinet,
        filter_status,
        filter_fragment,
    )
    table_rows = make_display_rows(filtered_records)
    if table_rows:
        table = pd.DataFrame(table_rows)
        styled_table = table.style.map(status_cell_style, subset=["Статус"])
        st.dataframe(
            styled_table,
            hide_index=True,
            width="stretch",
            height=min(460, 40 + len(table_rows) * 35),
        )
        st.caption("Найдено записей: " + str(len(table_rows)))
    else:
        st.info("По выбранным условиям талоны не найдены.")

st.write("")
with st.container(border=True):
    st.markdown('<div class="section-title"><span class="section-icon">⚙</span>Действие с талоном</div>', unsafe_allow_html=True)
    action_select, finish_column, cancel_column = st.columns([1.25, 1, 1])
    ticket_ids = [ticket["id"] for ticket in records]
    ticket_labels = {
        ticket["id"]: (
            "Талон #"
            + str(ticket["id"])
            + " · "
            + ticket["visitor_code"]
            + " · кабинет "
            + ticket["cabinet"]
        )
        for ticket in records
    }
    with action_select:
        chosen_ticket_id = st.selectbox(
            "Выберите талон",
            ticket_ids,
            format_func=ticket_labels.get,
            key="action_ticket",
        )
        chosen_ticket = find_ticket(records, chosen_ticket_id)
        if chosen_ticket is not None:
            st.markdown(
                '<div class="selected-card">Статус: <strong>'
                + escape(STATUS_LABELS[chosen_ticket["status"]])
                + "</strong> · порядок в очереди: "
                + str(chosen_ticket["queue_order"])
                + "</div>",
                unsafe_allow_html=True,
            )
    with finish_column:
        st.markdown('<div style="height:27px"></div>', unsafe_allow_html=True)
        st.button(
            "✓  Завершить обращение",
            key="finish_button",
            on_click=handle_finish,
        )
    with cancel_column:
        st.markdown('<div style="height:27px"></div>', unsafe_allow_html=True)
        st.button(
            "×  Отменить талон",
            key="cancel_button",
            on_click=handle_cancel,
        )
    st.markdown(
        '<div class="action-note">Завершить можно только талон со статусом «На приёме», а отменить — только ожидающий. Ошибочная операция не изменяет данные.</div>',
        unsafe_allow_html=True,
    )

st.write("")
st.markdown('<div id="history" class="anchor"></div>', unsafe_allow_html=True)
with st.container(border=True):
    st.markdown('<div class="section-title"><span class="section-icon">◴</span>История и файлы</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-copy">День 3 сохраняет рабочее состояние, а день 4 добавляет событие «Возврат в очередь».</div>',
        unsafe_allow_html=True,
    )

    file_info, save_column, load_column, export_column = st.columns([1.35, 1, 1, 1])
    with file_info:
        st.markdown(
            '<div class="selected-card">Событий в журнале: <strong>'
            + str(len(state["history"]))
            + '</strong><br>Снимок: registratura_state.json<br>Документ: work_document.txt</div>',
            unsafe_allow_html=True,
        )
    with save_column:
        st.markdown('<div style="height:27px"></div>', unsafe_allow_html=True)
        st.button(
            "▣  Сохранить",
            key="save_state_button",
            on_click=handle_save_state,
        )
    with load_column:
        st.markdown('<div style="height:27px"></div>', unsafe_allow_html=True)
        st.button(
            "↻  Восстановить",
            key="load_state_button",
            on_click=handle_load_state,
        )
    with export_column:
        st.markdown('<div style="height:27px"></div>', unsafe_allow_html=True)
        st.button(
            "⇩  Выгрузить",
            key="export_document_button",
            on_click=handle_export_document,
        )

    selected_events = events_for_record(state, chosen_ticket_id)
    st.markdown("#### История выбранного талона #" + str(chosen_ticket_id))
    if selected_events:
        st.dataframe(
            pd.DataFrame(make_history_rows(selected_events)),
            hide_index=True,
            width="stretch",
        )
    else:
        st.caption("У выбранного талона пока нет успешных событий.")

    if state["history"]:
        st.markdown("#### Весь журнал действий")
        st.dataframe(
            pd.DataFrame(make_history_rows(state["history"])),
            hide_index=True,
            width="stretch",
            height=min(360, 40 + len(state["history"]) * 35),
        )

st.write("")
joke_content, joke_button = st.columns([3.5, 1])
with joke_content:
    current_joke = CLINIC_JOKES[st.session_state["joke_index"]]
    st.markdown(
        '<div class="joke-panel"><div class="joke-orb">☺</div><div><div class="joke-label">Три шутки клиники</div><div class="joke-text">'
        + escape(current_joke)
        + "</div></div></div>",
        unsafe_allow_html=True,
    )
with joke_button:
    st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)
    st.button(
        "Следующая шутка",
        key="joke_button",
        on_click=handle_next_joke,
    )

reset_column, footer_column = st.columns([1, 3])
with reset_column:
    st.button(
        "Вернуть исходные данные",
        key="reset_demo_button",
        on_click=handle_reset_demo,
    )
with footer_column:
    st.markdown('<div class="page-footer">Электронная регистратура · учебный медицинский проект</div>', unsafe_allow_html=True)

# Уникальный хвост заставляет браузер воспринимать каждый клик как новый звук.
quack_serial = st.session_state.pop("quack_pending", None)
if quack_serial is not None:
    quack_sound = make_quack_sound()
    if quack_sound:
        unique_quack = quack_sound + b"QUACK" + quack_serial.to_bytes(8, "big")
        st.audio(unique_quack, format="audio/mp3", autoplay=True)

st.markdown('<div class="sound-status">Фоновый звук включён · кнопки крякают</div>', unsafe_allow_html=True)
