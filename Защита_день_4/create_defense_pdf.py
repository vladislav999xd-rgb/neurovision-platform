"""Создать общую PDF-защиту с распределением участников по дням."""

from io import BytesIO
from pathlib import Path

from PIL import Image
from reportlab.lib.colors import Color, HexColor
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
HERO = ROOT / "assets" / "clinic_hero.png"
UI_PREVIEW = ROOT / "qa" / "registratura_modern_preview.jpg"
HISTORY_PREVIEW = ROOT / "qa" / "day4_history_and_files.jpg"
OUTPUT = HERE / "Общая_защита_по_дням.pdf"

PAGE_W = 960
PAGE_H = 540

NAVY = HexColor("#071426")
SIDEBAR = HexColor("#081B32")
CARD = HexColor("#102642")
CARD_2 = HexColor("#142E4D")
CYAN = HexColor("#18D9E6")
BLUE = HexColor("#39B8FF")
PURPLE = HexColor("#A35CF6")
GREEN = HexColor("#2DD4A0")
YELLOW = HexColor("#F5B94F")
RED = HexColor("#FF6B86")
WHITE = HexColor("#F7FBFF")
MUTED = HexColor("#A8BCD1")
DARK = HexColor("#050F1C")


def register_fonts():
    fonts = Path("C:/Windows/Fonts")
    regular = fonts / "arial.ttf"
    bold = fonts / "arialbd.ttf"
    if regular.exists() and bold.exists():
        pdfmetrics.registerFont(TTFont("UI", regular))
        pdfmetrics.registerFont(TTFont("UI-Bold", bold))
    else:
        pdfmetrics.registerFont(TTFont("UI", fonts / "calibri.ttf"))
        pdfmetrics.registerFont(TTFont("UI-Bold", fonts / "calibrib.ttf"))


def rounded(c, x, y, w, h, color=CARD, stroke=None, radius=14):
    c.setFillColor(color)
    c.setStrokeColor(stroke or color)
    c.roundRect(x, y, w, h, radius, fill=1, stroke=1)


def background(c):
    c.setFillColor(NAVY)
    c.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)
    c.setFillColor(CARD_2)
    c.circle(900, 535, 130, fill=1, stroke=0)
    c.setFillColor(SIDEBAR)
    c.circle(15, -15, 105, fill=1, stroke=0)
    c.setFillColor(CYAN)
    c.rect(48, 28, 864, 2, fill=1, stroke=0)


def wrap_text(text, font, size, max_width):
    words = text.split()
    lines = []
    current = ""
    for word in words:
        candidate = word if not current else current + " " + word
        if pdfmetrics.stringWidth(candidate, font, size) <= max_width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def draw_text(c, text, x, y, width, size=18, color=WHITE, bold=False, leading=None, max_lines=None):
    font = "UI-Bold" if bold else "UI"
    leading = leading or size * 1.25
    lines = []
    for paragraph in text.split("\n"):
        lines.extend(wrap_text(paragraph, font, size, width) or [""])
    if max_lines:
        lines = lines[:max_lines]
    c.setFont(font, size)
    c.setFillColor(color)
    current_y = y
    for line in lines:
        c.drawString(x, current_y, line)
        current_y -= leading
    return current_y


def draw_bullets(c, items, x, y, width, size=16, accent=CYAN, gap=10):
    current_y = y
    for item in items:
        c.setFillColor(accent)
        c.circle(x + 4, current_y + 5, 3, fill=1, stroke=0)
        lines = wrap_text(item, "UI", size, width - 22)
        c.setFont("UI", size)
        c.setFillColor(WHITE)
        line_y = current_y
        for line in lines:
            c.drawString(x + 18, line_y, line)
            line_y -= size * 1.25
        current_y = line_y - gap
    return current_y


def header(c, number, title, subtitle, speaker, timing):
    c.setFont("UI-Bold", 12)
    c.setFillColor(CYAN)
    c.drawString(48, 497, f"{number:02d}")
    draw_text(c, title, 86, 492, 800, 27, WHITE, True)
    draw_text(c, subtitle, 86, 458, 790, 11, MUTED)
    c.setFont("UI", 8)
    c.setFillColor(MUTED)
    c.drawString(48, 12, "День 4 • Электронная регистратура • общая защита")
    c.setFont("UI-Bold", 8)
    c.setFillColor(CYAN)
    c.drawRightString(912, 12, f"Говорит: {speaker} • {timing}")


def cover_image(path, target_w, target_h):
    with Image.open(path) as image:
        image = image.convert("RGB")
        source_ratio = image.width / image.height
        target_ratio = target_w / target_h
        if source_ratio > target_ratio:
            new_width = int(image.height * target_ratio)
            left = (image.width - new_width) // 2
            image = image.crop((left, 0, left + new_width, image.height))
        else:
            new_height = int(image.width / target_ratio)
            top = (image.height - new_height) // 2
            image = image.crop((0, top, image.width, top + new_height))
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=88)
        buffer.seek(0)
        return ImageReader(buffer)


def draw_image(c, path, x, y, w, h):
    if Path(path).exists():
        c.drawImage(cover_image(path, w, h), x, y, w, h, mask="auto")


def page_title(c):
    background(c)
    draw_image(c, HERO, 455, 0, 505, 540)
    c.setFillColor(Color(7 / 255, 20 / 255, 38 / 255, alpha=0.92))
    c.rect(0, 0, 520, 540, fill=1, stroke=0)
    rounded(c, 48, 58, 430, 424, SIDEBAR, CYAN, 18)
    rounded(c, 76, 408, 56, 56, CARD_2, CYAN, 12)
    c.setFont("UI-Bold", 38)
    c.setFillColor(CYAN)
    c.drawCentredString(104, 420, "+")
    draw_text(c, "ОБЩАЯ ЗАЩИТА КОМАНДЫ", 150, 438, 290, 11, CYAN, True)
    draw_text(c, "Электронная\nрегистратура", 76, 342, 350, 32, WHITE, True, 38)
    draw_text(c, "Развитие проекта по дням и возвращение отменённого талона", 76, 250, 350, 16, MUTED, False, 22)
    draw_text(c, "Алёна — день 1\nВлад — день 2\nХеда — день 3\nИра — день 4", 76, 164, 320, 15, WHITE, True, 24)
    draw_text(c, "8 минут демонстрации • 5 минут вопросов", 76, 66, 330, 11, CYAN, True)
    c.showPage()


def two_columns(c, number, title, subtitle, speaker, timing, left_title, left_items, right_title, right_items, left_color=CYAN, right_color=PURPLE):
    background(c)
    header(c, number, title, subtitle, speaker, timing)
    rounded(c, 48, 72, 420, 340, CARD, HexColor("#235276"))
    rounded(c, 492, 72, 420, 340, CARD, HexColor("#235276"))
    draw_text(c, left_title, 72, 382, 360, 18, left_color, True)
    draw_bullets(c, left_items, 72, 342, 360, 15, left_color, 8)
    draw_text(c, right_title, 516, 382, 360, 18, right_color, True)
    draw_bullets(c, right_items, 516, 342, 360, 15, right_color, 8)
    c.showPage()


def code_page(c, number, title, subtitle, speaker, timing, code, side_title, side_items):
    background(c)
    header(c, number, title, subtitle, speaker, timing)
    rounded(c, 48, 66, 560, 350, DARK, HexColor("#246284"))
    rounded(c, 632, 66, 280, 350, CARD, HexColor("#235276"))
    c.setFont("UI", 12)
    c.setFillColor(HexColor("#D2F3FC"))
    y = 386
    for line in code.splitlines():
        c.drawString(68, y, line)
        y -= 21
    draw_text(c, side_title, 656, 382, 230, 17, PURPLE, True)
    draw_bullets(c, side_items, 656, 340, 230, 14, PURPLE, 8)
    c.showPage()


def image_page(c, number, title, subtitle, speaker, timing, image, side_title, side_items):
    background(c)
    header(c, number, title, subtitle, speaker, timing)
    rounded(c, 48, 66, 610, 350, DARK, HexColor("#235276"))
    draw_image(c, image, 58, 76, 590, 330)
    rounded(c, 680, 66, 232, 350, CARD, HexColor("#235276"))
    draw_text(c, side_title, 700, 382, 190, 17, CYAN, True)
    draw_bullets(c, side_items, 700, 340, 190, 13, CYAN, 8)
    c.showPage()


def speech_card(c, number, person, day, functions, speech, questions):
    background(c)
    header(c, number, f"Карточка выступления: {person}", f"{day} • готовая речь и личные вопросы", person, "≈ 1:45")
    rounded(c, 48, 78, 560, 334, CARD, HexColor("#235276"))
    rounded(c, 632, 78, 280, 334, CARD, HexColor("#235276"))
    draw_text(c, "Готовая речь", 72, 382, 500, 17, CYAN, True)
    draw_text(c, speech, 72, 346, 505, 12, WHITE, False, 17)
    draw_text(c, "Мои функции", 656, 382, 230, 17, PURPLE, True)
    draw_bullets(c, functions, 656, 342, 230, 12, PURPLE, 7)
    draw_text(c, "Вопросы", 656, 210, 230, 16, YELLOW, True)
    draw_bullets(c, questions, 656, 178, 230, 11, YELLOW, 5)
    c.showPage()


def build_pdf():
    register_fonts()
    c = canvas.Canvas(str(OUTPUT), pagesize=(PAGE_W, PAGE_H), pageCompression=1)
    c.setTitle("Общая защита электронной регистратуры по дням")
    c.setAuthor("Алёна, Влад, Хеда, Ира")

    page_title(c)
    two_columns(
        c, 2, "Четыре дня — один продукт", "Каждый защищает тот день, над которым работал", "Команда", "0:30",
        "Развитие по дням",
        ["День 1: базовые действия с талонами", "День 2: общие проверки и перенос", "День 3: история и файлы", "День 4: возврат canceled в очередь"],
        "Итоговый продукт",
        ["работающая Streamlit-регистратура", "старые сценарии сохранены", "отказы не меняют данные", "проект запускается из переданной папки"],
    )
    image_page(
        c, 3, "День 1 — основа регистратуры", "Алёна: записи, регистрация и статусы", "Алёна", "0:55",
        UI_PREVIEW, "Что появилось",
        ["6 исходных талонов", "регистрация P107", "вызов", "завершение", "отмена и фильтры"],
    )
    two_columns(
        c, 4, "День 1 — безопасные ветки", "Успех меняет данные, отказ оставляет список прежним", "Алёна", "0:50",
        "Обычный путь",
        ["register_ticket создаёт waiting", "call_next выбирает первого", "finish разрешён для in_service", "cancel разрешён для waiting"],
        "Что объяснить",
        ["аргументы и результат функции", "проверки стоят до append/присваивания", "canceled не считается ожидающим", "select_tickets возвращает новый список"],
    )
    two_columns(
        c, 5, "День 2 — общие функции и перенос", "Влад: одно правило используется регистрацией, вызовом и переносом", "Влад", "0:55",
        "Общие помощники",
        ["normalize_visitor_code", "validate_registration", "next_queue_order", "find_first_waiting"],
        "Перенос талона",
        ["только waiting", "новый кабинет отличается", "нет активного дубля", "меняются cabinet и queue_order"],
    )
    code_page(
        c, 6, "День 2 — id и порядок очереди", "После переноса номер сохраняется, а порядок становится последним", "Влад", "0:50",
        """largest_order = 0
for ticket in tickets:
    if ticket[\"queue_order\"] > largest_order:
        largest_order = ticket[\"queue_order\"]
return largest_order + 1

# id1 переносим в кабинет 103
# новый order = 7
# вызовы в 103: id6, затем id1""",
        "Ключевой ответ",
        ["id — номер записи", "queue_order — место поступления", "минимальный order раньше", "список не сортируется"],
    )
    image_page(
        c, 7, "День 3 — state, история и файлы", "Хеда: успешные действия становятся проверяемыми событиями", "Хеда", "0:55",
        HISTORY_PREVIEW, "Структура",
        ["records", "history", "extra", "append_event", "сохранение JSON"],
    )
    two_columns(
        c, 8, "День 3 — logged-функции", "История появляется только после успешного изменения", "Хеда", "0:50",
        "Действия",
        ["call_next_logged", "cancel_ticket_logged", "finish_ticket_logged", "transfer_ticket_logged"],
        "Безопасная загрузка",
        ["сначала читается candidate", "проверяется версия и проект", "validate_project_state проверяет поля", "только затем заменяется state"],
    )
    two_columns(
        c, 9, "День 4 — возвращение талона", "Ира: canceled становится waiting в том же кабинете", "Ира", "0:55",
        "До действия",
        ["id = 5", "visitor_code = P105", "cabinet = 101", "status = canceled", "queue_order = 5"],
        "После действия",
        ["id и кабинет прежние", "status = waiting", "queue_order = 7", "одно событие requeue", "талон стоит после 1 и 2"],
    )
    code_page(
        c, 10, "День 4 — проверки до изменения", "Повторный клик и неверный статус не портят state", "Ира", "0:50",
        """ticket = find_ticket(tickets, ticket_id)
if ticket is None:
    return failure(...)
if ticket[\"status\"] != \"canceled\":
    return failure(...)

errors = validate_registration(...)
if errors:
    return failure(...)

ticket[\"status\"] = \"waiting\"
ticket[\"queue_order\"] = next_queue_order(tickets)
append_event(state, action=\"requeue\", ...)""",
        "8 проверок",
        ["успех №5", "отказы №1, №3, №4, №99", "двойной клик", "активный дубль P105", "вызовы 1 → 2 → 5"],
    )
    two_columns(
        c, 11, "План живой демонстрации", "Начинаем со свежего набора и показываем поведение, а не только слайды", "Команда", "0:40",
        "На сайте",
        ["вернуть исходные данные", "показать обычный путь", "вернуть талон №5", "повторить и получить отказ", "показать историю"],
        "Проверено",
        ["тесты дней 1–4", "сохранение/восстановление", "12 кнопок и звук", "позиция прокрутки", "запуск переданной копии"],
    )
    two_columns(
        c, 12, "Итог и ограничение", "Новое правило встроено без поломки прежних действий", "Команда", "0:40",
        "Что предъявляем",
        ["запускаемое приложение", "новое правило с примерами", "отказ без изменения", "ключевые функции каждого", "PDF и инструкция"],
        "Ограничение",
        ["учебные вымышленные данные", "локальное JSON-хранилище", "нет авторизации пользователей", "следующее улучшение — база данных"],
    )

    speech_card(
        c, 13, "Алёна", "День 1",
        ["make_initial_records", "register_ticket", "call_next", "finish_ticket / cancel_ticket", "select_tickets"],
        "На первом дне мы создали основу электронной регистратуры. Каждая запись — словарь с id, кодом посетителя, кабинетом, статусом и порядком. Я показываю обычный путь: регистрация создаёт waiting-талон, вызов выбирает первого ожидающего, завершение разрешено только после вызова, а отмена — только в ожидании. Важное правило: сначала выполняются проверки и только потом изменяется список. Поэтому пустой код, занятый кабинет или неверный статус возвращают ok=False и не портят исходные данные. Фильтрация создаёт новый список и тоже не изменяет records.",
        ["Почему canceled не ожидает?", "Что меняет register_ticket?", "Где проверяется статус?"],
    )
    speech_card(
        c, 14, "Влад", "День 2",
        ["normalize_visitor_code", "validate_registration", "next_queue_order", "find_first_waiting", "transfer_ticket"],
        "На втором дне мы вынесли повторяющиеся правила в общие функции и добавили перенос. normalize_visitor_code убирает внешние пробелы и переводит код в верхний регистр. validate_registration проверяет формат, кабинет и активный дубль. next_queue_order просматривает все статусы и возвращает максимум плюс один. transfer_ticket разрешён только для waiting, требует другой известный кабинет и после проверок меняет только cabinet и queue_order. Если перенести id1 в кабинет 103, его id остаётся 1, order становится 7, поэтому сначала вызывается id6 с order6, затем id1.",
        ["Чем id отличается от order?", "Почему учитываются done/canceled?", "Почему список не сортируется?"],
    )
    speech_card(
        c, 15, "Хеда", "День 3",
        ["append_event", "logged-функции", "validate_project_state", "save_state / load_state", "build_work_document"],
        "На третьем дне появился единый state и журнал успешных действий. Обёртки logged сначала вызывают предметную функцию. Если получен отказ, история не меняется. После успеха append_event сохраняет независимые копии before и after и назначает следующий event_id. Состояние сохраняется в JSON вместе с records, history и extra. При восстановлении файл сначала читается в отдельный candidate, затем проверяется версия, проект и предметные ограничения. Только после этого он заменяет текущий state. Поэтому чужой или повреждённый файл не портит открытую работу.",
        ["Когда записывается событие?", "Зачем deepcopy?", "Почему нужен candidate?"],
    )
    speech_card(
        c, 16, "Ира", "День 4",
        ["requeue_ticket", "validate_registration с excluded id", "next_queue_order", "интерфейс возврата", "check_day4"],
        "На четвёртом дне я показываю возвращение отменённого талона. Функция находит запись и разрешает действие только для canceled. Затем повторно использует validate_registration, передавая исключаемый id, чтобы талон не считался своим дублем. После всех проверок статус становится waiting, а порядок — максимальным плюс один. Для №5 это order7, но id и кабинет сохраняются. Повторный клик получает отказ, не создаёт order8 и не добавляет второе событие. Восьмая проверка подтверждает правильную очередь: после возврата вызовы в 101 идут 1, 2, 5.",
        ["Почему №5 получает order7?", "Что меняется при отказе?", "Какие поля сохраняются?"],
    )

    c.save()
    print(OUTPUT)


if __name__ == "__main__":
    build_pdf()
