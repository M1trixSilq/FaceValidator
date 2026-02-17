import os
import sys
import cv2
import sqlite3
import numpy as np
import tkinter as tk
from tkinter import simpledialog, messagebox
from PIL import Image, ImageDraw, ImageFont, ImageTk
# Функция для получения пути к классификатору
CASCADE_SCALE = 1.2
CASCADE_MIN_NEIGHBORS = 5
CASCADE_MIN_SIZE = (40, 40)
DETECTION_INTERVAL = 3  # Детектируем не на каждом кадре, чтобы ускорить работу
RECOGNITION_THRESHOLD = 0.40
FACE_VECTOR_SIZE = (64, 64)

RTK_BG = "#1A0F3D"
RTK_CARD = "#2A1B59"
RTK_ACCENT = "#7A2BF0"
RTK_ACCENT_2 = "#F16822"
RTK_TEXT = "#F5F2FF"
RTK_MUTED = "#CFC7F5"
NAME_FONT_SIZE = 20


def load_cyrillic_font(size):
    """Возвращает шрифт с поддержкой кириллицы для подписи имён."""
    font_candidates = [
        "arial.ttf",
        "Arial.ttf",
        "DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/Library/Fonts/Arial.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ]

    for font_path in font_candidates:
        try:
            return ImageFont.truetype(font_path, size)
        except OSError:
            continue
    return ImageFont.load_default()

def get_classifier_path():
    """Возвращает путь к Haar-каскаду для обычного запуска и PyInstaller."""
    if getattr(sys, "frozen", False):
        return os.path.join(sys._MEIPASS, "haarcascade_frontalface_default.xml")
    return "haarcascade_frontalface_default.xml"


# Функция для подключения к базе данных SQLite
def connect_db():
    return sqlite3.connect("faces_db.sqlite")


# Функция для создания таблицы, если она не существует
def create_table():
    """Создаёт таблицу для хранения имени и дескриптора лица."""
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS faces (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            encoding BLOB NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()

# Функция для добавления нового лица в базу данных

def add_face_to_db(name, encoding):
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO faces (name, encoding) VALUES (?, ?)",
        (name, encoding.astype(np.float32).tobytes()),
    )
    conn.commit()
    conn.close()

# Функция для загрузки всех известных лиц из базы данных

def load_known_faces():
    """Загружает валидные дескрипторы лиц и имена из БД."""
    encodings = []
    names = []

    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute("SELECT name, encoding FROM faces")
    rows = cursor.fetchall()
    conn.close()
    expected_size = FACE_VECTOR_SIZE[0] * FACE_VECTOR_SIZE[1]
    for name, blob in rows:
        vec = np.frombuffer(blob, dtype=np.float32)
        if vec.size != expected_size:
            # Старые записи (например [x, y, w, h]) пропускаем, чтобы не ломать распознавание.
            continue
        encodings.append(vec)
        names.append(name)

    return encodings, names


def extract_face_encoding(gray_frame, box):
    """Формирует быстрый и устойчивый дескриптор из ROI лица."""
    x, y, w, h = box
    face_roi = gray_frame[y : y + h, x : x + w]

# Проверяем, доступна ли камера
    if face_roi.size == 0:
        return None

    # Гистограммная нормализация + масштабирование повышают устойчивость к освещению.
    face_roi = cv2.equalizeHist(face_roi)
    face_roi = cv2.resize(face_roi, FACE_VECTOR_SIZE, interpolation=cv2.INTER_AREA)

    vec = face_roi.astype(np.float32).reshape(-1)
    norm = np.linalg.norm(vec)
    if norm <= 1e-8:
        return None

    return vec / norm


def match_face(encoding, known_encodings, known_names):
    """Находит наиболее похожее лицо по косинусной близости (через L2 на нормализованных векторах)."""
    if encoding is None or not known_encodings:
        return "Unknown", None

    candidates = np.vstack(known_encodings)
    similarities = candidates @ encoding
    best_idx = int(np.argmax(similarities))
    best_similarity = float(similarities[best_idx])

    if best_similarity >= RECOGNITION_THRESHOLD:
        return known_names[best_idx], best_similarity
    return "Unknown", best_similarity


# Инициализация камеры
video_capture = cv2.VideoCapture(0)
if not video_capture.isOpened():
    print("Error: Camera not accessible")
    raise SystemExit(1)

# Буферизация=1 уменьшает задержку в некоторых драйверах
video_capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# Создание таблицы, если она не существует
# Инициализация распознавания
face_cascade = cv2.CascadeClassifier(get_classifier_path())
create_table()

# Загрузка известных лиц из базы данных
known_face_encodings, known_face_names = load_known_faces()

# Инициализация окна Tkinter
# UI
root = tk.Tk()
root.title("Распознавание лиц для пропускной системы")
root.configure(bg=RTK_BG)

window_width, window_height = 760, 700
root.geometry(f"{window_width}x{window_height}")
name_font = load_cyrillic_font(NAME_FONT_SIZE)

def build_rostelecom_logo():
    """Загружает логотип из папки с main.py; если файла нет — рисует компактный fallback."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    logo_candidates = [
        "rostelecom_logo.png"
    ]

    for name in logo_candidates:
        logo_path = os.path.join(base_dir, name)
        if os.path.exists(logo_path):
            logo = Image.open(logo_path).convert("RGBA")
            logo.thumbnail((170, 42), Image.Resampling.LANCZOS)
            canvas = Image.new("RGBA", (170, 42), (0, 0, 0, 0))
            x = (170 - logo.width) // 2
            y = (42 - logo.height) // 2
            canvas.paste(logo, (x, y), logo)
            return ImageTk.PhotoImage(canvas)

    logo = Image.new("RGBA", (170, 42), (0, 0, 0, 0))
    draw = ImageDraw.Draw(logo)
    draw.polygon([(6, 32), (20, 8), (34, 32)], fill=RTK_ACCENT)
    draw.polygon([(28, 32), (42, 8), (56, 32)], fill=RTK_ACCENT_2)

    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except OSError:
        font = ImageFont.load_default()
    draw.text((66, 11), "Ростелеком", fill=RTK_MUTED, font=font)
    return ImageTk.PhotoImage(logo)

header = tk.Frame(root, bg=RTK_BG)
header.pack(fill="x", padx=18, pady=(14, 6))

title_label = tk.Label(
    header,
    text="Пропускная система",
    bg=RTK_BG,
    fg=RTK_TEXT,
    font=("Arial", 20, "bold"),
)
title_label.pack(side="left")

logo_tk = build_rostelecom_logo()
logo_label = tk.Label(header, image=logo_tk, bg=RTK_BG, bd=0)
logo_label.image = logo_tk
logo_label.pack(side="right")

video_card = tk.Frame(root, bg=RTK_CARD, bd=0, highlightthickness=2, highlightbackground=RTK_ACCENT)
video_card.pack(fill="both", expand=True, padx=18, pady=10)


# Виджет для отображения видео
label = tk.Label(video_card, bg=RTK_CARD)
label.pack(fill="both", expand=True, padx=10, pady=10)

# Статусная метка
status_label = tk.Label(
    root,
    text="Ожидание для распознавания...",
    font=("Arial", 13),
    fg=RTK_MUTED,
    bg=RTK_BG,
)
status_label.pack(pady=(0, 10))

# Состояние
last_faces = []
frame_counter = 0

def capture_face():  
    """Сохраняет одно лицо в БД."""
    global known_face_encodings, known_face_names

    ret, frame = video_capture.read()
    if not ret:
        status_label.config(text="Ошибка: Не удалось получить кадр.", fg=RTK_ACCENT_2)
        return

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=CASCADE_SCALE,
        minNeighbors=CASCADE_MIN_NEIGHBORS,
        minSize=CASCADE_MIN_SIZE,
    )

    if len(faces) == 0:
        status_label.config(text="Ошибка: Лицо не обнаружено.", fg=RTK_ACCENT_2)
        return

    # Берём самое крупное лицо как главное
    box = max(faces, key=lambda f: f[2] * f[3])
    encoding = extract_face_encoding(gray, box)
    if encoding is None:
        status_label.config(text="Ошибка: Не удалось построить дескриптор лица.", fg=RTK_ACCENT_2)
        return

    name = simpledialog.askstring("Ввод данных", "Введите фамилию, имя и отчество нового работника:")
    if not name:
        status_label.config(text="Ошибка: Имя не введено.", fg=RTK_ACCENT_2)
        return

    add_face_to_db(name, encoding)
    known_face_encodings, known_face_names = load_known_faces()
    messagebox.showinfo("Успех", f"Лицо {name} добавлено в базу данных!")
    status_label.config(text="Лицо добавлено. Распознавание активно.", fg=RTK_MUTED)

def process_frame():
    """Единый цикл: захват, детекция, распознавание, отрисовка."""
    global frame_counter, last_faces

    ret, frame = video_capture.read()
    if not ret:
        label.after(30, process_frame)
        return

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Детекция не на каждом кадре для скорости.
    if frame_counter % DETECTION_INTERVAL == 0:
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=CASCADE_SCALE,
            minNeighbors=CASCADE_MIN_NEIGHBORS,
            minSize=CASCADE_MIN_SIZE,
        )

        refreshed = []
        for box in faces:
            encoding = extract_face_encoding(gray, box)
            name, score = match_face(encoding, known_face_encodings, known_face_names)
            refreshed.append((box, name, score))
        last_faces = refreshed

    frame_counter += 1

    for (x, y, w, h), name, score in last_faces:
        is_match = (score is not None) and (score > RECOGNITION_THRESHOLD) and (name != "Unknown")
        box_color = (0, 190, 0) if is_match else (242, 104, 34)
        cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    display_img = Image.fromarray(rgb_frame)
    draw = ImageDraw.Draw(display_img)

    for (x, y, w, h), name, score in last_faces:
        title = name if score is None else f"{name} ({score:.2f})"
        text_x = x + 6
        text_y = max(6, y - NAME_FONT_SIZE - 8)
        draw.text((text_x, text_y), title, fill=(255, 255, 255), font=name_font)

    img = display_img.resize((window_width - 56, window_height - 240))
    img_tk = ImageTk.PhotoImage(img)

    label.img_tk = img_tk
    label.configure(image=img_tk)

    label.after(15, process_frame)


new_pass_button = tk.Button(
    root,
    text="Сфотографировать и добавить",
    command=capture_face,
    font=("Arial", 13, "bold"),
    bg=RTK_ACCENT,
    fg="white",
    activebackground=RTK_ACCENT_2,
    activeforeground="white",
    bd=0,
    padx=18,
    pady=10,
    cursor="hand2",
)
new_pass_button.pack(pady=(2, 18))

process_frame()


def on_close():
    video_capture.release()
    root.destroy()


root.protocol("WM_DELETE_WINDOW", on_close)
root.mainloop()
