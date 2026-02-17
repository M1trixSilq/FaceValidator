import os
import sys
import cv2
import sqlite3
import numpy as np
import tkinter as tk
from tkinter import simpledialog, messagebox
from PIL import Image, ImageTk

# Функция для получения пути к классификатору
CASCADE_SCALE = 1.2
CASCADE_MIN_NEIGHBORS = 5
CASCADE_MIN_SIZE = (40, 40)
DETECTION_INTERVAL = 3  # Детектируем не на каждом кадре, чтобы ускорить работу
RECOGNITION_THRESHOLD = 0.45
FACE_VECTOR_SIZE = (64, 64)


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
    cursor.execute('INSERT INTO faces (name, encoding) VALUES (?, ?)', 
                   (name, encoding.tobytes()))
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
    distances = np.linalg.norm(candidates - encoding, axis=1)
    best_idx = int(np.argmin(distances))
    best_distance = float(distances[best_idx])

    if best_distance <= RECOGNITION_THRESHOLD:
        return known_names[best_idx], best_distance
    return "Unknown", best_distance


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

window_width, window_height = 640, 480
root.geometry(f"{window_width}x{window_height}")

# Виджет для отображения видео
label = tk.Label(root)
label.pack()

# Статусная метка
status_label = tk.Label(root, text="Ожидание для распознавания...", font=("Arial", 14))
status_label.pack()

# Состояние
last_faces = []
frame_counter = 0

def capture_face():  
    """Сохраняет одно лицо в БД."""
    global known_face_encodings, known_face_names

    ret, frame = video_capture.read()
    if not ret:
        status_label.config(text="Ошибка: Не удалось получить кадр.")
        return

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=CASCADE_SCALE,
        minNeighbors=CASCADE_MIN_NEIGHBORS,
        minSize=CASCADE_MIN_SIZE,
    )

    if len(faces) == 0:
        status_label.config(text="Ошибка: Лицо не обнаружено.")
        return

    # Берём самое крупное лицо как главное
    box = max(faces, key=lambda f: f[2] * f[3])
    encoding = extract_face_encoding(gray, box)
    if encoding is None:
        status_label.config(text="Ошибка: Не удалось построить дескриптор лица.")
        return

    name = simpledialog.askstring("Ввод данных", "Введите фамилию, имя и отчество нового работника:")
    if not name:
        status_label.config(text="Ошибка: Имя не введено.")
        return

    add_face_to_db(name, encoding)
    known_face_encodings, known_face_names = load_known_faces()
    messagebox.showinfo("Успех", f"Лицо {name} добавлено в базу данных!")
    status_label.config(text="Лицо добавлено. Распознавание активно.")
    
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
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        title = name if score is None else f"{name} ({score:.2f})"
        cv2.putText(frame, title, (x + 6, y + h - 8), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb_frame).resize((window_width, window_height))
    img_tk = ImageTk.PhotoImage(img)

    label.img_tk = img_tk
    label.configure(image=img_tk)

    label.after(15, process_frame)


new_pass_button = tk.Button(root, text="Сфотографировать и добавить", command=capture_face, font=("Arial", 14))
new_pass_button.pack()

process_frame()


def on_close():
    video_capture.release()
    root.destroy()


root.protocol("WM_DELETE_WINDOW", on_close)
root.mainloop()
