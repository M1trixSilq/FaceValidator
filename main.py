import os
import sys
import cv2
import sqlite3
import numpy as np
import tkinter as tk
from tkinter import simpledialog, messagebox
from PIL import Image, ImageTk

# Функция для получения пути к классификатору
def get_classifier_path():
    if getattr(sys, 'frozen', False):  # Проверяем, запущена ли программа как .exe
        return os.path.join(sys._MEIPASS, 'haarcascade_frontalface_default.xml')
    else:
        return 'haarcascade_frontalface_default.xml'  # Путь для разработки

# Функция для подключения к базе данных SQLite
def connect_db():
    conn = sqlite3.connect('faces_db.sqlite')  # Имя базы данных
    return conn

# Функция для создания таблицы, если она не существует
def create_table():
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS faces (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT NOT NULL,
                        encoding BLOB NOT NULL)''')
    conn.commit()
    conn.close()

# Функция для добавления нового лица в базу данных
def add_face_to_db(name, encoding):
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute('INSERT INTO faces (name, encoding) VALUES (?, ?)', 
                   (name, encoding.tobytes()))
    conn.commit()
    conn.close()

# Функция для загрузки всех известных лиц из базы данных
def load_known_faces():
    known_face_encodings = []
    known_face_names = []
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute('SELECT name, encoding FROM faces')
    rows = cursor.fetchall()
    for row in rows:
        name = row[0]
        encoding = np.frombuffer(row[1], dtype=np.float64)  # Преобразуем обратно в массив
        known_face_encodings.append(encoding)
        known_face_names.append(name)
    conn.close()
    return known_face_encodings, known_face_names

# Захват видео с основной камеры
video_capture = cv2.VideoCapture(0)

# Проверяем, доступна ли камера
if not video_capture.isOpened():
    print("Error: Camera not accessible")
    exit()

# Загрузка классификатора для распознавания лиц
face_cascade = cv2.CascadeClassifier(get_classifier_path())

# Создание таблицы, если она не существует
create_table()

# Загрузка известных лиц из базы данных
known_face_encodings, known_face_names = load_known_faces()

# Инициализация окна Tkinter
root = tk.Tk()
root.title("Распознавание лиц для пропускной системы")

# Размеры окна
window_width = 640
window_height = 480
root.geometry(f"{window_width}x{window_height}")

# Виджет для отображения видео
label = tk.Label(root)
label.pack()

# Статусная метка
status_label = tk.Label(root, text="Ожидание для распознавания...", font=("Arial", 14))
status_label.pack()

# Состояние кнопки
is_recording = False

# Функция для отображения видео в Tkinter
def update_video_frame():
    ret, frame = video_capture.read()
    if ret:
        # Преобразуем изображение для отображения в Tkinter
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb_frame)
        img = img.resize((window_width, window_height))
        img_tk = ImageTk.PhotoImage(img)

        label.img_tk = img_tk
        label.configure(image=img_tk)

    # Запланировать обновление через 10ms
    label.after(10, update_video_frame)

# Функция для добавления нового лица
def add_new_face():
    global is_recording
    is_recording = True
    status_label.config(text="Нажмите кнопку 'Сфотографировать' для добавления нового лица.")

# Функция для сфотографировать лицо
def capture_face():
    global is_recording
    status_label.config(text="Сфотографировано! Введите данные нового работника.")

    # Захват одного кадра для нового лица
    ret, frame = video_capture.read()
    if not ret:
        print("Error: Failed to capture frame")
        return

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Находим все лица на этом кадре
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) > 0:
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        
        # Ожидаем ввода данных о новом работнике
        name = simpledialog.askstring("Ввод данных", "Введите фамилию, имя и отчество нового работника:")
        if name:
            # Преобразуем лицо в "encoding" (в данном случае просто сохранение данных)
            encoding = np.array([x, y, w, h])  # Моделируем encoding (должен быть заменен на реальное распознавание)
            add_face_to_db(name, encoding)
            messagebox.showinfo("Успех", f"Лицо {name} добавлено в базу данных!")
            known_face_encodings, known_face_names = load_known_faces()

            # После добавления нового лица начинаем его распознавать
            is_recording = False
            status_label.config(text="Распознавание лиц включено.")
        else:
            status_label.config(text="Ошибка: Имя не введено.")
    else:
        status_label.config(text="Ошибка: Лицо не обнаружено.")

# Функция для обновления распознавания лиц
def recognize_faces():
    global is_recording
    ret, frame = video_capture.read()

    if not ret:
        return

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Находим все лица на кадре
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    face_names = []

    for (x, y, w, h) in faces:
        name = "Unknown"

        # Можно добавить проверку с базой данных известных лиц, как в предыдущем примере
        # Для простоты пока добавляем рамки

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(frame, name, (x + 6, y + h - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)

    # Отображаем результат
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    img = img.resize((window_width, window_height))
    img_tk = ImageTk.PhotoImage(img)

    label.img_tk = img_tk
    label.configure(image=img_tk)

    # Запланировать распознавание через 100ms
    label.after(100, recognize_faces)

# Кнопка "Новый пропуск"
new_pass_button = tk.Button(root, text="Новый пропуск", command=add_new_face, font=("Arial", 14))
new_pass_button.pack()

# Кнопка "Сфотографировать"
capture_button = tk.Button(root, text="Сфотографировать", command=capture_face, font=("Arial", 14))
capture_button.pack()

# Инициализация потока видео
update_video_frame()
recognize_faces()

# Запуск интерфейса
root.mainloop()