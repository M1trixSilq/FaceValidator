python -m venv .venv

pip install -r requirements.txt

pyinstaller --onefile --add-data "haarcascade_frontalface_default.xml;." --add-data "rostelecom_logo.png;." main.py