@echo off
python main.py
gunicorn app:app
