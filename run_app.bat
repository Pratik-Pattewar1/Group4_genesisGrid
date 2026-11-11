@echo off
REM Start the DeliveryIQ Streamlit app with the project venv.
REM (The venv's streamlit.exe launcher still points to the old D:\Project1 path, so call python -m.)
cd /d "%~dp0"
".venv\Scripts\python.exe" -m streamlit run app\streamlit_app.py
