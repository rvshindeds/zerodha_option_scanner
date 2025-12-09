@echo off
setlocal

REM Move to the folder where this BAT lives
cd /d "%~dp0"

REM Activate venv
call .venv\Scripts\activate.bat

REM Run Streamlit
streamlit run app.py

endlocal
