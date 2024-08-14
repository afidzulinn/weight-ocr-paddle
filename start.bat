@echo off

REM Activate the virtual environment
call venv\Scripts\activate.bat

REM Run the uvicorn server
uvicorn app:app --port 9123 --reload