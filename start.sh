#!/bin/bash

source venv\Scripts\activate.bat

uvicorn app:app --port 9123 --reload