@echo off
title WSB Sentiment Analyzer
echo.
echo  🚀 Starting WSB Sentiment Analyzer...
echo.
cd /d "%~dp0"
streamlit run app.py
pause
