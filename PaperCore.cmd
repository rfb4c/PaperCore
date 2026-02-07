@echo off
title PaperCore
cd /d "%~dp0"
pythonw papercore_gui.py
if errorlevel 1 (
    echo Starting with python instead...
    python papercore_gui.py
    pause
)
