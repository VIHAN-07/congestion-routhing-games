@echo off
echo Starting Game Theory Interactive Platform...
echo.
echo The app will open in your browser at http://localhost:8507
echo Press Ctrl+C to stop the server
echo.
C:/Users/Lenovo/AppData/Local/Microsoft/WindowsApps/python3.11.exe -m streamlit run working_interactive_ui.py --server.port 8507
pause
