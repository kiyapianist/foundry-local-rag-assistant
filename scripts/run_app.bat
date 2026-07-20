@echo off
cd /d "%~dp0.."
set RAG_OFFLINE_MODE=1
".\venv\Scripts\streamlit.exe" run app.py --server.port 8501 --server.address 127.0.0.1 --server.headless true --browser.gatherUsageStats false
