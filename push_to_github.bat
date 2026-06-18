@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo === AgentDesk 推送到 GitHub ===
git init
git add .
git commit -m "AgentDesk: Agentic RAG + LangGraph + Streamlit dashboard"
git branch -M main
git remote remove origin 2>nul
git remote add origin https://github.com/XIAOHAY/agentdesk.git
git push -u origin main
echo.
echo === 完成。若提示登录，按浏览器/凭据管理器提示登录 GitHub 即可。===
pause
