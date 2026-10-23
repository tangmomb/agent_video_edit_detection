@echo off
rem Lance l'interface de test du modele puis ouvre le navigateur.
rem Fermer cette fenetre (ou Ctrl+C) arrete le serveur.
chcp 65001 >nul
cd /d "%~dp0.."

if not exist ".venv\Scripts\python.exe" (
    echo Environnement .venv introuvable dans %cd%
    pause
    exit /b 1
)

rem Le modele met quelques secondes a se charger : le navigateur s'ouvre un peu apres.
start "" /b cmd /c "timeout /t 10 /nobreak >nul & start http://localhost:8780"

".venv\Scripts\python.exe" interface\server.py %*
pause
