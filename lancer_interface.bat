@echo off
rem Lance les deux interfaces en mode rechargement automatique, puis ouvre le navigateur :
rem - interface de test du modele : http://localhost:8780
rem - banc des annotateurs Mistral : http://localhost:8790
rem Chaque serveur redemarre quand un fichier .py du projet change, et sa page se recharge
rem toute seule (HTML modifie ou serveur redemarre). Fermer une fenetre arrete son serveur.
chcp 65001 >nul
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Environnement .venv introuvable dans %cd%
    pause
    exit /b 1
)

start "Interface du modele (8780)" cmd /k ".venv\Scripts\python.exe utils\dev_reload.py interface\server.py %*"
start "Banc des annotateurs (8790)" cmd /k ".venv\Scripts\python.exe utils\dev_reload.py utils\llm_bench\server.py"

rem Le modele met quelques secondes a se charger : le navigateur s'ouvre un peu apres.
timeout /t 10 /nobreak >nul
start http://localhost:8780
start http://localhost:8790
