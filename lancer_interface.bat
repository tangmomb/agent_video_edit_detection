@echo off
rem Lance les serveurs en mode rechargement automatique, puis ouvre le panneau de controle
rem (http://localhost:8770) : tous les outils sur une seule adresse, menu en bas a gauche.
rem Chaque outil reste dans son dossier interface\NOM-DE-L-OUTIL\ et sur son port :
rem - interface de test du modele : http://localhost:8780
rem - banc des annotateurs : http://localhost:8790
rem - visionneuse du dataset : http://localhost:8800
rem - test du saut d'images avant annotation : http://localhost:8810
rem - conversations des agents par run : http://localhost:8820
rem - resume des runs : http://localhost:8830
rem Chaque serveur redemarre quand un fichier .py du projet change, et sa page se recharge
rem toute seule (HTML modifie ou serveur redemarre). Fermer une fenetre arrete son serveur.
rem Relancer le .bat ferme d'abord les serveurs deja lances, puis les redemarre.
chcp 65001 >nul
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Environnement .venv introuvable dans %cd%
    pause
    exit /b 1
)

rem Arret des serveurs d'un lancement precedent : leurs fenetres (avec dev_reload et le serveur),
rem puis tout ce qui ecoute encore sur leurs ports (serveur lance a la main, par exemple).
for %%t in ("Panneau de controle (8770)" "Interface du modele (8780)" "Banc des annotateurs (8790)" "Visionneuse du dataset (8800)" "Saut d'images (8810)" "Conversations des runs (8820)" "Resume des runs (8830)") do (
    taskkill /FI "WINDOWTITLE eq %%~t*" /T /F >nul 2>&1
)
powershell -NoProfile -Command "Get-NetTCPConnection -State Listen -LocalPort 8770,8780,8790,8800,8810,8820,8830 -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }"

start "Panneau de controle (8770)" cmd /k ".venv\Scripts\python.exe interface\dev_reload.py interface\accueil\server.py"
start "Interface du modele (8780)" cmd /k ".venv\Scripts\python.exe interface\dev_reload.py interface\modele\server.py %*"
start "Banc des annotateurs (8790)" cmd /k ".venv\Scripts\python.exe interface\dev_reload.py interface\llm_bench\server.py"
start "Visionneuse du dataset (8800)" cmd /k ".venv\Scripts\python.exe interface\dev_reload.py interface\dataset_viewer\server.py"
start "Saut d'images (8810)" cmd /k ".venv\Scripts\python.exe interface\dev_reload.py interface\skip_tester\server.py"
start "Conversations des runs (8820)" cmd /k ".venv\Scripts\python.exe interface\dev_reload.py interface\run_chat\server.py"
start "Resume des runs (8830)" cmd /k ".venv\Scripts\python.exe interface\dev_reload.py interface\runs_summary\server.py"

rem Le panneau demarre tout de suite sur le resume des runs ; son menu indique quand les autres serveurs sont en ligne.
timeout /t 2 /nobreak >nul
start http://localhost:8770
