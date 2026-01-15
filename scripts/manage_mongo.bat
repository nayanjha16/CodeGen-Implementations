@echo off
echo MongoDB Service Manager
echo -----------------------

if "%1"=="start" (
    echo Starting MongoDB...
    net start MongoDB
) else if "%1"=="stop" (
    echo Stopping MongoDB...
    net stop MongoDB
) else if "%1"=="restart" (
    echo Restarting MongoDB...
    net stop MongoDB
    net start MongoDB
) else (
    echo Usage: manage_mongo.bat [start|stop|restart]
    echo Note: This must be run as Administrator.
)
pause
