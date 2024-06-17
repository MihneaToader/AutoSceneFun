@echo off

:: Get the current directory
set "CURRENT_DIR=%cd%"

:: Check if the script is not executed from the offlineProcessing folder
echo %CURRENT_DIR% | findstr /i /c:"offlineProcessing" >nul
if %errorlevel% neq 0 (
    echo Please navigate to the 'offlineProcessing' directory and run the setup script from there.
    exit /b 1
)

:: Default values for models to download
set "DOWNLOAD_LITE=true"
set "DOWNLOAD_FULL=true"
set "DOWNLOAD_HEAVY=true"

:: Parse command-line arguments (only download desired models)
:parse_args
if "%~1"=="" goto :create_conda_env
if "%~1"=="--model" (
    set "model=%~2"
    if "%model%"=="lite" (
        set "DOWNLOAD_FULL=false"
        set "DOWNLOAD_HEAVY=false"
    ) else if "%model%"=="full" (
        set "DOWNLOAD_LITE=false"
        set "DOWNLOAD_HEAVY=false"
    ) else if "%model%"=="heavy" (
        set "DOWNLOAD_LITE=false"
        set "DOWNLOAD_FULL=false"
    ) else (
        echo Invalid model specified. Valid options are 'lite', 'full', or 'heavy'.
        exit /b 1
    )
    shift
    shift
) else (
    echo Unknown option: %~1
    exit /b 1
)
shift
goto :parse_args

:create_conda_env
:: Create conda environment (if not already created)
conda env list | findstr /i /c:"3dv" >nul
if %errorlevel% neq 0 (
    echo Creating conda environment...
    conda env create -f windows_environment.yml
    conda env list
    if not %errorlevel% == 0 (
        echo Failed to create conda environment. Exiting.
        echo Tip: Make sure conda is available and the environment.yml file is correctly configured.
        exit /b 1
    )
) else (
    echo Conda environment already exists.
)

:: Create models folder if it doesn't exist
if not exist models (
    mkdir models
)

:: Change directory to models folder
cd models || exit /b 1

echo Starting download of body-pose models...

:: Download files
if "%DOWNLOAD_LITE%"=="true" (
    echo Downloading Pose landmarker lite...
    curl -O https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task
)


if "%DOWNLOAD_FULL%"=="true" (
    echo Downloading Pose landmarker full...
    curl -O https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task
)

if "%DOWNLOAD_HEAVY%"=="true" (
    echo Downloading Pose landmarker heavy...
    curl -O https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task
)
call conda activate 3dv
python -m pip install git+https://github.com/m-bain/whisperx.git

echo Download completed.

echo Setup completed. Use 'conda activate 3dv' to activate virtual environment.
