@echo off
setlocal EnableDelayedExpansion

:: ================= CONFIGURATION =================
set GENERATOR=python generate.py
set SOLVER=solution.exe
set LOG=benchmark_results.txt
:: Using sm_75 for compatibility (RTX 4050/3050 etc)
set NVCC_CMD=nvcc solution.cu -o solution.exe -arch=sm_75
:: =================================================

echo ========================================================
echo               CUDA BENCHMARK SCRIPT
echo ========================================================

:: 1. Compile
echo [INFO] Compiling...
%NVCC_CMD%
if %errorlevel% neq 0 (
    echo [ERROR] Compilation failed!
    pause
    exit /b %errorlevel%
)
echo [INFO] Success.
echo.

:: Init Log
echo CUDA Benchmark Results - %DATE% %TIME% > %LOG%
echo -------------------------------------------------------- >> %LOG%

:: TEST 1: 10x10
echo [SETUP] Generating 10x10 Matrix...
%GENERATOR% 10 10
call :RunSuite 10 10

:: TEST 2: 1000x1000
echo.
echo [SETUP] Generating 1000x1000 Matrix...
%GENERATOR% 1000 1000
call :RunSuite 1000 1000

:: TEST 3: 10000x10000
echo.
echo [SETUP] Generating 10000x10000 Matrix...
%GENERATOR% 10000 10000
call :RunSuite 10000 10000

echo.
echo [SUCCESS] Done. Saved to %LOG%
pause
goto :eof

:: ================= FUNCTION =================
:RunSuite
set M=%1
set N=%2

echo --------------------------------------------------------
echo [TEST] Matrix: %M%x%N%
echo [TEST] Matrix: %M%x%N% >> %LOG%

set total_time=0

for /L %%i in (1,1,10) do (
    :: Run executable and filter output for "Execution"
    for /f "tokens=3" %%t in ('%SOLVER% %M% %N% ^| find "Execution"') do (
        set current_time=%%t
        echo     Run %%i: !current_time! ms

        :: Accumulate Time
        for /f "usebackq" %%v in (`powershell -Command "!total_time! + !current_time!"`) do (
            set total_time=%%v
        )
    )
)

:: Calculate Average
for /f "usebackq" %%a in (`powershell -Command "%total_time% / 10"`) do (
    set avg_time=%%a
)

echo   >> AVERAGE Time: %avg_time% ms
echo   >> AVERAGE Time: %avg_time% ms >> %LOG%
echo -------------------------------------------------------- >> %LOG%

exit /b