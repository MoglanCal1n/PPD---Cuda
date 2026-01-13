@echo off
setlocal EnableDelayedExpansion

:: ================= CONFIGURATION =================
set GENERATOR=python generate.py
set SOLVER_GPU=solution.exe
set SOLVER_CPU=sequential.exe
set LOG=benchmark_results.txt
set NVCC_CMD=nvcc solution.cu -o solution.exe -arch=sm_89
set CPP_CMD=cl /EHsc /O2 sequential.cpp /Fe:sequential.exe
:: =================================================

echo ========================================================
echo        CUDA BENCHMARK (ADAPTAT: 64, 128, 256, 512)
echo ========================================================

:: 1. Compile
echo [INFO] Compiling...
%NVCC_CMD%
%CPP_CMD%
if %errorlevel% neq 0 (
    echo [ERROR] Compilation failed!
    pause
    exit /b %errorlevel%
)
echo [INFO] Success.
echo.

:: Init Log
echo CUDA Lab 2 Benchmark - %DATE% %TIME% > %LOG%
echo -------------------------------------------------------- >> %LOG%

:: ==============================================================
:: TEST 1: N=M=10 (Cerința: p=2 + executie secventiala)
:: Adaptare: Rulam cu 64 threads (cel mai mic block size uzual)
:: ==============================================================
echo [CASE 1] Matrix: 10x10
echo. >> %LOG%
echo [CASE 1] Matrix: 10x10 >> %LOG%

:: Generam datele
%GENERATOR% 10 10 >nul 2>&1

:: 1.1 Secvential
call :RunSequential 10 10

:: 1.2 CUDA (Rulam cu 64 threads)
call :RunCuda 10 10 64

:: ==============================================================
:: TEST 2: N=M=1000 (Cerința: variatie p)
:: Adaptare CUDA: Variem BlockSize (64, 128, 256, 512)
:: ==============================================================
echo.
echo [CASE 2] Matrix: 1000x1000
echo. >> %LOG%
echo [CASE 2] Matrix: 1000x1000 >> %LOG%

:: Generam datele
%GENERATOR% 1000 1000 >nul 2>&1

:: 2.1 Secvential
call :RunSequential 1000 1000

:: 2.2 CUDA (Variem Threads per Block)
for %%B in (64 128 256 512) do (
    call :RunCuda 1000 1000 %%B
)

:: ==============================================================
:: TEST 3: N=M=10000 (Cerința: variatie p)
:: ==============================================================
echo.
echo [CASE 3] Matrix: 10000x10000
echo. >> %LOG%
echo [CASE 3] Matrix: 10000x10000 >> %LOG%

:: Generam datele
%GENERATOR% 10000 10000 >nul 2>&1

:: 3.1 Secvential (Atentie: dureaza mult)
call :RunSequential 10000 10000

:: 3.2 CUDA (Variem Threads per Block)
for %%B in (64 128 256 512) do (
    call :RunCuda 10000 10000 %%B
)

echo.
echo [SUCCESS] Done. Saved to %LOG%
pause
goto :eof


:: ================= HELPERS =================

:RunSequential
set M=%1
set N=%2
echo   [CPU] Running Sequential %M%x%N%...
set total_time=0
:: Rulam de 3 ori pe CPU (pentru viteza)
for /L %%i in (1,1,3) do (
    set current_time=0
    for /f "tokens=3" %%t in ('%SOLVER_CPU% %M% %N% ^| find "Execution"') do (
        set current_time=%%t
        for /f "usebackq" %%v in (`powershell -Command "$t = !total_time! + !current_time!; $t"`) do (
            set total_time=%%v
        )
    )
)
for /f "usebackq" %%a in (`powershell -Command "$a = !total_time! / 3; [Math]::Round($a, 4)"`) do (
    set cpu_avg=%%a
)
echo     >> Avg CPU Time: !cpu_avg! ms
echo     >> Avg CPU Time: !cpu_avg! ms >> %LOG%
exit /b

:RunCuda
set M=%1
set N=%2
set B=%3
echo   [GPU] Running CUDA %M%x%N% (Threads: %B%)...
set total_time=0
:: Rulam de 10 ori pe GPU
for /L %%i in (1,1,10) do (
    set current_time=0
    for /f "tokens=3" %%t in ('%SOLVER_GPU% %M% %N% %B% ^| find "Execution"') do (
        set current_time=%%t
        for /f "usebackq" %%v in (`powershell -Command "$t = !total_time! + !current_time!; $t"`) do (
            set total_time=%%v
        )
    )
)
for /f "usebackq" %%a in (`powershell -Command "$a = !total_time! / 10; [Math]::Round($a, 4)"`) do (
    set gpu_avg=%%a
)
echo     >> Avg GPU Time (Block %B%): !gpu_avg! ms
echo     >> Avg GPU Time (Block %B%): !gpu_avg! ms >> %LOG%
exit /b