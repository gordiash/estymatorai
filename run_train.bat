@echo off
setlocal enableextensions enabledelayedexpansion
python -m src.train_rf --test-days %TEST_DAYS%
endlocal
