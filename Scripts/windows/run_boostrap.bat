set BASE_CONDA="C:\Users\l.corzo\AppData\Local\miniconda3"
set PYTHONPATH="C:\Users\l.corzo\source\repos\MLOPS_Repo_structure_template"
call %BASE_CONDA%\Scripts\activate.bat %BASE_CONDA%
call conda activate dataingest_localdev_env
cd %PYTHONPATH%
call python -m Code.Bootstrap.bootstrap.py  ^
--directory %PYTHONPATH% ^
--project_name "Monitoring" ^
--sonar_key "Monitoring" ^

