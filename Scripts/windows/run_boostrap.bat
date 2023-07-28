set BASE_CONDA="C:\Users\l.corzo\AppData\Local\miniconda3"
set PYTHONPATH="path_to_the_template_folder"
call %BASE_CONDA%\Scripts\activate.bat %BASE_CONDA%
call conda activate dataingest_localdev_env
cd %PYTHONPATH%
call rm -rf .git
call rm -rf .github
call python Code/Bootsrap/bootsrap.py  ^
--directory %PYTHONPATH% ^
--project_name "monitoring" ^
--ProjectName "Monitoring" ^
--sonar_key "Monitoring"
