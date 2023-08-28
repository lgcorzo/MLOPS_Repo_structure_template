set BASE_CONDA="C:\Users\l.corzo\AppData\Local\miniconda3"
set PYTHONPATH="path_to_the_template_folder"
call %BASE_CONDA%\Scripts\activate.bat %BASE_CONDA%
call conda activate code_development_env
cd %PYTHONPATH%
call rmdir /s /q .git
call rmdir /s /q .github
call python Code/Bootstrap/bootstrap.py  ^
--directory %PYTHONPATH% ^
--project_name "monitoring" ^
--ProjectName "Monitoring" ^
--sonar_key "Monitoring"
