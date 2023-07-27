set BASE_CONDA="C:\Users\l.corzo\AppData\Local\miniconda3"
set PYTHONPATH="C:\Users\l.corzo\source\repos\MLOPS_Repo_structure_template"
call %BASE_CONDA%\Scripts\activate.bat %BASE_CONDA%
call conda activate code_development_env
cd %PYTHONPATH%
call python -m  Code.Controller.app 