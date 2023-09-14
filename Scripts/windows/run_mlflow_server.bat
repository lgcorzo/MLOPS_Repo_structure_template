set BASE_CONDA="C:\Users\l.corzo\AppData\Local\miniconda3"
set PYTHONPATH="C:\Users\l.corzo\source\repos\MLOPS_Repo_structure_template"
call %BASE_CONDA%\Scripts\activate.bat %BASE_CONDA%
call conda activate code_development_env
cd %PYTHONPATH%
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./artifacts --host 127.0.0.1 -p 1234