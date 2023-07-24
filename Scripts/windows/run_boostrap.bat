set root="path the the conda base env"
set PYTHONPATH="path to the root folder"
call %root%\Scripts\activate.bat %root%
call conda activate BU_template_env
cd %PYTHONPATH%
call python -m Notebooks.BusinessUnderstanding.bootstrap --tenantid "SSC" ^
--plantid "SSC_Plant" ^
--path_org_data "C:\Users\l.corzo\OneDrive - LANTEK SHEET METAL SOLUTIONS SL\LANTEK\working\Repositories\Data\Raw\Merlin" ^
--path_dst_data "C:\Users\l.corzo\OneDrive - LANTEK SHEET METAL SOLUTIONS SL\LANTEK\working\Repositories\Data\Staging\Merlin\Smartquoting" ^
--org_data_type "parquet"
