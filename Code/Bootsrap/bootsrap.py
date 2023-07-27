import os
import sys
import platform
import argparse
import re
import glob


class Helper:

    def __init__(self, project_directory, project_name, projectname, sonar_key):
        self._project_directory = project_directory
        self._project_name = project_name
        self._projectname = projectname
        self._git_repo = "https://github.com/lgcorzo/MLOPS_Repo_structure_template.git"
        self._sonar_key = sonar_key

    @property
    def project_directory(self):
        return self._project_directory

    @property
    def project_name(self):
        return self._project_name

    @property
    def git_repo(self):
        return self._git_repo

    @property
    def sonar_key(self):
        return self._sonar_key

    @property
    def ProjectName(self):
        return self._projectname

    def rename_files(self):
        # Rename all files containg  project_name in the name
        strtoreplace = "project_name"
        search_pattern = f"{self._project_directory}/**/*{strtoreplace}*"
        dirs = glob.glob(search_pattern, recursive=True)

        for dir in dirs:
            normDir = os.path.normpath(dir)
            dirpath = os.path.join(self._project_directory, normDir)
            for filename in os.listdir(dirpath):
                if (filename.find(strtoreplace) != -1):
                    src = os.path.join(self._project_directory, normDir, filename)  # NOQA: E501
                    dst = os.path.join(self._project_directory,
                                       normDir,
                                       filename.replace(strtoreplace, self._project_name))  # NOQA: E501
                    os.rename(src, dst)

    def rename_dir(self):
        dirname = "ProjectName"
        src = os.path.join(self._project_directory, dirname)
        for path, subdirs, files in os.walk(src):
            for name in files:
                newpath = path.replace(dir, self._project_name)
                if (not (os.path.exists(newpath))):
                    os.mkdir(newpath)
                file_path = os.path.join(path, name)
                new_name = os.path.join(newpath, name).lower()
                os.rename(file_path, new_name)

    def delete_dir(self):
        # Delete unwanted directories
        dirs = [
            "Data/Archive",
            "Data/Core",
            "Data/Results",
            "Notebooks/Commissioning",
            "Notebooks/DataIngestion",
            "Notebooks/DataUnderstanding",
            "Notebooks/Deployment",
            "Notebooks/Experimenting",
            "Notebooks/Modelling",
            "Notebooks/Monitoring",
            "Notebooks/Testing",
            # "Code/FrontEnd/assets",
            # "Code/FrontEnd",
            # "Code/Controller",
            # "Code/Domain/Models",
            # "Code/Domain",
            # "Code/Application/Services",
            # "Code/Application",
            "Code/Bootsrap"
        ]
        if (platform.system() == "Windows"):
            cmd = 'rmdir /S /Q "{}"'
        else:
            cmd = 'rm -r "{}"'
        for dir in dirs:
            os.system(cmd.format(os.path.join(self._project_directory, os.path.normpath(dir))))  # NOQA: E501

    def clean_dir(self):
        # Clean up directories
        dirs = ["Data/Archive",
                "Data/Core",
                "Scripts/windows/",
                "Data/Results",
                "Notebooks/Commissioning",
                "Notebooks/DataIngestion",
                "Notebooks/DataUnderstanding",
                "Notebooks/Deployment",
                "Notebooks/Experimenting",
                "Notebooks/Modelling",
                "Notebooks/Monitoring",
                "Notebooks/Testing",
                # "Code/Controller",
                # "Code/Domain/Models",
                # "Code/Domain",
                # "Code/FrontEnd/assets",
                # "Code/FrontEnd",
                # "Code/Application/Services",
                # "Code/Application",
                "Code/Bootsrap"
                ]
        for dir in dirs:
            for root, dirs, files in os.walk(os.path.join(self._project_directory, dir)):  # NOQA: E501
                for file in files:
                    os.remove(os.path.join(root, file))

    def validate_args(self):
        # Validate arguments
        if (os.path.isdir(self._project_directory) is False):
            raise Exception("Not a valid directory. Please provide an absolute directory path.")  # NOQA: E501
        if (len(self._project_name) < 3 or len(self._project_name) > 15):
            raise Exception("Invalid project name length. Project name should be 3 to 15 chars long, letters and underscores only.")  # NOQA: E501
        if (not re.search("^[\\w_]+$", self._project_name)):
            raise Exception(
                "Invalid characters in project name. Project name should be 3 \
                to 15 chars long, letters and underscores only.")
        if (not re.search("^[\\w_]+$", self._sonar_key)):
            raise Exception("Invalid characters in project name. Project name should be 3 to 15 chars long, letters and underscores only.")  # NOQA: E501
# NOQA: E501


def replace_project_name(project_dir, project_name, rename_name):
    # Replace instances of rename_name within files with project_name
    files = [
        r"Pipelines/DevopsPipelines/ci_build_ProjectName.yaml",
        r"Notebooks/BusinessUnderstanding/BU_Project_env_init_notebook.ipynb",
        r"Code/Application/project_name_algorithm.py",
        r"Code/Application/project_name_evaluation.py",
        r"Code/Application/project_name_model.py",
        r"Code/FrontEnd/app.py",
        r"Code/Domain/Models/project_name.py"
    ]

    for file in files:
        path = os.path.join(project_dir, os.path.normpath(file))
        try:
            with open(path, "rt", encoding="utf8") as f_in:
                data = f_in.read()
            data = data.replace(rename_name, project_name)
            with open(os.path.join(project_dir, file), "wt", encoding="utf8") as f_out:  # NOQA: E501
                f_out.write(data)
        except IOError as e:
            print("Could not modify \"%s\". Is the MLOpsPython repo already cloned at \"%s\"?" % (path, project_dir))  # NOQA: E501
            raise e


def main(args):
    parser = argparse.ArgumentParser(description='New proejct init')
    parser.add_argument("-d",
                        "--directory",
                        type=str,
                        required=True,
                        help="Absolute path to new project direcory")
    parser.add_argument("-M",
                        "--ProjectName",
                        type=str,
                        required=True,
                        help="Name of the project [3-15 chars, letters and underscores only]")
    parser.add_argument("-n",
                        "--project_name",
                        type=str,
                        required=True,
                        help="Name of the project sonar key  [3-15 chars, letters and underscores only]")  # NOQA: E501# NOQA: E501
    parser.add_argument("-k",
                        "--sonar_key",
                        type=str,
                        required=True,
                        help="Name of the project sonar key  [3-15 chars, letters and underscores only]")  # NOQA: E501
    try:
        args = parser.parse_args()

        project_directory = args.directory
        projectname = args.ProjectName
        project_name = args.project_name
        sonar_key = args.sonar_key

        helper = Helper(project_directory, project_name, projectname, sonar_key)
        helper.validate_args()
        helper.clean_dir()

        replace_project_name(project_directory, projectname, "ProjectName")  # NOQA: E501
        replace_project_name(project_directory, project_name, "project_name")  # NOQA: E501
        replace_project_name(project_directory, sonar_key, "SONAR_KEY")

        helper.rename_files()
        helper.rename_dir()
        helper.delete_dir()
    except Exception as e:
        print(e)

    return 0


if '__main__' == __name__:
    sys.exit(main(sys.argv))
