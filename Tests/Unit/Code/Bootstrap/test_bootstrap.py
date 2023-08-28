import unittest

from Code.Bootstrap.bootstrap import Helper


class HelperTestCase(unittest.TestCase):
    def setUp(self):
        self.helper = Helper("project_directory", "project_name", "ProjectName", "sonar_key")

    def test_project_directory(self):
        self.assertEqual(self.helper.project_directory, "project_directory")

    def test_project_name(self):
        self.assertEqual(self.helper.project_name, "project_name")

    def test_git_repo(self):
        self.assertEqual(self.helper.git_repo, "https://github.com/lgcorzo/MLOPS_Repo_structure_template.git")

    def test_sonar_key(self):
        self.assertEqual(self.helper.sonar_key, "sonar_key")

    def test_ProjectName(self):
        self.assertEqual(self.helper.ProjectName, "ProjectName")

    def test_rename_files(self):
        # TODO: Write positive and negative test cases for rename_files function
        pass

    def test_rename_dir(self):
        # TODO: Write positive and negative test cases for rename_dir function
        pass

    def test_delete_dir(self):
        # TODO: Write positive and negative test cases for delete_dir function
        pass

    def test_clean_dir(self):
        # TODO: Write positive and negative test cases for clean_dir function
        pass

    def test_validate_args(self):
        # TODO: Write positive and negative test cases for validate_args function
        pass


class ReplaceProjectNameTestCase(unittest.TestCase):
    def test_replace_project_name(self):
        # TODO: Write positive and negative test cases for replace_project_name function
        pass


if __name__ == '__main__':
    unittest.main()
