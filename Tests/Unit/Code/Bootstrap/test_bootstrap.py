import unittest
from unittest.mock import patch, call
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

    @patch('os.rename')
    @patch('glob.glob')
    def test_rename_files(self, mock_glob, mock_rename):
        # Set up mock data
        mock_glob.return_value = ["file1", "file2", "file3"]
        mock_rename.side_effect = lambda src, dst: dst

        # Call the method
        self.helper.rename_files()

        # Assert that the expected calls were made
        mock_glob.assert_called_once_with("project_directory/**/*project_name*",
                                          recursive=True)
        mock_rename.assert_has_calls([
            call("file1", "file1"),
            call("file2", "file2"),
            call("file3", "file3")
        ])

    @patch('os.walk')
    @patch('os.mkdir')
    @patch('os.rename')
    def test_rename_dir(self, mock_rename, mock_mkdir, mock_walk):
        # Set up mock data
        mock_walk.return_value = [
            ("src", ["subdir1", "subdir2"], ["file1.py", "file2.py"]),
            ("src/subdir1", [], ["file3.py"]),
            ("src/subdir2", [], ["file4.py"])
        ]

        # Call the method
        self.helper.rename_dir()

        # Assert that the expected calls were made
        mock_walk.assert_called_once_with("project_directory")
        mock_mkdir.assert_has_calls([call('src'),
                                     call('src'),
                                     call('src/subdir1'),
                                     call('src/subdir2')])
        mock_rename.assert_has_calls([call('src/file1.py', 'src/file1.py'),
                                      call('src/file2.py', 'src/file2.py'),
                                      call('src/subdir1/file3.py', 'src/subdir1/file3.py'),
                                      call('src/subdir2/file4.py', 'src/subdir2/file4.py')])

    @patch('os.system')
    def test_delete_dir(self, mock_system):
        # Set up mock data
        expected_calls = [
            call('rmdir /S /Q "project_directory/Data/Archive"'),
            call('rmdir /S /Q "project_directory/Data/Core"'),
            call('rmdir /S /Q "project_directory/Data/Results"'),
            call('rmdir /S /Q "project_directory/Notebooks/Commissioning"'),
            call('rmdir /S /Q "project_directory/Notebooks/DataIngestion"'),
            call('rmdir /S /Q "project_directory/Notebooks/DataUnderstanding"'),
            call('rmdir /S /Q "project_directory/Notebooks/Deployment"'),
            call('rmdir /S /Q "project_directory/Notebooks/Experimenting"'),
            call('rmdir /S /Q "project_directory/Notebooks/Modelling"'),
            call('rmdir /S /Q "project_directory/Notebooks/Monitoring"'),
            call('rmdir /S /Q "project_directory/Notebooks/Testing"'),
            call('rmdir /S /Q "project_directory/Code/Bootstrap"')
        ]

        # Call the method
        self.helper.delete_dir()

        # Assert the number of calls
        assert mock_system.call_count == len(expected_calls)

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
