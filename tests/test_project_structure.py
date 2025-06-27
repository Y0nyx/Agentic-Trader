"""
Unit tests to verify the existence of essential project structure components.

This test module verifies that all required directories and files are present
as specified in the project initialization requirements.
"""

import unittest
import os
import sys
from pathlib import Path

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent


class TestProjectStructure(unittest.TestCase):
    """Test class to verify the complete project structure exists."""

    def setUp(self):
        """Set up test fixtures."""
        self.project_root = PROJECT_ROOT
        self.required_directories = [
            'data',
            'strategies', 
            'simulation',
            'evaluation',
            'optimization',
            'tests'
        ]
        self.required_files = [
            'requirements.txt',
            'setup.py',
            'README.md'
        ]

    def test_project_root_exists(self):
        """Test that the project root directory exists."""
        self.assertTrue(self.project_root.exists(), 
                       f"Project root directory does not exist: {self.project_root}")

    def test_required_directories_exist(self):
        """Test that all required directories exist."""
        for directory in self.required_directories:
            dir_path = self.project_root / directory
            with self.subTest(directory=directory):
                self.assertTrue(dir_path.exists(), 
                               f"Required directory does not exist: {directory}")
                self.assertTrue(dir_path.is_dir(), 
                               f"Path exists but is not a directory: {directory}")

    def test_init_files_exist(self):
        """Test that __init__.py files exist in all required directories."""
        for directory in self.required_directories:
            init_file = self.project_root / directory / '__init__.py'
            with self.subTest(directory=directory):
                self.assertTrue(init_file.exists(), 
                               f"__init__.py does not exist in: {directory}")
                self.assertTrue(init_file.is_file(), 
                               f"__init__.py exists but is not a file in: {directory}")

    def test_required_files_exist(self):
        """Test that all required project files exist."""
        for file_name in self.required_files:
            file_path = self.project_root / file_name
            with self.subTest(file=file_name):
                self.assertTrue(file_path.exists(), 
                               f"Required file does not exist: {file_name}")
                self.assertTrue(file_path.is_file(), 
                               f"Path exists but is not a file: {file_name}")

    def test_requirements_contains_yfinance(self):
        """Test that requirements.txt contains yfinance dependency."""
        requirements_file = self.project_root / 'requirements.txt'
        self.assertTrue(requirements_file.exists(), 
                       "requirements.txt does not exist")
        
        with open(requirements_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        self.assertIn('yfinance', content.lower(), 
                     "yfinance dependency not found in requirements.txt")

    def test_setup_py_is_valid_python(self):
        """Test that setup.py is valid Python code."""
        setup_file = self.project_root / 'setup.py'
        self.assertTrue(setup_file.exists(), "setup.py does not exist")
        
        # Try to compile the setup.py file
        with open(setup_file, 'r', encoding='utf-8') as f:
            setup_content = f.read()
        
        try:
            compile(setup_content, str(setup_file), 'exec')
        except SyntaxError as e:
            self.fail(f"setup.py contains syntax errors: {e}")

    def test_readme_is_not_empty(self):
        """Test that README.md exists and is not empty."""
        readme_file = self.project_root / 'README.md'
        self.assertTrue(readme_file.exists(), "README.md does not exist")
        
        with open(readme_file, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        
        self.assertGreater(len(content), 0, "README.md is empty")
        self.assertIn('Agentic Trader', content, 
                     "README.md does not contain project name")

    def test_project_structure_completeness(self):
        """Test that the complete project structure is present."""
        # Check all directories and their __init__.py files
        missing_components = []
        
        for directory in self.required_directories:
            dir_path = self.project_root / directory
            if not dir_path.exists():
                missing_components.append(f"Directory: {directory}")
            else:
                init_file = dir_path / '__init__.py'
                if not init_file.exists():
                    missing_components.append(f"File: {directory}/__init__.py")
        
        # Check required files
        for file_name in self.required_files:
            file_path = self.project_root / file_name
            if not file_path.exists():
                missing_components.append(f"File: {file_name}")
        
        if missing_components:
            self.fail(f"Missing project components: {', '.join(missing_components)}")


if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)