from setuptools import setup, find_packages

from typing import List


def get_requirements(file_path: str) -> List[str]:
    """
    This function reads a requirements file and returns a list of packages.
    It removes any '-e .' entries and comments.
    """
    with open(file_path) as file:
        lines = file.readlines()

    # Remove any '-e .' entries and comments
    requirements = []
    for line in lines:
        line = line.strip()
        if line and not line.startswith("#") and line != "-e .":
            requirements.append(line)

    return requirements


setup(
    name="data-science-project",
    version="2.0.0",
    author="Reza",
    author_email="Arabporr@yahoo.com",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt"),
    description="The first version of the data science challenge project",
)
