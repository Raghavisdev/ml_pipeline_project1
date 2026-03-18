from setuptools import setup, find_packages
from typing import List

def get_requirements(file_path: str) -> List[str]:
    with open(file_path, 'r') as file:
        requirements = file.readlines()
        requirements = [i.replace("\n", "") for i in requirements]

        if "-e ." in requirements:
            requirements.remove("-e .")

    return requirements

setup(
    name="ml_pipeline_project1",
    version="0.1.0",
    author="Raghav",
    description="A machine learning pipeline project",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt")
)