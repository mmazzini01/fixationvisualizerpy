from setuptools import setup, find_packages

def read_requirements():
    with open('requirements.txt') as req:
        return [line.strip() for line in req.readlines() if line.strip() and not line.startswith('#')]

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="fixation_visualizer",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=read_requirements(),
    author="Matteo Mazzini",
    author_email="matteo.mazzini@estudiantat.upc.edu",
    description="A package for visualizing eye tracking fixation data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mmazzini01/fixationvisualizerpy",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
) 