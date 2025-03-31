from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="fixation_visualizer",
    version="0.1.0",
    author="Matteo Mazzini",
    author_email="matteo.mazzini@estudiantat.upc.edu",
    description="A package for visualizing eye tracking fixation data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mmazzini01/fixationvisualizerpy",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[
        "numpy",
        "pandas",
        "opencv-python",
        "matplotlib",
    ],
) 