# /oscar/data/sramacha/projects_dev/ponderosa/PONDEROSA/setup.py
from setuptools import setup, find_packages

setup(
    name="ponderosa",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "polars>=0.20.0",
        "numpy",
        "pandas",
        "scikit-learn",
        "networkx",
        "pyyaml",
        "matplotlib",
        "seaborn"
    ],
    python_requires=">=3.8",
)
