from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="mastpy",
    version="0.1.0",
    description="Model-based Analysis of Single Cell Transcriptomics in Python",
    author="MAST-py Team",
    author_email="mastpy@example.com",
    url="https://github.com/yourusername/mastpy",
    packages=find_packages(),
    install_requires=requirements,
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    keywords="single-cell transcriptomics bioinformatics MAST",
    include_package_data=True,
)
