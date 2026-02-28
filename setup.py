from setuptools import setup, find_packages

setup(
    name="mastpy",
    version="0.1.0",
    description="Model-based Analysis of Single-cell Transcriptomics in Python",
    long_description="""MASTpy is a Python implementation of the MAST (Model-based Analysis of Single-cell Transcriptomics) package originally written in R. It provides methods for analyzing single cell assay data using hurdle models, with optimized performance using Numba and multi-threading.""",
    author="MASTpy Team",
    author_email="",
    url="",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "scipy",
        "statsmodels",
        "numba",
        "tqdm"
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Bio-Informatics"
    ],
    license="MIT",
    python_requires=">=3.7"
)