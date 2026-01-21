from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = fh.read().splitlines()

setup(
    name="football-match-prediction",
    version="1.0.0",
    author="Guilauem Stelniceanu",
    author_email="g.stelniceanu@gmail.com",
    description="Système de prédiction de matchs de football utilisant le Machine Learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/GuillaumeStelniceanu/football-match-prediction-ml.git",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "fmp-predict=src.main:predict_cli",
            "fmp-train=src.main:train_cli",
        ],
    },
)