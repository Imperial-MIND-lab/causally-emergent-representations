from setuptools import setup, find_packages

setup(
    name="info-theory-experiments",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.26.0",
        "scipy>=1.12.0",
        "torch>=2.1.0",
        "matplotlib>=3.8.0",
        "tqdm>=4.66.0",
        "requests>=2.31.0",
        "wandb>=0.16.0",
        "einops>=0.7.0",
    ],
)