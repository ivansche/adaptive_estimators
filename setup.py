from setuptools import setup


setup(
    author="Ivan Chelombiev",
    description="Code for Adaptive estimators paper.",
    install_requires=[
        "tensorflow~=1.11.0",
        "numpy~=1.15.0",
        "joblib~=0.12.5",
        "keras~=2.2.4",
        "mpmath~=1.0.0",
        "matplotlib~=3.0"
    ],
    license="MIT LICENSE",
    name="adaptive_estimators"
)
