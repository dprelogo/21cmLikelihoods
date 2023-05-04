from setuptools import setup

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="21cmLikelihoods",
    version="0.1dev",
    author="David Prelogović",
    author_email="david.prelogovic@gmail.com",
    description="NDE Likelihoods of the 21cm power spectrum",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dprelogo/21cmLikelihoods",
    packages=["py21cmlikelihoods"],
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[
        "tensorflow",
        "tensorflow-probability",
        "numpy",
    ],
)
