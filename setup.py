from setuptools import setup, find_packages

setup(
    name="langevin",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
    ],
    author="Fivos Perakis",
    author_email="f.perakis@fysik.su.se",
    description="A package for simulating and analyzing Langevin dynamics with switching friction.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/fperakis/langevin-dynamics",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)
