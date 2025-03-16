# setup.py

from setuptools import setup, find_packages

setup(
    name="Bible-AI",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        # You can list dependencies here if not managed solely by requirements.txt.
    ],
    python_requires=">=3.12",  # Ensure the package is installed on Python 3.12+
    classifiers=[
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
)
