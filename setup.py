from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="your_library_name",
    version="0.1.0",
    author="Your Name",
    author_email="your_email@example.com",
    description="A short description of your library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your_username/your_library_name",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)