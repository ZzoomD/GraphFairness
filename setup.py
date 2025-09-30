import os.path as osp

from setuptools import find_packages, setup

def get_version():
    # From: https://github.com/facebookresearch/iopath/blob/master/setup.py
    # Author: Facebook Research
    init_py_path = osp.join(osp.abspath(osp.dirname(__file__)), "graphfairness",
                            "version.py")
    init_py = open(init_py_path, "r").readlines()
    version_line = [
        line.strip() for line in init_py if line.startswith("__version__")
    ][0]
    version = version_line.split("=")[-1].strip().strip("'\"")

    return version


VERSION = get_version()
url = 'https://github.com/ZzoomD/GraphFairness'

setup(
    name="graphfairness",
    version="0.1.0",
    author="Yuchang Zhu",
    author_email="zhuych@mail2.sysu.edu.cn",
    description="Fair graph deep learning toolkit",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url=url,
    packages=find_packages(),
    license="MIT LICENSE",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)