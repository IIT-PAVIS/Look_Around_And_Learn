import glob
import os
import re
import subprocess
import sys
from setuptools import sandbox
import setuptools


def install_subs():
    base_third_parties_dir = "third_parties"
    for sub_mod in glob.glob(os.path.join(base_third_parties_dir, "*.whl")):
        print(f"Installing {sub_mod}")
        subprocess.call([sys.executable, "-m", "pip", "install", sub_mod])
    subdirs = [
        os.path.join(base_third_parties_dir, o)
        for o in os.listdir(base_third_parties_dir)
        if os.path.isdir(os.path.join(base_third_parties_dir, o))
    ]

    for sub_mod in subdirs:

        submod_setup_path = sub_mod + "/setup.py"
        if os.path.exists(submod_setup_path):
            # Run submodule setup.py file
            print(f"Installing {sub_mod}")
            sandbox.run_setup(os.path.join(os.getcwd(), sub_mod, "setup.py"), ['clean'])


setuptools.setup(
    name="experimenting_env",
    version="0.1",
    author="Gianluca Scarpellini",
    author_email="gianluca.scarpellini@iit.it",
    packages=setuptools.find_packages(exclude=("tests", "scripts")),
    install_requires=[],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GPL License",
        "Operating System :: Linux",
    ],
    python_requires=">=3.7",
)

