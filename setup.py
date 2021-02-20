#!/usr/bin/env python

import os
import setuptools
import torchist

with open('README.md', 'r') as f:
    readme = f.read()

with open('requirements.txt', 'r') as f:
    required = f.read().splitlines()

setuptools.setup(
    name='torchist',
    version=torchist.__version__,
    description=torchist.__doc__,
    long_description=readme,
    long_description_content_type='text/markdown',
    keywords='pytorch histogram',
    author='François Rozet',
    author_email='francois.rozet@outlook.com',
    url='https://github.com/francois-rozet/torchist',
    install_requires=required,
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
