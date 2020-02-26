#!/usr/bin/env python

try:
	from setuptools import setup
except ImportError:
	from distutils.core import setup

from setuptools import find_packages
import os

readme = open("README.rst").read()

PACKAGE_PATH = os.path.abspath(os.path.join(__file__, os.pardir))

setup(
    name='avocet',
    version='0.0.1',
    description='Abundance matching package with emulators.',
    long_description=readme,
    author='Sebastian Wagner-Carena',
    author_email='sebaswagner@outlook.com',
    url='https://github.com/swagnercarena/avocet',
    packages=find_packages(PACKAGE_PATH),
    package_dir={'avocet': 'avocet'},
    include_package_data=True,
    license='MIT',
    zip_safe=False
)