# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


setup(
    name='bag',
    version='2.0',
    description='Berkeley Analog Generator',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Programming Language :: Python :: 2.7',
    ],
    author='Eric Chang',
    author_email='pkerichang@berkeley.edu',
    packages=find_packages(),
    install_requires=[
        'setuptools>=18.5',
        'PyYAML>=3.11',
        'Jinja2>=2.8',
        'numpy>=1.10',
        'networkx>=1.11',
        'pexpect>=4.0',
        'pyzmq>=15.2.0',
        'scipy>=0.17',
        'matplotlib>=1.5',
        'h5py',
        'pytest',
    ],
    package_data={
        'bag': ['virtuoso_files/*'],
        'bag.interface': ['templates/*'],
        'bag.verification': ['templates/*'],
    },
)
