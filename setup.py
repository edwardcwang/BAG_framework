# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


setup(
    name='bag',
    version='3.0',
    license='BSD 3-Clause License',
    description='Berkeley Analog Generator',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: BSD License',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3.7',
    ],
    author='Eric Chang',
    author_email='pkerichang@berkeley.edu',
    python_requires='>=3.7',
    install_requires=[
        'setuptools>=18.5',
        'PyYAML>=3.11',
        'Jinja2>=2.9',
        'numpy>=1.10',
        'networkx>=1.11',
        'pexpect>=4.0',
        'pyzmq>=15.2.0',
        'scipy>=0.17',
        'matplotlib>=1.5',
        'h5py',
    ],
    extras_require={
        'mdao': ['openmdao']
    },
    tests_require=[
        'openmdao',
        'pytest',
    ],
    packages=find_packages('src'),
    package_dir={'': 'src'},
    package_data={
        'bag.interface': ['templates/*'],
        'bag.verification': ['templates/*'],
    },
)
