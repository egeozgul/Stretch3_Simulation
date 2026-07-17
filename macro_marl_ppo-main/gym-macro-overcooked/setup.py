#!/usr/bin/env python
# encoding: utf-8
from setuptools import setup

setup(
    name='gym_macro_overcooked',
    version='0.0.1',
    description='Overcooked gym environment with macro actions',
    packages=['gym_macro_overcooked'],
    package_dir={},

    install_requires=[
        'cloudpickle==1.3.0',
        'decorator==5.1.1',
        'dill==0.3.8',
        'future==1.0.0',
        'gym==0.17.2',
        'matplotlib==3.7.5',
        'networkx==2.5',
        'numpy==1.24.4',
        'pandas==2.0.3',
        'Pillow>=8.1.1',
        'pygame==2.5.2',
        'pyglet==1.5.0',
        'pyparsing==3.0.4',
        'python-dateutil==2.9.0',
        'pytz==2024.1',
        'scipy==1.10.0',
        'seaborn==0.13.0',
        'six==1.16.0',
        'termcolor==2.4.0',
        'tqdm==4.66.2',
        ],

    license='MIT',
)

