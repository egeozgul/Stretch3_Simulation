#!/usr/bin/env python
# encoding: utf-8
from setuptools import setup, find_packages

setup(
    name='macro_marl',
    version='0.0.2',
    description='Macro-action-based multi-agent reinforcement learning',
    packages=find_packages(where='src'), 
    package_dir={'': 'src'},  

    python_requires='>=3.6',

    scripts=[
        'scripts/value_based_main.py',
        'scripts/pg_based_main.py',
    ],

    install_requires=[
        'wandb>=0.16.6',
        # Sweep tooling (experiments/discovery/*.py)
        'ruamel.yaml>=0.17',
        'filelock>=3.0',
        # Add other required dependencies here
    ],
    license='MIT',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
)

