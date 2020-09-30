#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from setuptools import setup, find_packages

with open('README.md') as f:
    readme = f.read()

setup(
    name='dpr',
    version='0.1.0',
    description='Facebook AI Research Open Domain Q&A Toolkit',
    url='',  # TODO
    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    long_description=readme,
    long_description_content_type='text/markdown',
    packages=find_packages(),
    setup_requires=[
        'setuptools>=18.0',
    ],
    install_requires=[
        'cython',
        'faiss-cpu>=1.6.1',
        'filelock',
        'numpy',
        'regex',
        'torch>=1.2.0',
        'transformers>=2.2.2,<3.1.0',
        'tqdm>=4.27',
        'wget',
        'spacy>=2.1.8',
    ],
)
