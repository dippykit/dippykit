"""Setuptools based setup module for dippykit.

"""

from setuptools import setup, find_packages
import os

cur_path = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(cur_path, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='dippykit',
    version='1.0.0',
    author='Brighton Ancelin',
    author_email='bancelin3@gatech.edu',
    description='A Python Package for Digital Image Processing Education',
    long_description=long_description,
    long_description_content_type='txt/x-rst',
    url="https://github.com/dippykit/dippykit",
    classifiers=[
        'Development Status :: Alpha',
        'Intended Audience :: Students',
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: GNU GPLv3',
        'Versioning Scheme :: Semantic Versioning 2.0.0',
    ],
    packages=find_packages(exclude=['docs', 'doc']),
    install_requires=[
        'matplotlib',
        'numpy',
        'opencv-python',
        'Pillow',
        'scipy',
        'scikit-image',
    ],
    python_requires='>=3',
)
