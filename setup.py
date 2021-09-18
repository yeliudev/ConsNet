# -----------------------------------------------------
# ConsNet
# Licensed under the GNU General Public License v3.0
# Written by Ye Liu (ye-liu at whu.edu.cn)
# -----------------------------------------------------

import os
import re

from setuptools import find_packages, setup


def get_version():
    version_file = os.path.join('consnet', '__init__.py')
    with open(version_file, encoding='utf-8') as f:
        lines = f.readlines()
    for line in lines:
        if line.startswith('__version__'):
            exec(line.strip())
    return locals()['__version__']


def get_readme():
    with open('README.md', encoding='utf-8') as f:
        content = re.sub(r'## Model([\s\S]*)respectively.\n\n', '', f.read())
    return content


setup(
    name='consnet',
    version=get_version(),
    author='Ye Liu',
    author_email='yeliudev@outlook.com',
    license='GPLv3',
    url='https://github.com/yeliudev/ConsNet',
    description='A general toolkit for human-object interaction detection',
    long_description=get_readme(),
    long_description_content_type='text/markdown',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Utilities',
    ],
    python_requires='>=3.8',
    install_requires=['nncore==0.2.4', 'scipy>=1.6', 'torch>=1.6'],
    extras_require={
        'full': [
            'allennlp>=2.2,<2.3', 'mmcv-full>=1.3,<1.4', 'mmdet>=2.11,<2.12',
            'torchvision>=0.7'
        ]
    },
    packages=find_packages(exclude=('.github', 'configs', 'docs', 'tools')))
