#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = ['Click>=7.0', 'pexpect==4.8', 'joblib==1.2.0',
                'hydra-core==1.3.2', 
                'scipy', 'pandas==2.0.1', 'seaborn', 
                'tqdm']

test_requirements = ['pytest>=3', ]

setup(
    author="Louis Tiao",
    author_email='louistiao@gmail.com',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="Aerofoil Design Toolkit",
    entry_points={
        'console_scripts': [
            'aerofoils=aerofoils.cli:main',
        ],
    },
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='aerofoils',
    name='aerofoils',
    packages=find_packages(include=['aerofoils', 'aerofoils.*']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/ltiao/aerofoils',
    version='0.1.1',
    zip_safe=False,
)
