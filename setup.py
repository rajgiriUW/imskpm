from setuptools import setup, find_packages

from os import path

setup(
    name='IMSKPM',
    version='0.0.1',
    description='IM-SKPM Simulations',
    author='Rajiv Giridharagopal',
    author_email='rgiri@uw.edu',
    license='MIT',
    url='https://github.com/rajgiriUW/imskpm/',

    packages=find_packages(exclude=['xop', 'docs', 'data', 'notebooks']),

    install_requires=['numpy>=1.18.1',
                      'scipy>=1.4.1'
                      ]


)
