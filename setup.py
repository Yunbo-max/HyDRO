from setuptools import setup, find_packages

setup(
    name='graphslim',
    version='0.1.0',
    packages=find_packages(),
    include_package_data=True,
    package_data={
        'graphslim': ['configs/**/*.json'],
    },
    python_requires='>=3.8',
)
