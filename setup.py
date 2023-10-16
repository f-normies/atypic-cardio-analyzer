from setuptools import setup, find_packages

setup(
    name="atypic_cardio_analyzer",
    version="0.4.1",
    description='Library created to analyse atypical cardiomyocyte action potential',
    author='Oleg Zakharov, Aleksei Aredov, Ilya Polusmak',
    author_email='os.zakharov.04@gmail.com',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'dask',
        'openpyxl',
        'scipy',
        'scikit-image'
    ],
)
