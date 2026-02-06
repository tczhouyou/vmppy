from setuptools import setup, find_packages

setup(
    name='vmppy',
    version='0.0.0',
    packages=find_packages(),
    long_description=open('README.md').read(),
    python_requires='>=3.8',
    install_requires=[
        'numpy',
        'pydantic>=2',
        'torch',
        'omegaconf',
        'rff',
    ],
)
