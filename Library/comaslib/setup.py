from setuptools import find_packages, setup

with open('requirements.txt') as f:
    requirements = f.read().splitlines()
setup(
    name='comaslib',
    packages=find_packages(include=['comaslib'], exclude=['build', 'build.*', 'dist', 'dist.*']),
    install_requires=requirements,
    version='0.0.0',
    description='GNN paper library',
    author='Alex Comas',
    license='MIT',
)