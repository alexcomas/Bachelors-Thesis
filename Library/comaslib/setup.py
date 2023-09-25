from setuptools import find_packages, setup
setup(
    name='comaslib',
    packages=find_packages(include=['comaslib'], exclude=['build', 'build.*', 'dist', 'dist.*']),
    version='0.0.0',
    description='GNN paper library',
    author='Alex Comas',
    license='MIT',
)