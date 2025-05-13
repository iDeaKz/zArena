from setuptools import setup, find_packages

setup(
    name='zArena',
    version='0.1.0',
    description='Symbolic RL Environment Framework for Healing-Aware Agents',
    author='zkaedi',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'numpy',
        'torch',
        'matplotlib',
        'pandas'
    ],
    entry_points={
        'console_scripts': []
    },
)
