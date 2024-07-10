"""Installation script for the 'neugraspnet' python package."""

from setuptools import setup, find_packages

# Installation operation
setup(
    name="neugraspnet",
    author="Snehal Jauhri",
    author_email='snehal@robot-learning.de',
    version="2024.7.1",
    description="https://sites.google.com/view/neugraspnet",
    keywords=["grasping", "robotics", "neural fields"],
    install_requires=[],
    packages=find_packages(),
    zip_safe=False,
)

# EOF