from setuptools import find_packages, setup

setup(
    name="ecommercerecommendation",
    version="0.1.0",
    author="Gergely Farkas",
    packages=find_packages(),
    package_data={"ecommercerecommendation.data": ["sql/*.sql"]},
)
