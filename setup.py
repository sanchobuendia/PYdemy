import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="PYdemy", # Replace with your own username
    version="0.0.1",
    author="ModelingTaskForce",
    author_email="author@example.com",
    description="A python package to run compartimental models on epidemic data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ModelingTaskForce/PYdemy",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)