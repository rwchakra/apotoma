import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="apotoma",
    version="0.0.1",
    author="Rwiddhi Chakraborty and Michael Weiss",
    author_email="code@mweiss.ch",
    description="Short description",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rwiddhic96/apotoma",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)