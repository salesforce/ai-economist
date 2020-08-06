import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setuptools.setup(
    name="ai-economist",
    version="1.1",
    author="Stephan Zheng, Alex Trott, Sunil Srinivasa",
    author_email="stephan.zheng@salesforce.com",
    description="Foundation: An Economics Simulation Framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/salesforce/ai-economist",
    packages=setuptools.find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
