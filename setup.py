import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setuptools.setup(
    name="ai-economist",
    version="1.0.0",
    author="Stephan Zheng",
    author_email="stephan.zheng@salesforce.com",
    description="Economics+RL Simulator",
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
    python_requires=">=3.7",
)
