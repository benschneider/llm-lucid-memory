from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="llm-lucid-memory",
    version="0.1.3",
    description="Lucid Memory - Modular reasoning brain for small LLMs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Ben Schneider",
    author_email="benh.schneider@gmail.com",
    url="https://github.com/benschneider/llm-lucid-memory",
    project_urls={
        "Source": "https://github.com/benschneider/llm-lucid-memory",
        "Issue Tracker": "https://github.com/benschneider/llm-lucid-memory/issues",
    },
    packages=find_packages(),
    install_requires=[
        "fastapi",
        "uvicorn",
        "requests",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires='>=3.8',
)
