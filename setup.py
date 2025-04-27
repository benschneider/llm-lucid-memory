from setuptools import setup, find_packages

setup(
    name="llm-lucid-memory",
    version="0.1.0",
    description="Lucid Memory - Modular reasoning brain for small LLMs",
    author="Ben Schneider",
    author_email="benh.schneider@gmail.com",
    url="https://github.com/benschneider/llm-lucid-memory",
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
    ],
    python_requires='>=3.8',
)
