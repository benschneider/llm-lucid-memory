from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="llm-lucid-memory",
    version="0.2.3",
    description="Lucid Memory - Modular reasoning graph for LLMs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Ben Schneider",
    author_email="benh.schneider@gmail.com",
    url="https://github.com/benschneider/llm-lucid-memory",
    project_urls={
        "Source": "https://github.com/benschneider/llm-lucid-memory",
        "Issue Tracker": "https://github.com/benschneider/llm-lucid-memory/issues",
    },
    packages=find_packages(include=['lucid_memory', 'lucid_memory.*']),
    package_data={
         'lucid_memory': ['prompts.yaml', 'proxy_config.example.json'],
    },
    include_package_data=True,
    install_requires=[
        "fastapi>=0.90.0",      # FastAPI for proxy
        "uvicorn",              # Uvicorn for running proxy
        "requests>=2.25.0",     # For LLM calls
        "PyYAML>=5.0",          # For prompts.yaml
        "streamlit>=1.20.0"     # For the new UI
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires='>=3.8',
    entry_points={
        'console_scripts': [
            'lucid-memory=lucid_memory.cli:run_streamlit',
        ],
    },
)