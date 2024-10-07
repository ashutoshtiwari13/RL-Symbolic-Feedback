import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="rlsf-sentiment-analysis",
    version="0.1.0",
    author="Ashutosh Tiwari",
    author_email="ashutosh.tiwari@berkeley.edu",
    description="Fine-tuning of sentiment analysis model using Reinforcement Learning and Symbolic AI Feedback. Enhancing resoning capabilties of LLMs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ashutoshtiwari13/RL-Symbolic-Feedback.git",
    packages=setuptools.find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    python_requires=">=3.9",
    install_requires=[
        "torch>=1.9.0",
        "transformers>=4.5.0",
        "datasets>=1.6.0",
        "tqdm>=4.62.0",
        "numpy>=1.19.5",
        "wandb>=0.12.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.2.4",
            "black>=21.6b0",
            "isort>=5.9.2",
            "flake8>=3.9.2",
        ],
    },
    entry_points={
        "console_scripts": [
            "train-rlsf=scripts.train:main",
        ],
    },
)