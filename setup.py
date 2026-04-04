from setuptools import setup, find_packages

setup(
    name="audio-eval",
    version="0.1.0",
    description="Audio Quality Verification Benchmark for AI-Generated Video",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "librosa>=0.10.0",
        "soundfile",
        "pydub",
        "numpy",
        "pandas",
        "scipy",
        "openai",
        "requests",
        "tqdm",
        "matplotlib",
        "seaborn",
    ],
    entry_points={
        "console_scripts": [
            "audio-eval=cli:main",
        ],
    },
)
