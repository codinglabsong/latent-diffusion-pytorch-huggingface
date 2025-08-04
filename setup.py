from setuptools import setup, find_packages

setup(
    name="ldm",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "tqdm",
        "matplotlib",
        "diffusers[torch]",
        "torchvision",
        "gradio",
    ],
)
