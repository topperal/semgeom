from setuptools import setup, find_packages

with open("README.md", "r") as f:
    description = f.read()
    
setup(
    name='semgeom',
    version='1.1',
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "scikit-learn",
        "matplotlib",
        "pandas",
    ],
    extras_require={
        "viz": ["plotly", "seaborn"],
        "models": ["sentence-transformers"],
        "spatial": ["alphashape", "shapely"]
    },
    author="Alina Topper",
    long_description=description,
    long_description_content_type="text/markdown",
    license="MIT",
)
