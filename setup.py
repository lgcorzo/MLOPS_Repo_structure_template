"""
This file configures the Python package with entrypoints used for future runs on Databricks.

Please follow the `entry_points` documentation for more details on how to configure the entrypoint:
* https://setuptools.pypa.io/en/latest/userguide/entry_point.html
"""

from setuptools import find_packages, setup
__version__ = '0.0.1'

PACKAGE_REQUIREMENTS = ["pyyaml",
                        "pandas",
                        "tqdm",
                        "scikit-learn",
                        "python-dotenv"
                        ]

# packages for local development and unit testing
# please note that these packages are already available in DBR, there is no need to install them on DBR.
LOCAL_REQUIREMENTS = [
    "ipykernel"
]

TEST_REQUIREMENTS = [
    # development & testing tools
    "pytest",
    "coverage",
    "pytest-cov"
]

setup(
    name="Code",
    packages=find_packages(exclude=["tests", "tests.*"]),
    setup_requires=["setuptools", "wheel"],
    install_requires=PACKAGE_REQUIREMENTS,
    extras_require={"local": LOCAL_REQUIREMENTS, "test": TEST_REQUIREMENTS},
    version=__version__,
    description="project template MLOPS lantek360",
    author="Luis Galo Corzo",
)
