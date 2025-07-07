from setuptools import setup, find_packages

setup(
    name="cascade2p",
    version="1.0",
    packages=find_packages(),           # finds your cascade2p package dirs
    install_requires=[

        "tensorflow>=2.3.*",
        "numpy>=1.21",
       "ruamel.yaml>=0.16.5",  # for YAML config parsing
        # any other deps cascade2p needs
    ],
    description="Legacy Cascade2p wrapper for spike inference",
)
