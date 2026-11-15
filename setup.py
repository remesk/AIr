from setuptools import setup, find_namespace_packages

with open("README.md", "r") as f:
    long_desc = f.read()

setup(name="AIr",
      version="1.0.0",
      author="Eeli Remes",
      description="Machine Learning development platform",
      long_description=long_desc,
      package_dir={'': 'src'},
      packages=find_namespace_packages(where='src'),
      install_requires=["numpy"]
      )
