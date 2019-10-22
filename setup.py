from importlib.machinery import SourceFileLoader
import io
import os.path

from setuptools import find_packages, setup

autoearth = SourceFileLoader(
    "autoearth", "./autoearth/__init__.py",
).load_module()

with io.open(os.path.join(os.path.dirname(__file__), "README.md"), encoding="utf-8") as f:
    long_description = f.read()

include_tests = False
exclude_packages = ("autoearth.tests",) if not include_tests else ()

package_data = {"": ["README.md"]}
if include_tests:
    test_data_dirs = ["./data/*"]
    package_data["autoearth.tests"] = test_data_dirs

setup(
    name="autoearth",
    description="Magics for working with earth models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    version=autoearth.__version__,
    license="Proprietary",
    author="source{d}",
    author_email="production-machine-learning@sourced.tech",
    url="https://github.com/Guillemdb/autoearth",
    download_url="https://github.com/Guillemdb/autoearth",
    packages=find_packages(exclude=exclude_packages),
    keywords=["autoearth"],
    install_requires=["numpy>=1.16.2,<2",
                      "packaging>=19.0",
                      "pandas>=0.23.4,<0.24",
                      "scipy>=1.1.0,<2",
                      "panel",
                      "holoviews",
                      "bokeh",
                      "statsmodels",
                      ],
    package_data=package_data,
    python_requires=">=3.5",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "License :: Propietary license",
        "Operating System :: POSIX",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Topic :: Software Development :: Libraries",
    ],
)
