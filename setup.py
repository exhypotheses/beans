import setuptools


NAME = 'beans'
VERSION = '0.0.1'
DESCRIPTION = 'Beans ...'
AUTHOR = 'greyhypotheses'
URL = 'https://github.com/exhypotheses/beans'
PYTHON_REQUIRES = '=3.7.7'


with open('README.md') as f:
    readme_text = f.read()


setuptools.setup()(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=readme_text,
    author=AUTHOR,
    url=URL,
    python_requires=PYTHON_REQUIRES,
    license='',
    packages=setuptools.find_packages(exclude=['docs', 'tests'])
)