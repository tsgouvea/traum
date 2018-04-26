from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='traum',
    version='0.0.dev0',
    description='Analyses neural data acquired with Trodes in the context of Bpod behavioral tasks',
    long_description_content_type='text/markdown',
    url='https://bitbucket.org/tsgouvea/traum/',
    author='Thiago S. Gouvea',

    packages=find_packages(),  # Required

    project_urls={  # Optional
        'Source': 'https://bitbucket.org/tsgouvea/traum'
    }
)
