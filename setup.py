from setuptools import setup
from setuptools import find_packages

setup(name='gembed',
      version='0.0.1',
      description='Produces embedding vectors for each node in a graph',
      download_url='https://github.com/bhaney/graph_embeddings',
      license='MIT',
      install_requires=['numpy',
                        'tensorflow',
                        'keras',
                        'rdflib',
                        'future',
                        'scipy',
                        'pandas',
                        'rgcn',
                        'unicodecsv',
                        'wget',
                        ],
      package_data={'gembed': ['README.md', 'gembed/data']},
      packages=find_packages())
