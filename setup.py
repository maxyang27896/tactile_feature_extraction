import os
from setuptools import setup

lib_folder = os.path.dirname(os.path.realpath(__file__))

# get required packages from requirements.txt
requirement_path = os.path.join(lib_folder, 'requirements.txt')
install_requires = []
if os.path.isfile(requirement_path):
    with open(requirement_path) as f:
        install_requires = f.read().splitlines()

setup(name='tactile_feature_extraction',
      version='0.0.1',
      description='Tools for enabling deep learning on tactile data.',
      author='Max Yang',
      author_email='max.yang@bristol.ac.uk',
      license='',
      packages=['tactile_feature_extraction'],
      install_requires=install_requires,
      zip_safe=False)
