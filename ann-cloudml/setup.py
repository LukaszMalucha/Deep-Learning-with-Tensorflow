from setuptools import setup, find_packages

REQUIRED_PACKAGES = ['pandas','keras','scikit-learn','matplotlib','numpy']

## Setup Params for GCloud ML Engine - keras has to be mentioned, otherwise it'll say 'can't find keras'

setup(name='churn-modelling',
      version='0.1',
      packages=find_packages(),
      include_package_data=True,
      description='Churn Modelling with keras and tensorflow',
      author='Lukasz Malucha',
      author_email='lucasmalucha@gmail.com',
      license='MIT',
      install_requires=[
                      'keras',
                      'h5py',
                      ],
                      zip_safe=False) 