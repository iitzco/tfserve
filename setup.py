#!/usr/bin/env python

import sys
from setuptools import setup

if sys.version_info < (3, 5):
    raise NotImplementedError("Sorry, you need at least Python 3.5 to use tfserve.")

def readme():
    with open('README.md') as f:
        return f.read()

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(name='tfserve',
      version="0.1.0",
      description='Serve TF models simple and easy as an HTTP API server.',
      long_description=readme(),
      long_description_content_type="text/markdown",
      author="Ivan Itzcovich",
      author_email='i.itzcovich@gmail.com',
      url='http://github.com/iitzco/tfserve',
      keywords='tensorflow deep-learning serving',
      # scripts=["bin/tfserve"],
      packages=['tfserve'],
      license='MIT',
      platforms='any',
      install_requires=required,
      python_requires='>3.5',
      classifiers=['License :: OSI Approved :: MIT License',
                   'Topic :: Internet :: WWW/HTTP :: HTTP Servers',
                   'Topic :: Internet :: WWW/HTTP :: WSGI',
                   'Topic :: Internet :: WWW/HTTP :: WSGI :: Application',
                   'Topic :: Internet :: WWW/HTTP :: WSGI :: Middleware',
                   'Topic :: Internet :: WWW/HTTP :: WSGI :: Server',
                   'Topic :: Software Development :: Libraries :: Application Frameworks',
                   'Programming Language :: Python :: 3.5',
                   'Programming Language :: Python :: 3.6',
                   'Programming Language :: Python :: 3.7',
                   ],
      test_suite='nose.collector',
      tests_require=['nose'],
      include_package_data=True,
      )
