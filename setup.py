# Copyright (c) 2015. Mount Sinai School of Medicine
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function
import os

from setuptools import setup, find_packages

readme_dir = os.path.dirname(__file__)
readme_filename = os.path.join(readme_dir, 'README.md')

try:
    with open(readme_filename, 'r') as f:
        readme = f.read()
except:
    readme = ""

try:
    import pypandoc
    readme = pypandoc.convert(readme, to='rst', format='md')
except:
    print(
        "Conversion of long_description from MD to reStructuredText failed...")


if __name__ == '__main__':
    setup(
        name='kerseq',
        version="0.0.1",
        description="Helpers for sequence prediction using Keras",
        author="Alex Rubinsteyn",
        author_email="alex {dot} rubinsteyn {at} gmail {dot} com",
        url="https://github.com/hammerlab/kerseq",
        license="http://www.apache.org/licenses/LICENSE-2.0.html",
        classifiers=[
            'Development Status :: 3 - Alpha',
            'Environment :: Console',
            'Operating System :: OS Independent',
            'License :: OSI Approved :: Apache Software License',
            'Programming Language :: Python',
        ],
        install_requires=[
            'numpy >=1.7, <2.0',
            'pandas >=0.13.1',
            'keras',
            'nose >=1.3.6, <1.4'
        ],
        long_description=readme,
        packages=find_packages(exclude="test"),
    )
