from setuptools import setup

def readme():
    with open('README.md') as f:
        return f.read()

setup(
    name='sympyplus',
    version='0.0.1',
    description='Extension of SymPy to handle expressions from PDEs',
    long_description=readme(),
    long_description_content_type='text/markdown',
    classifiers=[
        'Development Status :: 1 - Planning',
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
    url='',
    author='Andrew Hicks',
    author_email='andrew.hicksm@gmail.com',
    keywords='',
    license='',
    packages=[],
    install_requires=['sympy'],
    include_package_data=True,
)