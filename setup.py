from setuptools import setup


packname = 'swarm'
version = '0.1'


setup(
    name=packname,
    version=version,
    license='GPLv3',
    python_requires='!=3.0.*, !=3.1.*, !=3.2.*, <4',
    install_requires=[],
    packages=[packname],
    package_dir={'swarm': 'src'},
)
