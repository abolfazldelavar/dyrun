from setuptools import setup, find_packages

setup(
    name = 'dyrun',
    version = '1.0.0',
    author = 'Abolfazl Delavar',
    author_email = 'faryadell@gmail.com',
    description = 'DYRUN is a simple tool that facilitates dynamic simulations and illustration expeditiously and effortlessly.',
    long_description = open('README.md').read(),
    long_description_content_type = 'text/markdown',
    url = 'https://github.com/abolfazldelavar/dyrun',
    packages = find_packages(),
    license = open('LICENSE.md').read()
)