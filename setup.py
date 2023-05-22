from setuptools import setup, find_packages

setup(
    name = 'dyrun',
    version = '1.0.23',
    author = 'Abolfazl Delavar',
    author_email = 'faryadell@gmail.com',
    description = 'Easy way to simulation',
    long_description = 'DYRUN is a simple tool that facilitates dynamic simulations and illustration expeditiously and effortlessly.',
    long_description_content_type = 'text/markdown',
    url = 'https://github.com/abolfazldelavar/dyrun',
    packages = find_packages(),
    license = open('LICENSE.md').read(),
    install_requires = ['numpy', 'pandas', 'matplotlib', 'control', 'pyqt5', 'opencv-python', 'click'],
    keywords=['Control', 'Simulation', 'Academic', 'Mathematics', 'Differential', 'Closed-loop', 'Numberical calculation'],
    classifiers = [
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires = '>=3.6',
    include_package_data = True,
    entry_points = {
        'console_scripts': [
            'dyrun=dyrun.cli:build_cmd'
        ]
    }
)