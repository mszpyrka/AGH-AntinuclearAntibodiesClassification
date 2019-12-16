from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='antinuclear-antibodies-classification',
    version='0.0.4',
    description='Library that detects the type of antibodies present in serum in the diagnosis of immunological '
                'diseases',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Jan Gołda, Mateusz Szpyrka',
    author_email='jan.golda@gmail.com',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Intended Audience :: Healthcare Industry',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering :: Image Recognition',
        'Topic :: Scientific/Engineering :: Medical Science Apps.'
    ],
    keywords='antinuclear antibodies classification ana',
    python_requires='>=3.5',
    packages=['ana_classification'],
    install_requires=[
        'click>=7.0',
        'numpy>=1.17',
        'opencv-python>=4.1.2.30',
        'scikit-image>=0.16',
        'scipy>=1.3',
        'tensorflow>=2.0.0',
        'scikit-learn>=0.22'
    ],
    extras_require={
        'dev': ['matplotlib>=3.1'],
    },
    package_data={
        'ana_classification': ['resources/*'],
    },
    entry_points={
        'console_scripts': [
            'ana-classification = ana_classification.cli:cli'
        ]
    },
    url='https://github.com/meszszi/AGH-AntinuclearAntibodiesClassification',
    project_urls={
        'Source': 'https://github.com/meszszi/AGH-AntinuclearAntibodiesClassification',
    }
)
