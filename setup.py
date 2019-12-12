from setuptools import setup


setup(
    name="AntinuclearAntibodiesClassification",
    author="Jan Gołda, Mateusz Szpyrka",
    version="1.0",
    packages=['ana_classification'],
    install_requires=['click', 'numpy', 'opencv-python', 'matplotlib', 'scikit-image', 'scipy', 'tensorflow', 'scikit-learn', 'pickle'],
    package_data={
        'ana_classification': ['tag-template-20.tif'],
    },
    entry_points={
        'console_scripts': [
            'ana-classification = ana_classification.cli:cli'
        ]
    }
)
