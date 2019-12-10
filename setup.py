from setuptools import setup


setup(
    name="HEp2CellClassification",
    author="Jan Gołda, Mateusz Szpyrka",
    version="1.0",
    packages=['hep2_classification'],
    install_requires=['click', 'numpy', 'opencv-python', 'matplotlib', 'scikit-image', 'scipy', 'tensorflow'],
    package_data={
        'hep2_classification': ['tag-template-20.tif'],
    },
    entry_points={
        'console_scripts': [
            'hep2-classification = hep2_classification.cli:cli'
        ]
    }
)
