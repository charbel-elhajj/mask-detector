from setuptools import setup, find_packages

setup(
    name='Yolo-v4',
    version='1.0',
    scripts=find_packages(),
    entry_points={
        'console_scripts': [
            'my_start=package.__init__:main',
        ]
    },
    install_requires=[
        'numpy',
        'flask',
        'opencv-python',
        'imutils'
    ],
)
