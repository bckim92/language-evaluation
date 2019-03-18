from setuptools import setup, find_packages
from setuptools.command.install import install
from subprocess import check_call


install_requires = [
    # For coco-caption-py3
    'numpy',
    'matplotlib',
    'scikit-image',
    # For rouge
    'absl-py',
    'nltk',
    'six',
    # For pyrouge
    'more_itertools',
    # etc
    'colorlog',
]

tests_requires = [
]

setup(
    name='language_evaluation',
    version='0.1.0',
    license='MIT',
    description='NLG Evaluation Toolkit (Experimental)',
    url='https://github.com/bckim92/language-evaluation',
    author='Byeongchang Kim',
    author_email='byeongchang.kim@gmail.com',
    keywords='nlg evaluation rouge bleu spice meteor cider',
    classifiers=[
        # https://pypi.python.org/pypi?%3Aaction=list_classifiers
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    packages=find_packages(),
    install_requires=install_requires,
    extras_require={'test': tests_requires},
    setup_requires=['pytest-runner'],
    tests_require=tests_requires,
    entry_points={
    },
    include_package_data=True,
    zip_safe=False,
)
