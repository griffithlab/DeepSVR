from distutils.core import setup

setup(
    name='manual_review_classifier',
    version='0.0.1',
    description='MR processing',
    url='https://github.com/bainscou/manual_review_classifier',
    author='Ben Ainscough',
    author_email='b.ainscough@wustl.edu',
    license='MIT',
    classifiers=[
        'Development Status :: 1 - Planning'
        'Intended Audience :: Developers',
        'Topic :: Utilities',
        'License :: OSI Approved :: MIT License'
        'Programming Language :: Python :: 3.6'
    ],
    packages=['manual_review_classifier'], requires=['pandas']
)
