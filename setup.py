from setuptools import setup, find_packages

long_description = '''
Todo
'''


setup(
    name='item_generation',
    packages=['item_generation'],  # this must be the same as the name above
    version='0.0.1',
    description="Todo",
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Veronika Kotova',
    author_email='veronika.kotova@tum.de',
    url='https://github.com/vnvkotova/item_generation',
    keywords=['deep learning', 'transformers', 'text generation'],
    classifiers=[],
    license='Todo',
    entry_points={
        'console_scripts': ['item_generation=item_generation.item_generation:cmd'],
    },
    python_requires='>=3.5',
    include_package_data=True,
    install_requires=['transformers', 'torch', 'tqdm', 'numpy',]
)
