import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name='protein_holography',
    version='0.0',
    author='Michael Pun',
    author_email='mpun@uw.edu',
    description='learning protein neighborhoods by incorporating rotational symmetry',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/mpun/protein_holography',
    python_requires='>=3.8',
    install_requires='',
    packages=setuptools.find_packages(),
)
