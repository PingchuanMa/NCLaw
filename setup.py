from setuptools import setup, find_packages

setup(
    name='nclaw',
    version='1.0',
    packages=find_packages(),
    zip_safe=False,
    install_requires=[
        'numpy',
        'scipy',
        'hydra-core',
        'einops',
        'trimesh',
        'tqdm',
        'psutil',
        'pyvista',
        'tensorboard',
    ],
)
