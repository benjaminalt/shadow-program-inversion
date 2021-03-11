from setuptools import setup

setup(
    name='shadow-program-inversion',
    version='1.0',
    packages=['shadow_program_inversion', 'shadow_program_inversion.data', 'shadow_program_inversion.model',
              'shadow_program_inversion.utils', 'shadow_program_inversion.utils.ur', 'shadow_program_inversion.common',
              'shadow_program_inversion.priors', 'shadow_program_inversion.objectives',
              'shadow_program_inversion.experiments', 'shadow_program_inversion.experiments.contact',
              'shadow_program_inversion.experiments.contact.dmp',
              'shadow_program_inversion.experiments.contact.urscript'],
    url='https://github.com/benjaminalt/shadow-program-inversion',
    license='MIT',
    author='Benjamin Alt',
    author_email='benjamin.alt@artiminds.com',
    description='Code for ICRA paper "Robot Program Parameter Inference via Differentiable Shadow Program Inversion"'
)
