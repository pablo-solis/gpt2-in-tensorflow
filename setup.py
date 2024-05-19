from setuptools import setup

setup(
  name='gpt2_in_tf',
  version='1.0',
  description='An educational module',
  author='Pablo',
  author_email='pablos.inbox@gmail.com',
  packages=['gpt2_in_tf'],  #same as name
  install_requires=['tensorflow', 'einops'], #external packages as dependencies
)
