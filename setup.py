from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

d = generate_distutils_setup(
    packages=['lane_detection_lib'],
    package_dir={'': 'src'}
)
setup(**d)

### RUN python3 setup.py install