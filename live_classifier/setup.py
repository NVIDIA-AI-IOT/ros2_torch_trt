from setuptools import setup

package_name = 'live_classifier'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='rishabh',
    maintainer_email='rchadha@nvidia.com',
    description='A ROS2 node for live classification using PyTorch',
    license='MIT License',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        'live_classfier = live_classifier.live_classifier:main',
        ],
    },
)
