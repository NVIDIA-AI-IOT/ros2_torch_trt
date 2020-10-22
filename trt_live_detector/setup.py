from setuptools import setup

package_name = 'trt_live_detector'

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
    description='ROS2 package for object detection using TensorRT',
    license='MIT License',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
	'trt_detector = trt_live_detector.trt_detection:main',
        ],
    },
)
