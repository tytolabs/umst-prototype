from setuptools import find_packages, setup
import os
from glob import glob

package_name = "umst_ros2_bridge"

setup(
    name=package_name,
    version="1.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.py")),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
        (os.path.join("share", package_name, "msg"), glob("msg/*.msg")),
        (os.path.join("share", package_name, "srv"), glob("srv/*.srv")),
    ],
    install_requires=["setuptools", "requests", "websocket-client"],
    zip_safe=True,
    author="Santhosh Shyamsundar, Santosh Prabhu Shenbagamoorthy",
    author_email="santhosh@tyto.studio",
    description="ROS2 bridge for UMST thermodynamic gate — physics-gated robotic fabrication",
    license="MIT",
    entry_points={
        "console_scripts": [
            "gate_bridge = umst_ros2_bridge.gate_bridge_node:main",
            "telemetry_bridge = umst_ros2_bridge.telemetry_bridge_node:main",
            "robot_controller = umst_ros2_bridge.robot_controller_node:main",
        ],
    },
)
