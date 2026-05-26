from setuptools import setup

package_name = "ttt_hidden"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/policy", [f"{package_name}/policy.json"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="ISDN3000E TA",
    maintainer_email="xingxin.he@connect.ust.hk",
    description="Private hidden players for ISDN3000E final project grading",
    license="MIT",
    entry_points={
        "console_scripts": [
            "rl_player_node = ttt_hidden.rl_player_node:main",
        ],
    },
)
