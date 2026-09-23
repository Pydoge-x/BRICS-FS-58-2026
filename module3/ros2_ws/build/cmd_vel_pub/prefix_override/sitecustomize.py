import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/mnt/hgfs/BRICS-FS-58-2026/module3/ros2_ws/install/cmd_vel_pub'
