# Accelerator-bias-resolvability-in-VIO-system
在VIO或组合导航定位系统初始化过程中，acc bias是最难观测出来的一个量。

这里通过手写实现一个VI-SFM系统来研究 acc bias 的激励条件。

同时，也手写实现了一个基于ESKF的算法来研究 acc bias 的激励条件。

研究表明，由于噪声的影响，当IMU的激励不充分时，acc bias很难估计出来。

读者可以通过修改"Control Parameters"来进行相关研究，相信对于理解VIO初始化条件的工作原理会很有帮助。

本算法仅借助Eigen库手写实现。

结论：必须同时有大的加速度激励(e.g. (-1, 1) m/s^2)和大的各轴角度激励(e.g. (-50, 90) degree)才能使得 acc bias可观；

ESKF算法由于将 acc bias 相关信息保留在了P阵中，因此可在激励较大的运动过程中实现收敛。
