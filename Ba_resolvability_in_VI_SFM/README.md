在VIO系统初始化过程中，acc bias是最难观测出来的一个量。

本文通过手写实现一个虚拟的VI-SFM系统来研究acc bias的激励条件。

研究表明，由于噪声的影响，当IMU的激励不充分时，acc bias很难估计出来。

读者可以通过修改"Control Parameters"来进行相关研究，相信对于理解VI-SFM的工作原理会很有帮助。

本算法仅借助Eigen库手写实现。

结论：必须同时有大的加速度激励(e.g. (-1, 1) m/s^2)和大的各轴角度激励(e.g. (-50, 90) degree)才能求解
