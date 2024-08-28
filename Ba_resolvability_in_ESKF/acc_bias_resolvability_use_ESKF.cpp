#include <chrono>
#include <iostream>
#include <map>
#include <fstream>
#include <random>
#include <vector>
#include <Eigen/Dense>
#include <variant>
#include <iomanip>

using namespace std;

/******** 手写实现一个ESKF，并据此研究acc bias等变量的可观性激励条件********
* 一、定义真实量与误差状态量、名义状态量的关系方程，其中 Q、V、P均在世界系W下描述
* Gw表示世界系下的重力向量
* Vt = V + δV
* Rt = R * exp(δΘ^)
* Pt = P + δP
* Ba_t = Ba + δBa
* Bg_t = Bg + δBg
* 
* IMU测量模型定义如下：
* a_t = a - Ba_t - n_a = a - Ba - δBa - n_a
* w_t = w - Bg_t - n_g = w - Bg - δBg - n_g
*
* 噪声n将体现在不确定度即P阵中，而不会显含在后续的数学方程推导过程
* ======================================================================
*
*
* 二、求解上述相关误差状态量的运动方程，其中，δV和δQ的运动方程求解是最为复杂的，
*     这里先对δV的运动方程进行求解。这里以'表示转置，即矩阵A的转置为A'，dt表示IMU的采样周期。
*     令 Exp(Θ) = exp(Θ^)
*
* 1、求解d(δV)/dt
*
*    Vt = V + δV，对方程两边同时求导：
*    d(Vt)/dt = d(V)/dt + d(δV)/dt ==>
*    R * Exp(δΘ) * (a - Ba - δBa) + Gw = R * (a - Ba) + Gw + d(δV)/dt
*    d(δV)/dt = R * (I + δΘ^) * (a - Ba - δBa) - R * (a - Ba)
*             = -R * δBa + R * δΘ^ * (a - Ba - δBa)
*             = -R * δBa - R * (a - Ba - δBa) * δΘ
*    忽略上式二阶小量 δBa * δΘ，即
*             = -R * (a - Ba)^ * δΘ - R * δBa
*
*    由上式可知：
*    d(δV)/dt是角度误差状态量δΘ和acc bias误差状态量δBa的线性函数，
*    并且，系数矩阵与名义旋转R、名义测量a强相关，
*    因此，欲将δΘ、δBa这两个状态量分离出来，
*    那么R、a必须有变化才能构建多个线性无关方程组以求解δΘ、δBa
* ========================================================================
*
* 2、求解d(δΘ)/dt
* 
*    首先，基于Gyroscope模型求解R关于时间的导数，根据定义：
*    dR/dt = {R * Exp(w * dt) - R} / dt
*          = {R * (I + w^ * dt) - R} / dt
*          = (R * w^ * dt) / dt
*          = R * w^
*
*    其次，对于误差状态量δΘ的SO3表示Exp(δΘ)，易知δΘ瞬时角度变化量为d(δΘ)/dt
*    根据上述，由于Exp(δΘ) = exp(δΘ^)，类比指数函数e^x导数，易知：
*    d{Exp(δΘ)}/dt = Exp(δΘ) * {d(δΘ)/dt}^
*    
*    OK，有了上述基础，接下来推导 d(δΘ)/dt 的线性形式：
*    
*    Rt = R * exp(δΘ^)，对方程两边同时对dt进行求导，并运用复合函数求导法则，
*    有：
*    dRt/dt = dR/dt * Exp(δΘ) + R * d{Exp(δΘ)}/dt
*    即：
*    R * Exp(δΘ) * (w - Bg - δBg)^ = R * (w - Bg)^ * Exp(δΘ) + R * Exp(δΘ) * {d(δΘ)/dt}^
*
*    方程两边同时乘以R的转置R'，
*    易得：
*    Exp(δΘ) * (w - Bg - δBg)^ = (w - Bg)^ * Exp(δΘ) + Exp(δΘ) * {d(δΘ)/dt}^
*
*    接下来，希望消除两边的Exp(δΘ)项，利用SO3上的伴随性质，
*    有伴随性质：
*    Φ^ * R = R * (R' * Φ)^
*    移项，并使用伴随性质得：
*    Exp(δΘ) * {d(δΘ)/dt}^ = Exp(δΘ) * (w - Bg - δBg)^ - Exp(δΘ) * {Exp(-δΘ) * (w - Bg)}^
*    ===>
*    {d(δΘ)/dt}^ = { (w - Bg - δBg)^ - {(I - δΘ^) * (w - Bg)}^ }
*                = { (w - Bg - δBg)^ - {I * (w - Bg) - δΘ^ * (w - Bg)}^ }
*                = { -δBg^ - {- δΘ^ * (w - Bg)}^ }
*                = { -δBg - (w - Bg)^ * δΘ}^
*    ==>
*    d(δΘ)/dt = -(w - Bg)^ * δΘ - δBg
*  
*    由上式可知：
*    d(δΘ)/dt 是误差状态量 δΘ 和 δBg 的线性函数，
*    欲分离 δΘ 和 δBg，需要 w 具有变化以构建线性无关方程来分离 δΘ 和 δBg
*    即物体需要做旋转运动
* ============================================================================
*
* 3、求解d(δP)/dt
*    Pt = P + δP
*    dPt/dt = dP/dt + d(δP)/dt
*    Vt + R * Exp(δΘ) * (a - Ba - δBa) * dt + Gw * dt = V + R * (a - Ba) * dt + Gw * dt + d(δP)/dt
*    d(δP)/dt = δV + R * (I + δΘ^) * (a - Ba - δBa) * dt - R * (a - Ba) * dt
*             = δV - R * δBa * dt - R * (a - Ba - δBa)^ * δΘ * dt
*             = δV
*
*    分析可知： δBa * dt、δΘ * dt均是二阶小量可以忽略，
*    因此， d(δP)/dt 是误差状态量 δV的线性函数
* ==============================================================================
*
* 4、求解d(δBa)/dt、d(δBg)/dt
*    根据IMU噪声模型，上述被建模为 bias 噪声模型，
*    即：
*    d(δBa)/dt = n_ba
*    d(δBg)/dt = n_bg
*    由于ESKF使用P阵来表示了不确定度，
*    那么有：
*    d(δBa)/dt = 0
*    d(δBg)/dt = 0
* ===============================================================================
*
* 5、综上，以δP、δQ、δV、δBa、δBg的形式写出第k到k+1时刻的状态转移方程：
*    δP_k+1 = δP_k + δV * dt
*    δΘ_k+1 = Exp{-(w - Bg) * dt} * δΘ_k - δBg * dt  // 这一项比较复杂的原因是：δΘ关于dt的导数与自身δΘ有关，这符合指数函数e^x的导数性质
*    δV_k+1 = δV_k + {-R * (a - Ba)^ * δΘ - R * δBa} * dt
*    δBa_k+1 = δBa_k
*    δBg_k+1 = δBg_k
* ==============================================================================
* 
*
* 三、常见的INS/GNSS组合导航以position、velocity作为量测更新状态量，分别记为Pw, Vw
* 1、关于Pw的量测方程：
*    Pw = Pt = P + δP，
*    那么，量测方程线性化为：
*    d(Pw)/d(δP) = I 
*    构建残差：
*    ΔP = Pw - P = P + Hp * δP - P，其中 Hp = I 为雅可比，即量测方程关于状态量δP的导数，
*    注意： 求残差关于状态量的雅可比矩阵时，一定是要对所有的状态量都进行求导的，即便该例残差只与δP有关，
*    其雅可比矩阵 H 也应该是 [3x15]维的，即 Hq = 0, Hv = 0, Hba = 0, Hbg = 0.
*
* 2、关于Vw的量测方程：
*    Vw = V + δV
*    那么量测方程关于误差状态量的线性化为：
*    d(Vw)/d(δV) = I
*    构建残差：
*    ΔV = Vw - V = V + Hv * δV，其中 Hv = I 为量测方程关于误差状态量δV的导数
* ==================================================================================
*/

// 实现ESKF，其中状态量顺序为： δQ、δP、δV、δBg、δBa，重点关注Ba的收敛过程
// 输入：IMU测量值、Pw量测、Vw量测，量测更新周期为1s，省略重力Gw初始化过程
// 输出：估计状态量值：Q、P、V、Bg、Ba值

/*===========================================================*/
/*================ Control Parameter ========================*/
/*===========================================================*/
constexpr int kImuFrequency = 120; // hz
constexpr double kImuSamplePeriod = 1.0 / 120; // s
constexpr int kKeyFrameFrequency = 10; // hz
const int kImu2ImgRate = kImuFrequency / kKeyFrameFrequency; // 每隔多少个IMU取一个关键帧
constexpr double kRad2Deg = 180 / M_PI;
constexpr double kDeg2Rad = M_PI / 180;
constexpr double kGravityValue = 9.80;
constexpr bool kAddNoise2Acc = true;
constexpr double kAccNoiseStd = 0.1; // noise相比acc真实值过大时，将影响acc bias的可观性
constexpr bool kAddNoise2Gyr = true;
constexpr double kGyrNoiseStd = 0.01 * kDeg2Rad; // gyro noise会极大地影响 acc bias的可观性
const Eigen::Vector3d v0(0.0, 0.0, 0.0); // IMU 需要考虑初始速度, c0系下
const Eigen::Vector3d accBias(0.2, 0.19, 0.05);
const Eigen::Vector3d gyrBias(0.1 * kDeg2Rad, 0.15 * kDeg2Rad, 0.05 * kDeg2Rad);
const Eigen::Vector3d Gw(0, 0, -kGravityValue); // c0 系下的实际重力(含方向)，认为c系与b系重合

// ESKF的初始化及量测噪声参数，调试算法性能的主要参数
constexpr double kStdQuat = 0.5 * kDeg2Rad;
constexpr double kStdPos = 0.1;
constexpr double kStdVel = 0.05;
constexpr double kStdBg = 0.1 * kDeg2Rad;
constexpr double kStdBa = 0.1;
constexpr double kObvPwStd = 0.1;
constexpr double kObvVwStd = 0.05;
const double kMinQPVBaBgVariance[5] = {pow(0.00001 * kDeg2Rad, 2), pow(0.00005, 2), pow(0.00001, 2),
                                            pow(0.00001, 2), pow(0.00005 * kDeg2Rad, 2)};

constexpr double kUpdateRatioEachStep = 1.0; // 每次更新使用的dx比例

// IMU连续噪声数据，从Notavel手册获取
// 位置噪声可由速度、加速度、角速度的噪声参数经过转移矩阵F得到，所以不需要直接提供位置的噪声参数
constexpr double kAngularRandomWalk = 0.1; // deg/hr^0.5，即σ_g
constexpr double kGyrBiasStability = 3.5; // deg/hr，与σ_bg有关
constexpr double kVelocityRandomWalk = 0.5; // m/s/hr^0.5，即σ_a
constexpr double kAccBiasStability = 0.1; // mg, 10^-3m/s^2，与σ_ba有关
// 将连续噪声转为离散噪声数据
const double kNoiseAngStd = kAngularRandomWalk * kDeg2Rad / 60 / sqrt(kImuSamplePeriod) * // rad/s
                                kImuSamplePeriod; // rad，即η_θ，含义为经过IMU的每个采样时间dt，角度的漂移量
const double kNoiseBgStd = kGyrBiasStability * kDeg2Rad / 3600; // rad/s，即η_bg，数值上与零偏不稳定性相等
const double kNoiseVelStd = kVelocityRandomWalk / 60 / sqrt(kImuSamplePeriod) * // m/s^2
                                kImuSamplePeriod; // m/s，即η_v，含义为经过每个采样时间dt，速度的漂移量
const double kNoiseBaStd = kAccBiasStability * 1e-3; // m/s^2，即η_ba，数值上与零偏不稳定性相等
const double kNoiseGyrStd = kNoiseAngStd / kImuSamplePeriod; // rad/s，即η_g，角速度测量值噪声
const double kNoiseAccStd = kNoiseVelStd / kImuSamplePeriod; // m/s^2，即η_a，加速度测量值噪声
const bool kAddImuSensorNoise = true;

/*===========================================================*/
/*=================== Declaration ===========================*/
/*===========================================================*/
struct Estimate
{
    // Q、P、V、Bg、Ba
    Eigen::Quaterniond Q = Eigen::Quaterniond::Identity();
    Eigen::Vector3d P = Eigen::Vector3d::Zero();
    Eigen::Vector3d V = Eigen::Vector3d::Zero();
    Eigen::Vector3d Bg = Eigen::Vector3d::Zero();
    Eigen::Vector3d Ba = Eigen::Vector3d::Zero();
};

struct ErrorState {
    // δQ、δP、δV、δBg、δBa
    Eigen::Matrix<double, 15, 1> dX = Eigen::Matrix<double, 15, 1>::Zero();
    Eigen::Matrix<double, 15, 15> Cov = Eigen::Matrix<double, 15, 15>::Zero();
};

Eigen::Quaterniond DeltaQ(const Eigen::Vector3d& angle);

Eigen::Matrix3d skewSymmetric(const Eigen::Vector3d &v);

void GenerateIMUdata(const int keyframeNum, vector<Eigen::Vector3d> &gyr, vector<Eigen::Vector3d> &acc);

void GenerateObservation(const vector<Eigen::Vector3d> &gyr, const vector<Eigen::Vector3d> &acc, const int keyframeNum,
        vector<Eigen::Vector3d> &Pw, vector<Eigen::Vector3d> &Vw, vector<Eigen::Quaterniond> &Qw);

Eigen::Matrix3d ExpSO3(const Eigen::Vector3d &w);

void Propagate(const Eigen::Vector3d &gyr, const Eigen::Vector3d &acc, ErrorState &errorState, Estimate &estimate);

void UpdatePwObv(const Eigen::Vector3d &Pw, const Estimate &estimate, Eigen::Matrix<double, 6, 15> &H_p_v, 
                  Eigen::Matrix<double, 6, 1> &residual_p_v);

void UpdateVwObv(const Eigen::Vector3d &Vw, const Estimate &estimate, Eigen::Matrix<double, 6, 15> &H_p_v, 
                  Eigen::Matrix<double, 6, 1> &residual_p_v);

void UpdateESKF(const Eigen::Matrix<double, 6, 15> &H_p_v, const Eigen::Matrix<double, 6, 1> &residual_p_v, 
                  const Eigen::Matrix3d &obvPcov, const Eigen::Matrix3d &obvVcov,
                  Estimate &estimate, ErrorState &errorState);

/*========================================================*/
/*=================== Pipeline ===========================*/
/*========================================================*/
int main() {
    cout << "IMU discrete noise metric parameters report:\n"
         << "  angular noise(walk): " << kNoiseAngStd << " rad\n"
         << "  Bg noise: " << kNoiseBgStd << " rad/s\n"
         << "  velocity noise(walk): " << kNoiseVelStd << " m/s\n"
         << "  Ba noise: " << kNoiseBaStd << " m/s^2\n"
         << "  gyr measurement noise: " << kGyrNoiseStd << " rad/s^2\n"
         << "  acc measurement noise: " << kAccNoiseStd << " m/s^2\n\n";

    constexpr int keyframeNum = 130; 
    Estimate estimate;
    ErrorState errorState;

    // 初始化
    Eigen::Matrix<double, 15, 15> &Cov = errorState.Cov;
    // lambda
    auto Square = [](const double v) -> double {return v * v;};
    Cov.block(0, 0, 3, 3) = Eigen::Matrix3d::Identity() * Square(kStdQuat);
    Cov.block(3, 3, 3, 3) = Eigen::Matrix3d::Identity() * Square(kStdPos);
    Cov.block(6, 6, 3, 3) = Eigen::Matrix3d::Identity() * Square(kStdVel);
    Cov.block(9, 9, 3, 3) = Eigen::Matrix3d::Identity() * Square(kStdBg);
    Cov.block(12, 12, 3, 3) = Eigen::Matrix3d::Identity() * Square(kStdBa);


    // 产生传感器数据
    vector<Eigen::Vector3d> gyr, acc;
    GenerateIMUdata(keyframeNum, gyr, acc);

    // 生成量测数据即Pw、Vw，这里假设物体从静止开始运动
    vector<Eigen::Vector3d> Pw, Vw;
    vector<Eigen::Quaterniond> Qw; // 使用待定
    GenerateObservation(gyr, acc, keyframeNum, Pw, Vw, Qw);
    assert(Pw.size() == keyframeNum);

    // 处理传感器数据
    for(int i = 1; i < gyr.size(); ++i) {
        // 预测过程
        const Eigen::Vector3d a = (acc[i] + acc[i-1]) * 0.5;
        const Eigen::Vector3d w = (gyr[i] + gyr[i-1]) * 0.5;
        Propagate(w, a, errorState, estimate);

        const int j = i / kImu2ImgRate; // 商值从1开始
        if(i%kImu2ImgRate == 0) {
            // 量测更新过程
            const Eigen::Vector3d obvP = Pw[j];
            const Eigen::Matrix3d obvPcov = Eigen::Matrix3d::Identity() * pow(kObvPwStd, 2);
            const Eigen::Vector3d obvV = Vw[j];
            const Eigen::Matrix3d obvVcov = Eigen::Matrix3d::Identity() * pow(kObvVwStd, 2);
            Eigen::Matrix<double, 6, 1> residual_p_v = Eigen::Matrix<double, 6, 1>::Zero();
            Eigen::Matrix<double, 6, 15> H_p_v = Eigen::Matrix<double, 6, 15>::Zero();
            // 用Pos量测更新残差及雅可比
            UpdatePwObv(obvP, estimate, H_p_v, residual_p_v);
            // 用Vel量测更新残差及雅可比
            UpdateVwObv(obvV, estimate, H_p_v, residual_p_v);

            // 最后使用标准EKF更新过程更新状态量
            UpdateESKF(H_p_v, residual_p_v, obvPcov, obvVcov, estimate, errorState);


            // Report
            cout << "Pos diff | norm: " << setprecision(3) << (obvP - estimate.P).transpose() << " | " << (obvP - estimate.P).norm() << endl;
            cout << "Ba | std " << setprecision(3) << estimate.Ba.transpose() << " | " 
                                << errorState.Cov.diagonal().middleRows(12, 3).transpose().cwiseSqrt() << endl;
            cout << "Bg | std " << setprecision(3) << estimate.Bg.transpose() * kRad2Deg << " | " 
                                << errorState.Cov.diagonal().middleRows(9, 3).transpose().cwiseSqrt() * kRad2Deg << endl << endl;
        }
    }

    return 0;
}

/*=======================================================*/
/*================= Realization =========================*/
/*=======================================================*/
Eigen::Matrix3d skewSymmetric(const Eigen::Vector3d &v) {
    Eigen::Matrix3d m;
    m.setZero();
    m << 0, -v[2], v[1],
         v[2], 0, -v[0],
         -v[1], v[0], 0;
    return m;
}

void GenerateIMUdata(const int keyframeNum, vector<Eigen::Vector3d> &gyr, vector<Eigen::Vector3d> &acc) {
    // 任意相邻2个IMU读数可以构成一个dt间隔，所需的IMU数量即：(N-1) * imuRate + 1
    const int imuDataNum = (keyframeNum - 1) * kImu2ImgRate + 1;

    std::default_random_engine generator;
    // 均匀分布数值
    std::uniform_real_distribution<double> accReader(-10.1, 10.1);
    std::uniform_real_distribution<double> gyrReader(-10.10 * kDeg2Rad, 30.10 * kDeg2Rad);
    
    Eigen::Quaterniond Q_b0_b = Eigen::Quaterniond::Identity();
    const double dt = kImuSamplePeriod;
    for(int i = 0; i < imuDataNum; ++i) {
        // Eigen::Vector3d a(accReader(generator), accReader(generator), accReader(generator));
        Eigen::Vector3d a(accReader(generator), accReader(generator), 0);
        // 在假设完成初始对准的情况下，所有状态量都被对准到W系，
        // 将重力转到b系下表示
        a -= Q_b0_b.inverse() * Gw;
        acc.push_back(a);

        Eigen::Vector3d w(gyrReader(generator), gyrReader(generator), gyrReader(generator));
        // Eigen::Vector3d w(0, 0, gyrReader(generator)); // 退化运动，bias std无法收敛
        gyr.push_back(w);
        if(gyr.size() > 1) {
            Eigen::Vector3d w_t = (gyr[gyr.size()-2] + w) * 0.5;
            Q_b0_b *= DeltaQ(w_t * dt);
        } 
    }
}

Eigen::Quaterniond DeltaQ(const Eigen::Vector3d& angle) {
        Eigen::Matrix3d Rc1c2 = Eigen::AngleAxisd(angle.norm(), angle.normalized()).toRotationMatrix();
        Eigen::Quaterniond DeltaQ(Rc1c2);
        DeltaQ.normalize();
        return DeltaQ;
};

void GenerateObservation(const vector<Eigen::Vector3d> &gyr, const vector<Eigen::Vector3d> &acc, const int keyframeNum,
        vector<Eigen::Vector3d> &Pw, vector<Eigen::Vector3d> &Vw, vector<Eigen::Quaterniond> &Qw) {
    
    Eigen::Vector3d p = Eigen::Vector3d::Zero();
    Eigen::Vector3d v = Eigen::Vector3d::Zero();
    Eigen::Quaterniond q = Eigen::Quaterniond::Identity();
    const double dt = kImuSamplePeriod;
    const double dt2 = pow(dt, 2);
    
    Pw.push_back(p);
    Vw.push_back(v);
    Qw.push_back(q);
    for(int i = 1; i < gyr.size(); ++i) {
        // 使用中值积分
        // bias扰动IMU读数在预测过程中进行，
        const Eigen::Vector3d w_t = (gyr[i] + gyr[i-1]) * 0.5;
        const Eigen::Vector3d a_t = (acc[i] + acc[i-1]) * 0.5;
        p += v * dt + 0.5 * (q * a_t) * dt2 + 0.5 * Gw * dt2;
        v += q * a_t * dt + Gw * dt;
        q *= DeltaQ(w_t * dt);

        if(i%kImu2ImgRate == 0) {
            Pw.push_back(p);
            Vw.push_back(v);
            Qw.push_back(q);
        }
    }
}

Eigen::Matrix3d ExpSO3(const Eigen::Vector3d &w) {
    auto NormalizeRotation = [](const Eigen::Matrix3d &R) -> Eigen::Matrix3d {
        Eigen::JacobiSVD<Eigen::Matrix3d> svd(R, Eigen::ComputeFullU | Eigen::ComputeFullV);
        return svd.matrixU() * svd.matrixV().transpose();
    };

    const double x = w[0], y = w[1], z = w[2];
    const double d2 = x * x + y * y + z * z;
    const double d = sqrt(d2);
    Eigen::Matrix3d W;
    W << 0.0, -z, y, z, 0.0, -x, -y, x, 0.0;
    if (d < 1e-5) {
        Eigen::Matrix3d res = Eigen::Matrix3d::Identity() + W + 0.5 * W * W;
        return NormalizeRotation(res);
    } else {
        Eigen::Matrix3d res = Eigen::Matrix3d::Identity() + W * sin(d) / d + W * W * (1.0 - cos(d)) / d2;
        return NormalizeRotation(res);
    }
}

void Propagate(const Eigen::Vector3d &gyr, const Eigen::Vector3d &acc, ErrorState &errorState, Estimate &estimate) {
    // 引用赋值
    auto dq = errorState.dX.block(0, 0, 3, 1);
    auto dp = errorState.dX.block(3, 0, 3, 1);
    auto dv = errorState.dX.block(6, 0, 3, 1);
    auto dBg = errorState.dX.block(9, 0, 3, 1);
    auto dBa = errorState.dX.block(12, 0, 3, 1);
    Eigen::Matrix<double, 15, 15> &Cov = errorState.Cov;

    Eigen::Quaterniond &q = estimate.Q;
    Eigen::Vector3d &p = estimate.P;
    Eigen::Vector3d &v = estimate.V;
    Eigen::Vector3d &bg = estimate.Bg;
    Eigen::Vector3d &ba = estimate.Ba;

    const double dt = kImuSamplePeriod;
    const double dt2 = dt * dt;

    // 添加bias
    Eigen::Vector3d ab = acc + accBias;
    Eigen::Vector3d wb = gyr + gyrBias;

    // 添加噪声数据以及噪声矩阵Q
    if(kAddImuSensorNoise) {
        static std::default_random_engine generator;
        static std::normal_distribution<double> normalAcc(0, kAccNoiseStd);
        static std::normal_distribution<double> normalGyr(0, kGyrNoiseStd);
        ab += Eigen::Vector3d(normalAcc(generator), normalAcc(generator), normalAcc(generator));
        wb += Eigen::Vector3d(normalGyr(generator), normalGyr(generator), normalGyr(generator));
    }


    // OK，根据开头的推导写出误差状态转移方程F
    Eigen::MatrixXd F(15, 15);
    F.setZero();
    
    // about δQ
    F.block(0, 0, 3, 3) = ExpSO3(-(wb - bg) * dt);
    F.block(0, 9, 3, 3) = -Eigen::Matrix3d::Identity() * dt;

    // about δP
    F.block(3, 3, 3, 3) = Eigen::Matrix3d::Identity();
    F.block(3, 6, 3, 3) = Eigen::Matrix3d::Identity() * dt;

    // about δV
    F.block(6, 0, 3, 3) = -q.toRotationMatrix() * skewSymmetric(ab - ba) * dt;
    F.block(6, 6, 3, 3) = Eigen::Matrix3d::Identity();
    F.block(6, 12, 3, 3) = -q.toRotationMatrix() * dt;

    // about δBg
    F.block(9, 9, 3, 3) = Eigen::Matrix3d::Identity();

    // about δBa
    F.block(12, 12, 3, 3) = Eigen::Matrix3d::Identity();

    // 更新误差状态量，这里直接采用分块矩阵计算
    dp = F.block(3, 3, 3, 3) * dp + F.block(3, 6, 3, 3) * dv;
    dv = F.block(6, 0, 3, 3) * dp + F.block(6, 6, 3, 3) * dv + F.block(6, 12, 3, 3) * dBa;
    dq = F.block(0, 0, 3, 3) * dp + F.block(0, 9, 3, 3) * dBg;

    // 更新误差状态量
    errorState.dX = F * errorState.dX;
    // 更新协方差矩阵
    Cov = F * Cov * F.transpose();
    if(kAddImuSensorNoise) {
        Eigen::Matrix<double, 15, 15> Q = Eigen::Matrix<double, 15, 15>::Zero();
        Q.block(0, 0, 3, 3) = Eigen::Matrix3d::Identity() * pow(kNoiseAngStd, 2);
        Q.block(6, 6, 3, 3) = Eigen::Matrix3d::Identity() * pow(kNoiseVelStd, 2);
        Q.block(9, 9, 3, 3) = Eigen::Matrix3d::Identity() * pow(kNoiseBgStd, 2);
        Q.block(12, 12, 3, 3) = Eigen::Matrix3d::Identity() * pow(kNoiseBaStd, 2);
        Cov += Q;
    }


    // 扣除估计的bias
    const Eigen::Vector3d a_t = ab - ba;
    const Eigen::Vector3d w_t = wb - bg;
    // 更新名义状态量
    p += v * dt + 0.5 * (q * a_t) * dt2 + 0.5 * Gw * dt2; 
    v += q * a_t * dt + Gw * dt;
    q *= DeltaQ(w_t * dt);
}

void UpdatePwObv(const Eigen::Vector3d &Pw, const Estimate &estimate, Eigen::Matrix<double, 6, 15> &H_p_v, 
                    Eigen::Matrix<double, 6, 1> &residual_p_v) {
    // 残差是[3x1]维向量，状态向量维度是[15x1]维
    const Eigen::Vector3d dp = Pw - estimate.P;
    // 易知，关于position的量测方程对于 δP 的雅可比为I
    Eigen::Matrix3d Hp = Eigen::Matrix3d::Identity();
    residual_p_v.middleRows(0, 3) = dp;
    H_p_v.block(0, 3, 3, 3) = Hp;
}

void UpdateVwObv(const Eigen::Vector3d &Vw, const Estimate &estimate, Eigen::Matrix<double, 6, 15> &H_p_v, 
                  Eigen::Matrix<double, 6, 1> &residual_q_p_v) {
    // 残差是[3x1]维向量，状态向量维度是[15x1]维
    const Eigen::Vector3d dv = Vw - estimate.V;
    // 易知，关于position的量测方程对于 δP 的雅可比为I
    Eigen::Matrix3d Hv = Eigen::Matrix3d::Identity();
    residual_q_p_v.middleRows(3, 3) = dv;
    H_p_v.block(3, 6, 3, 3) = Hv;
}

void UpdateESKF(const Eigen::Matrix<double, 6, 15> &H_p_v, const Eigen::Matrix<double, 6, 1> &residual_p_v, 
                  const Eigen::Matrix3d &obvPcov, const Eigen::Matrix3d &obvVcov,
                  Estimate &estimate, ErrorState &errorState) {
    // 初始化ESKF更新步骤所需参数
    const auto &H = H_p_v;
    const auto &r = residual_p_v;
    Eigen::Matrix<double, 6, 6> obvCov;
    obvCov.block(0, 0, 3, 3) = obvPcov;
    obvCov.block(3, 3, 3, 3) = obvVcov;
    cout << "residual | norm: " << r.transpose() << " | " << r.norm() << endl;

    // 计算卡尔曼增益
    Eigen::Matrix<double, 15, 15> &Cov = errorState.Cov;
    Eigen::Matrix<double, 15, 6> K = Cov * H.transpose() * (H * Cov * H.transpose() + obvCov).inverse();
    // 更新误差状态量
    const Eigen::Matrix<double, 15, 1> dx = K * r + errorState.dX;
    // 因为Ba、Bg不是直接显含在量测方程，通过协方差使其在转移方程中方差变小了
    const Eigen::Matrix<double, 15, 15> I = Eigen::Matrix<double, 15, 15>::Identity();
    Cov = (I - K * H) * Cov;
    cout << "P V & Ba gain: " << setprecision(3) 
         << (I - K * H).diagonal().middleRows(3, 6).transpose() << " | "
         << (I - K * H).diagonal().middleRows(12, 3).transpose() << endl;

    // 更新反馈校正误差余量
    cout << "dx: " << setprecision(3) << dx.transpose() << endl;
    const Eigen::Matrix<double, 15, 1> updateDx = dx * kUpdateRatioEachStep;
    errorState.dX = dx - updateDx;

    
    // 最后利用误差量更新当前名义状态值
    estimate.Q *= DeltaQ(updateDx.middleRows(0, 3));
    estimate.P += updateDx.middleRows(3, 3);
    estimate.V += updateDx.middleRows(6, 3);
    estimate.Bg += updateDx.middleRows(9, 3);
    estimate.Ba += updateDx.middleRows(12, 3);

    // 校验协方差矩阵在合理范围内
    auto covDiagonal = Cov.diagonal();
    for(int i = 0; i < 5; ++i) {
        auto curCov = covDiagonal.middleRows(i*3, 3);
        for(int i = 0; i < 3; ++i) {
            if(curCov[i] < kMinQPVBaBgVariance[i]) {
                curCov[i] = kMinQPVBaBgVariance[i];
            }
        }
    }    
}
