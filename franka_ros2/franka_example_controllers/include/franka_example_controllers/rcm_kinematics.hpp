/* By Celal Umut Kenanoglu
March 2023
umut.kenanoglu@tum.de

Multipriority control in kinematic level (Based on Hamid Sadeghian code)
TUM endowrist actuation box (actuation box joints are passive)

*/
#pragma once

#include <math.h>

#include <eigen3/Eigen/Dense>

using namespace Eigen;

typedef Matrix<double, 8, 8> Matrix8d;
typedef Matrix<double, 6, 6> Matrix6d;
typedef Matrix<double, 9, 1> Vector9d;
typedef Matrix<double, 7, 7> Matrix7d;
typedef Matrix<double, 8, 1> Vector8d;
typedef Matrix<double, 7, 1> Vector7d;
typedef Matrix<double, 6, 1> Vector6d;
typedef Matrix<double, 5, 1> Vector5d;

#define alpha_endo M_PI / 2.0
#define beta_endo (-M_PI) / 4.0

#define dx_endo -0.0765
#define dy_endo 0.023
#define dz_endo 0.0381

#define tool_length 0.59

#define Q_MAX_1_2_3_4 2.62
#define Q_MAX_5_7 5.26
#define Q_MAX_6 4.18
#define Q_MAX_OFFSET_1_4 -0.30
#define Q_MAX_OFFSET_2_3 -0.20
#define Q_MAX_OFFSET_5_6_7 -0.35

#define Q_MIN_1_2_3_4 -Q_MAX_1_2_3_4
#define Q_MIN_5_7 -Q_MAX_5_7
#define Q_MIN_6 -Q_MAX_6
#define Q_MIN_OFFSET_1_4 -Q_MAX_OFFSET_1_4
#define Q_MIN_OFFSET_2_3 -Q_MAX_OFFSET_2_3
#define Q_MIN_OFFSET_5_6_7 -Q_MAX_OFFSET_5_6_7

void RCMForwardKinematics(std::array<double, 7> q, double eta, Matrix4d &XX, Matrix4d &XXp, Matrix<double, 3, 8> &JJe, Matrix<double, 3, 8> &JJp);