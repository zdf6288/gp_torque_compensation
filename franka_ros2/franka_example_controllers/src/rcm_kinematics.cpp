#ifndef RCM_KINEMATICS_HPP
#define RCM_KINEMATICS_HPP
#include "franka_example_controllers/rcm_kinematics.hpp"
#endif

static Matrix4d Ab0, A01, A12, A23, A34, A45, A56, A67, A7f, Afr, Arp, Ape, Afr1, Afr2, Afr3;
static Matrix4d Ab1, Ab2, Ab3, Ab4, Ab5, Ab6, Ab7, Abf, Abp, Abe;
static Matrix3d R1, R2, R3, R4, R5, R6, R7;
static Matrix<double, 6, 8> Je, Jp;
static Matrix<double, 5, 5> eye5;
static Matrix<double, 8, 8> eye8;
static double C1, S1, C2, S2, C3, S3, C4, S4, C5, S5, C6, S6, C7, S7;
static const double d1{0.333}, d3{0.316}, d5{0.384}, df{0.107};
static const double a3{0.0825}, a4{-0.0825}, a6{0.088};

static Vector3d z0, z1, z2, z3, z4, z5, z6, z7, zp, zer;
static Vector3d p0, p1, p2, p3, p4, p5, p6, p7, pp, pe;


void RCMForwardKinematics(std::array<double, 7> q, double eta, Matrix4d &XX, Matrix4d &XXp, Matrix<double, 3, 8> &JJe,
           Matrix<double, 3, 8> &JJp) 
{
    eye5.setIdentity();
    eye8.setIdentity();

    zer.setZero(); 

    C1 = cos(q[0]);
    S1 = sin(q[0]);
    C2 = cos(q[1]);
    S2 = sin(q[1]);
    C3 = cos(q[2]);
    S3 = sin(q[2]);
    C4 = cos(q[3]);
    S4 = sin(q[3]);
    C5 = cos(q[4]);
    S5 = sin(q[4]);
    C6 = cos(q[5]);
    S6 = sin(q[5]);
    C7 = cos(q[6]);
    S7 = sin(q[6]);

    Ab0 << 1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1;

    A01 << C1, -S1, 0, 0,
        S1, C1, 0, 0,
        0, 0, 1, d1,
        0, 0, 0, 1;

    A12 << C2, -S2, 0, 0,
        0, 0, 1, 0,
        -S2, -C2, 0, 0,
        0, 0, 0, 1;

    A23 << C3, -S3, 0, 0,
        0, 0, -1, -d3,
        S3, C3, 0, 0,
        0, 0, 0, 1;

    A34 << C4, -S4, 0, a3,
        0, 0, -1, 0,
        S4, C4, 0, 0,
        0, 0, 0, 1;

    A45 << C5, -S5, 0, a4,
        0, 0, 1, d5,
        -S5, -C5, 0, 0,
        0, 0, 0, 1;

    A56 << C6, -S6, 0, 0,
        0, 0, -1, 0,
        S6, C6, 0, 0,
        0, 0, 0, 1;

    A67 << C7, -S7, 0, a6,
        0, 0, -1, 0,
        S7, C7, 0, 0,
        0, 0, 0, 1;

    A7f << 1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, df,
        0, 0, 0, 1; 


    Afr1 << cos(beta_endo), -sin(beta_endo), 0, 0,
        sin(beta_endo), cos(beta_endo), 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1;

    Afr2 << cos(alpha_endo), 0, sin(alpha_endo), dx_endo,
        0, 1, 0, dy_endo,
        -sin(alpha_endo), 0, cos(alpha_endo), dz_endo,
        0, 0, 0, 1;


    Afr = Afr1 * Afr2;

    Arp << 1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, eta,
        0, 0, 0, 1;

    Ape << 1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, tool_length - eta,
        0, 0, 0, 1;

    Ab1 = Ab0 * A01;
    Ab2 = Ab1 * A12;
    Ab3 = Ab2 * A23;
    Ab4 = Ab3 * A34;
    Ab5 = Ab4 * A45;
    Ab6 = Ab5 * A56;
    Ab7 = Ab6 * A67;
    Abp = Ab7 * A7f * Afr * Arp;
    Abe = Abp * Ape;

    z0 = Ab0.block<3, 1>(0, 2);
    z1 = Ab1.block<3, 1>(0, 2);
    z2 = Ab2.block<3, 1>(0, 2);
    z3 = Ab3.block<3, 1>(0, 2);
    z4 = Ab4.block<3, 1>(0, 2);
    z5 = Ab5.block<3, 1>(0, 2);
    z6 = Ab6.block<3, 1>(0, 2);
    z7 = Ab7.block<3, 1>(0, 2);
    zp = Abp.block<3, 1>(0, 2);

    p0 = Ab0.block<3, 1>(0, 3);
    p1 = Ab1.block<3, 1>(0, 3);
    p2 = Ab2.block<3, 1>(0, 3);
    p3 = Ab3.block<3, 1>(0, 3);
    p4 = Ab4.block<3, 1>(0, 3);
    p5 = Ab5.block<3, 1>(0, 3);
    p6 = Ab6.block<3, 1>(0, 3);
    p7 = Ab7.block<3, 1>(0, 3);
    pp = Abp.block<3, 1>(0, 3);
    pe = Abe.block<3, 1>(0, 3);

    Je << z1.cross(pe - p1), z2.cross(pe - p2), z3.cross(pe - p3), z4.cross(pe - p4), z5.cross(pe - p5), z6.cross(pe - p6), z7.cross(pe - p7), zer,
        z1, z2, z3, z4, z5, z6, z7, zer;

    Jp << z1.cross(pp - p1), z2.cross(pp - p2), z3.cross(pp - p3), z4.cross(pp - p4), z5.cross(pp - p5), z6.cross(pp - p6), z7.cross(pp - p7), zp,
        z1, z2, z3, z4, z5, z6, z7, zer;


    JJe = Je.block<3, 8>(0, 0);
    XX = Abe;
    XXp = Abp;
    JJp = Jp.block<3, 8>(0, 0);
}