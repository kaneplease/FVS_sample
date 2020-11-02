#include <iostream>
#include <fstream>
#include <cmath>
#include <Eigen/Sparse>
#include <Eigen/Dense>
using namespace std;
using namespace Eigen;

int main() {
    //初期条件
    const int nstep = 8000;
    const int n = 100;   //格子点数
    const double dx = 0.1;
    const double dt = 0.0005;
    const double gamma = 1.4;

    Matrix<double, n, 3> q0;    //保存量
    Matrix<double, n, 3> q_old;    //保存量
    //Matrix<double, n, 3> q_mean;    //保存量
    Matrix<double, n, 3> q_new;    //保存量
    Matrix<double, n, 3> e0;    //フラックス
    Matrix<double, n, 3> e_old;    //フラックス
    Matrix<double, n, 3> e_new;    //フラックス
    Matrix<double, n, 3> e_mean;    //j+1/2の値をjに格納しているフラックス

    //左側の要素が大きな値
    const double rho0_l = 2.0;
    const double u0_l = 0;
    const double en0_l = 2.0;
    //右側の要素が小さな値
    const double rho0_r = 1.0;
    const double u0_r = 0;
    const double en0_r = 1.0;

    //  初期条件の設定
    q0 = MatrixXd::Zero(n,3);
    for(int i=0;i< static_cast<int>(n/2);i++){
        q0(i,0)=rho0_l;
        q0(i,1)=rho0_l*u0_l;
        q0(i,2)=en0_l;
    }
    for(int i=static_cast<int>(n/2);i<n;i++){
        q0(i,0)=rho0_r;
        q0(i,1)=rho0_r*u0_r;
        q0(i,2)=en0_r;
    }

    e0 = MatrixXd::Zero(n,3);
    for(int i=0;i< static_cast<int>(n/2);i++){
        e0(i,0)=rho0_l*u0_l;
        e0(i,1)=(gamma-1)*en0_l+0.5*(3-gamma)*rho0_l*u0_l*u0_l;
        e0(i,2)=gamma*en0_l*u0_l - 0.5*(gamma-1)*rho0_l*pow(u0_l,3.0);
    }
    for(int i=static_cast<int>(n/2);i<n;i++){
        e0(i,0)=rho0_r*u0_r;
        e0(i,1)=(gamma-1)*en0_r+0.5*(3-gamma)*rho0_r*u0_r*u0_r;
        e0(i,2)=gamma*en0_r*u0_r - 0.5*(gamma-1)*rho0_r*pow(u0_r,3.0);
    }
    q_old = q0;
    e_old = e0;

    //初期条件設定終わり

    //A_j+1/2を求めるためにRoeの平均計算する
    //j=0,nの点は値を保存
    //なので，計算は1~n-1までの格子点で行う

    //jの点ではplusのAしか用いられない
    Matrix<double, 3, 3> A_p_j;
    Matrix<double, 3, 3> R_j;
    Matrix<double, 3, 3> R_inv_j;
    Matrix<double, 3, 3> Gam_j;
    Matrix<double, 3, 3> Gam_abs_j;
    Matrix<double, 3, 3> Gam_p_j;

    //j+1の点ではminusのAしか用いられない
    Matrix<double, 3, 3> A_m_jp1;
    Matrix<double, 3, 3> R_jp1;
    Matrix<double, 3, 3> R_inv_jp1;
    Matrix<double, 3, 3> Gam_jp1;
    Matrix<double, 3, 3> Gam_abs_jp1;
    Matrix<double, 3, 3> Gam_m_jp1;

    double rho_ave;
    double u[2];
    double H[2];
    double c[2];
    double D;   //D=sqrt(rho[1]/rho[0])
    double p[2];
    double rho[2];
    double m[2];
    double e[2];
    //計算するときのパラメタ
    double a_par[2];
    double b_par[2];

    //時間ステップだけ行う
    for (int i=0; i<nstep; i++) {
        //  [0]は[j]の情報、[1]は[j+1]の情報
        for (int j = 0; j < n - 1; j++) {
            //  わかりにくいので名前を置き換え
            for (int k = 0; k < 2; k++) {
                rho[k] = q_old(j + k, 0);
                m[k] = q_old(j + k, 1);
                e[k] = q_old(j + k, 2);
                //  まずは，保存量からp,Hを計算
                p[k] = (gamma - 1) * (e[k] - 0.5 * rho[k] * pow(m[k] / rho[k], 2));
                H[k] = (e[k] + p[k]) / rho[k];
                u[k] = m[k]/rho[k];
                c[k] = sqrt((gamma - 1) * (H[k] - 0.5*pow(u[k],2)));
                //  さらに、行列の中で使うパラメタ
                b_par[k] = (gamma - 1)/pow(c[k],2);
                a_par[k] = 0.5 * b_par[k] * pow(u[k],2);
            }

            //A^(+)_(j)の計算
            R_j << 1, 1, 1,
                    u[0] - c[0], u[0], u[0] + c[0],
                    H[0] - u[0] * c[0], 0.5 * pow(u[0], 2), H[0] + u[0] * c[0];

            R_inv_j << 0.5 * (a_par[0] + u[0] / c[0]), 0.5 * (-b_par[0] * u[0] - 1 / c[0]), 0.5 * b_par[0],
                    1 - a_par[0], b_par[0] * u[0], -b_par[0],
                    0.5 * (a_par[0] - u[0] / c[0]), 0.5 * (-b_par[0] * u[0] + 1 / c[0]), 0.5 * b_par[0];

            Gam_j << (u[0] - c[0]), 0, 0,
                    0, (u[0]), 0,
                    0, 0, (u[0] + c[0]);

            Gam_abs_j << std::abs(u[0] - c[0]), 0, 0,
                    0, std::abs(u[0]), 0,
                    0, 0, std::abs(u[0] + c[0]);

            Gam_p_j = 0.5*(Gam_j + Gam_abs_j);
            A_p_j = R_j * Gam_p_j * R_inv_j;

            //A^(-)_(j+1)の計算
            R_jp1 << 1, 1, 1,
                    u[1] - c[1], u[1], u[1] + c[1],
                    H[1] - u[1] * c[1], 0.5 * pow(u[1], 2), H[1] + u[1] * c[1];

            R_inv_jp1 << 0.5 * (a_par[1] + u[1] / c[1]), 0.5 * (-b_par[1] * u[1] - 1 / c[1]), 0.5 * b_par[1],
                    1 - a_par[1], b_par[1] * u[1], -b_par[1],
                    0.5 * (a_par[1] - u[1] / c[1]), 0.5 * (-b_par[1] * u[1] + 1 / c[1]), 0.5 * b_par[1];

            Gam_jp1 << (u[1] - c[1]), 0, 0,
                    0, (u[1]), 0,
                    0, 0, (u[1] + c[1]);

            Gam_abs_jp1 << std::abs(u[1] - c[1]), 0, 0,
                    0, std::abs(u[1]), 0,
                    0, 0, std::abs(u[1] + c[1]);

            Gam_m_jp1 = 0.5*(Gam_jp1 - Gam_abs_jp1);
            A_m_jp1 = R_jp1 * Gam_m_jp1 * R_inv_jp1;

            //std::cout << Gam_p_j << std::endl;
            //j+1/2のフラックスの計算
            e_mean.row(j).transpose() = A_p_j * q_old.row(j).transpose() + A_m_jp1 * q_old.row(j+1).transpose();
        }
        /*if (i==0){
            std::cout << e_mean << std::endl;
        }*/


        //次の時刻の値に更新
        //j-1/2の値が要求されるので，更新できるのは1<j<n-1まで
        double p_new;
        double rho_new;
        double u_new;
        double ene_new;

        //境界付近の保存量は普遍であるとする
        e_new.row(0) = e_old.row(0);
        e_new.row(n-1) = e_old.row(n-1);
        q_new.row(0) = q_old.row(0);
        q_new.row(n-1) = q_old.row(n-1);

        for (int j = 1; j < n - 1; j++) {
            q_new.row(j) =
                    q_old.row(j) - dt / dx * (e_mean.row(j) - e_mean.row(j - 1));

            rho_new = q_new(j, 0);
            u_new = q_new(j, 1) / q_new(j, 0);
            ene_new = q_new(j, 2);
            p_new = (gamma - 1) * (ene_new - 0.5 * rho_new * pow(u_new, 2));

            e_new(j, 0) = rho_new * u_new;
            e_new(j, 1) = p_new + rho_new * pow(u_new, 2);
            e_new(j, 2) = (ene_new + p_new) * u_new;

        }

        q_old = q_new;
        e_old = e_new;

        if (i%100 == 0){
            std::cout << i << std::endl;
        }

        if (i == nstep-1){
            std::ofstream ofs("q_500_fvs.csv");
            if (!ofs) {
                std::cerr << "ファイルオープンに失敗" << std::endl;
                std::exit(1);

            }
            for (int i=0; i<n; i++){
                ofs << dx*i << "," << q_new(i,0) << "," << q_new(i,1)/q_new(i,0) << "," <<
                    (gamma - 1) * (q_new(i,2) - 0.5 * q_new(i,0) * pow(q_new(i,1)/q_new(i,0), 2)) << std::endl;
            }

        }

    }


    return 0;
}