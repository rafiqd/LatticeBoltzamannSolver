#pragma once

#include <vector>
#include <Eigen/Core>

#include "Lattice.h"

struct D2Q9 {
  Eigen::Matrix2Xf C;
  Eigen::Matrix2Xi C_int;
  Eigen::VectorXf w;
  Eigen::VectorXi bounce_back_mapping;

  D2Q9();
};

class LatticeBoltzman2D9Q {

public:

  LatticeBoltzman2D9Q(Lattice2D lattice,
    Eigen::Matrix2Xf velocity_field=Eigen::Matrix2Xf(),
    Eigen::VectorXf pressure_field=Eigen::VectorXf()
  );

  void solve();

  /*
  float computeFequilibrium(const Eigen::Vector2f c, 
    const Eigen::Vector2f u, const float w, const float density) const
  {
    const float u_c_dot = u.dot(c);
    const float u_c_dot_sq = powf(u_c_dot, 2);
    const float u_u = u.dot(u);
    const float eq = w * density * (1 + u_c_dot / _c_s_2 + u_c_dot_sq / (2 * _c_s_4) - u_u / (2 * _c_s_2));
    assert(eq >= 0);
    return eq;
  }
  */

    float compute_feq(const Eigen::Vector2f c, const Eigen::Vector2f u,
                            const float w, const float density) const {
    const float ci_dot_u = c.dot(u);
      const float u_dot_u = u.dot(u);
    const auto sum =
        (1.0 + 3.0 * ci_dot_u + 4.5 * ci_dot_u * ci_dot_u - 1.5 * u_dot_u);
    float eq =

        w * density * sum;
          ;

    return eq;
  }

  float computeMassDensity(int node_num, const Eigen::Ref<const Eigen::MatrixXf> vdfs) const
  {
    const auto velocity_distribution_function = vdfs.col(node_num);
    return velocity_distribution_function.sum();
  }

  Eigen::Vector2f computeMomentumDensity(int node_num, const Eigen::Ref<const Eigen::MatrixXf> vdfs) const
  {
    const auto velocity_distribution_function = vdfs.col(node_num);
    return _velocity_set.C * velocity_distribution_function;
  }

  const Lattice2D &getLattice() const noexcept { return _lattice; }
  const Eigen::Matrix2Xf & getVelocityField() const noexcept { return _velocity_field; }
  const Eigen::VectorXf & getPressureField() const noexcept { return _pressure_field; }


private:

  Lattice2D _lattice;

  static constexpr float _delta_x = 1.0f;
  static constexpr float _delta_t = 1.0f;

  float _tau;
  float _nu;
  float _c_s;
  float _c_s_2;
  float _c_s_4;
  static constexpr int _Q = 9;

  Eigen::Matrix2Xf _velocity_field;
  Eigen::VectorXf _pressure_field;
  Eigen::VectorXf _mass_density;
  Eigen::Matrix2Xf _momentum_density;
  Eigen::MatrixXf _F;
  Eigen::MatrixXf _F_star;
  Eigen::MatrixXf _F_next;
  
  static const D2Q9 _velocity_set;

  
};