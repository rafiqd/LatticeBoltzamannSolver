#include "Boltzman.h"
#include <fmt/core.h>

D2Q9::D2Q9()
{
  constexpr int Q = 9;
  C = Eigen::Matrix2Xf(2, Q); 
  std::vector<std::pair<int, int>> directions = {
    {0, 0},
    {1, 0}, {0, 1}, {-1, 0}, {0, -1},
    {1, 1}, {-1, 1}, {-1, -1}, {1, -1}
  };
  for (int i = 0; i < Q; ++i)
  {
    C.col(i)[0] = directions[i].first;
    C.col(i)[1] = directions[i].second;
  }
  
  C_int = C.cast<int>();

  constexpr float a_1 = 4.0f / 9.0f;
  constexpr float a_2 = 1.0f / 9.0f;
  constexpr float a_3 = 1.0f / 36.0f;

  w = (Eigen::VectorXf(Q) <<
    a_1,
    a_2,a_2,a_2,a_2,
    a_3,a_3,a_3,a_3).finished(); 
  
  bounce_back_mapping = (Eigen::VectorXi(Q) <<
    0, 3, 4, 1, 2, 7, 8, 5, 6
  ).finished();

  assert(C.cols() == Q);
  assert(C.cols() == w.size());
  assert(C.cols() == bounce_back_mapping.size());
}

const D2Q9 LatticeBoltzman2D9Q::_velocity_set = D2Q9();

LatticeBoltzman2D9Q::LatticeBoltzman2D9Q(Lattice2D lattice,
  Eigen::Matrix2Xf velocity_field, Eigen::VectorXf pressure_field)
  : _lattice(std::move(lattice)), _velocity_field(velocity_field), _pressure_field(pressure_field)
{
  
  assert(_velocity_set.C.cols() == _Q);


  if (_velocity_field.size() == 0)
  {
    _velocity_field = Eigen::Matrix2Xf(2, _lattice.numNodes());
    _velocity_field.setZero();
  } else if (_velocity_field.rows() != 2 &&
             _velocity_field.cols() != _lattice.numNodes())
  {
    throw std::runtime_error(fmt::format("The velocity field has wrong dimensions [{}x{}], should be [{}x{}]",
        _velocity_field.rows(), _velocity_field.cols(), 2,
        _lattice.numNodes()));
  }
  for (int node_num = 0; node_num < _lattice.numNodes(); ++node_num)
  {
    if (_lattice.getState(node_num) == Lattice2D::State::Solid ||
      _lattice.getState(node_num) == Lattice2D::State::Periodic)
    {
      _velocity_field.col(node_num).setZero();
    }
  }


  _mass_density = Eigen::VectorXf(_lattice.numNodes());
  _mass_density.setConstant(1.f);

  _momentum_density = Eigen::Matrix2Xf(2, _lattice.numNodes());
  _momentum_density.setZero();

  _c_s_2 = float((1.0 / 3.0) * ((pow(_delta_x, 2) / pow(_delta_t, 2))));
  _c_s = sqrtf(_c_s_2);
  _c_s_4 = powf(_c_s_2, 2);

  if (_pressure_field.size() == 0) {
    _pressure_field = Eigen::VectorXf(_lattice.numNodes());
    _pressure_field = _c_s_2 * _mass_density;
  } else if (_pressure_field.size() != _lattice.numNodes()) {
    throw std::runtime_error(fmt::format(
        "The pressure field has wrong dimensions [{}], should be [{}]",
        _pressure_field.size(), _lattice.numNodes()));
  }

  _F = Eigen::MatrixXf(_Q, _lattice.numNodes());
  _F.setZero();
  for (int node_num = 0; node_num < _lattice.numNodes(); ++node_num)
  {
    if (_lattice.isSolid(node_num)  || _lattice.getState(node_num) == Lattice2D::State::Periodic) {
      continue;
    }

    for (int i = 0; i < _Q; ++i)
    {
      _F(i, node_num) =
          compute_feq(_velocity_set.C.col(i), _velocity_field.col(node_num),
                      _velocity_set.w[i], _pressure_field[node_num]);
    }
  }



  _F_star = Eigen::MatrixXf(_Q, _lattice.numNodes());
  _F_star.setZero();
  
  _F_next = Eigen::MatrixXf(_Q, _lattice.numNodes());
  _F_next.setZero();

  _tau = 0.63f;
  _nu = (_tau - 0.5 ) / 3.0;
}

void LatticeBoltzman2D9Q::solve()
{
  const float t_over_tau = _delta_t / _tau;
  const float one_minus_t_over_tau = 1 - t_over_tau;

  for (int node_num = 0; node_num < _lattice.numNodes(); ++node_num)
  {
    auto state = _lattice.getState(node_num);
    switch (state) {
    case Lattice2D::State::Simulate:
    case Lattice2D::State::Dirichlet:
    case Lattice2D::State::Neuman: {

      // compute density, momentum, and velocity
      _momentum_density.col(node_num) = _velocity_set.C * _F.col(node_num);
      auto rho = _F.col(node_num).sum();
      assert(fabs(rho) > 1e-5f);
      _velocity_field.col(node_num) = _momentum_density.col(node_num) / rho;
      assert(!_velocity_field.hasNaN());
      _mass_density[node_num] = rho;
      
      // collision step
      Eigen::VectorXf feq;
      feq.resize(_Q);
      for (int i = 0; i < _Q; ++i) {
        feq[i] = 
            compute_feq(_velocity_set.C.col(i), _velocity_field.col(node_num),
                                _velocity_set.w[i], rho);
      }

      auto feq_rho = feq.sum();
      const float eps = fabs(feq_rho - rho);
      //assert(eps < 1e-4f);
      _F_star.col(node_num) = _F.col(node_num) * one_minus_t_over_tau + feq * t_over_tau;

      break;
    }
    case Lattice2D::State::Periodic:
    case Lattice2D::State::Solid: {
      break;
    }
    }
  }

  _pressure_field = _mass_density * _c_s_2;

  for (int node_num = 0; node_num < _lattice.numNodes(); ++node_num)
  {
    if (_lattice.getState(node_num) != Lattice2D::State::Simulate)
    {
      continue;
    }

    const auto idx_2d = _lattice.get2DIndex(node_num);
    const auto row = idx_2d.first;
    const auto col = idx_2d.second;
    for (int i = 0; i < _Q; ++i) {
      auto next_row = row + _velocity_set.C_int(1, i);
      auto next_col = col + _velocity_set.C_int(0, i);
      assert(0 <= next_row && next_row <= _lattice.height());
      assert(0 <= next_col && next_col <= _lattice.width());
      const auto next_node_num = _lattice.get1DIndex(next_row, next_col);
      assert(next_node_num < _lattice.numNodes());
      _F_next(i, next_node_num) = _F_star(i, node_num);
    }
  }

  const auto &dirichlet_bc_nodes = _lattice.getDirichletBoundaries();
  for (const auto &node : dirichlet_bc_nodes) {
    const auto node_num = node.node_num;
    const Eigen::Vector2f velocity = node.v.cast<float>();
    
    // Compute F taking into consideration the dirichlet BC
    Eigen::VectorXf f_local;
    f_local.resize(_Q);
    const auto rho = _mass_density[node_num];
    for (int i = 0; i < _Q; ++i) {
      const Eigen::Vector2f c_i = _velocity_set.C.col(i);
      f_local[i] = _F_star(i, node_num) - 2.0f * _velocity_set.w[i] *
                                     rho *
                                     (c_i.dot(velocity) / _c_s_2);
    }

    // Stream to adjacent nodes that ARENT solid (or out of bounds)
    // for nodes that are out of bounds or solid, we bounce their
    // stream back to the conjugate index
    const auto idx_2d = _lattice.get2DIndex(node_num);
    const auto row = idx_2d.first;
    const auto col = idx_2d.second;
    for (int i = 0; i < _Q; ++i) {

      auto next_row = row + _velocity_set.C_int(1, i);
      if (next_row < 0 || next_row >= _lattice.width()) { 
        auto bounced_idx = _velocity_set.bounce_back_mapping[i];
        _F_next(bounced_idx, node_num) = f_local[i];
        continue;
      }

      auto next_col = col + _velocity_set.C_int(0, i);
      if (next_col < 0 || next_col >= _lattice.width()) {
        auto bounced_idx = _velocity_set.bounce_back_mapping[i];
        _F_next(bounced_idx, node_num) = f_local[i];
        continue;
      }

      const auto next_node_num = _lattice.get1DIndex(next_row, next_col);
      if (_lattice.getState(next_node_num) == Lattice2D::State::Solid) {
        auto bounced_idx = _velocity_set.bounce_back_mapping[i];
        _F_next(bounced_idx, node_num) = f_local[i];
        continue;
      }

      _F_next(i, next_node_num) = _F_star(i, node_num);
    }
    /*
    // Handle bounce back
    const auto lattice_position = node.adjacency_info;
    switch (lattice_position) {

      // Single Walls
    case DirichletBC::LatticePosition::LeftWall : { 
      _F_next(1, node_num) = f_local[3];
      _F_next(5, node_num) = f_local[7];
      _F_next(8, node_num) = f_local[6];
      break; 
    }
    case DirichletBC::LatticePosition::RightWall: {
      _F_next(3, node_num) = f_local[1];
      _F_next(7, node_num) = f_local[5];
      _F_next(6, node_num) = f_local[8];
      break;
    }
    case DirichletBC::LatticePosition::TopWall: {
      _F_next(4, node_num) = f_local[2];
      _F_next(7, node_num) = f_local[5];
      _F_next(8, node_num) = f_local[6];
      break;
    }
    case DirichletBC::LatticePosition::BottomWall: {
      _F_next(2, node_num) = f_local[4];
      _F_next(5, node_num) = f_local[7];
      _F_next(6, node_num) = f_local[8];
      break;
    }

    // Corners
    case DirichletBC::LatticePosition::LeftTopCorner: {
      _F_next(5, node_num) = f_local[7];
      _F_next(1, node_num) = f_local[3];
      _F_next(8, node_num) = f_local[6];
      _F_next(4, node_num) = f_local[2];
      _F_next(7, node_num) = f_local[5];
      break;
    }
    case DirichletBC::LatticePosition::LeftBottomCorner: {
      _F_next(8, node_num) = f_local[6];
      _F_next(1, node_num) = f_local[3];
      _F_next(5, node_num) = f_local[7];
      _F_next(2, node_num) = f_local[4];
      _F_next(6, node_num) = f_local[8];
      break;
    }
    case DirichletBC::LatticePosition::RightTopCorner: {
      _F_next(8, node_num) = f_local[6];
      _F_next(6, node_num) = f_local[2];
      _F_next(7, node_num) = f_local[5];
      _F_next(3, node_num) = f_local[1];
      _F_next(6, node_num) = f_local[8];
      break;
    }
    case DirichletBC::LatticePosition::RightBottomCorner: {
      _F_next(8, node_num) = f_local[5];
      _F_next(6, node_num) = f_local[1];
      _F_next(7, node_num) = f_local[8];
      _F_next(3, node_num) = f_local[4];
      _F_next(6, node_num) = f_local[7];
      break;
    }


   // 3 Walls
   case DirichletBC::LatticePosition::LeftTopRight: {
      break;
    }
    case DirichletBC::LatticePosition::TopRightBottom: {
      break;
    }
    case DirichletBC::LatticePosition::RightBottomLeft: {
      break;
    }
    case DirichletBC::LatticePosition::BottomLeftTop: {
      break;
    }

    // Fully surrounded                                        
    case DirichletBC::LatticePosition::All: {
      break;
    }
    }*/

  }

  // handle periodic BC
  for (int node_num = 0; node_num < _lattice.numNodes(); ++node_num) {
    if (_lattice.getState(node_num) != Lattice2D::State::Periodic) {
      continue;
    }

    auto twin_node = _lattice.getTwinPeriodic(node_num);
    assert(twin_node < _lattice.numNodes());
    _F_next.col(twin_node) += _F_next.col(node_num);
    _F_next.col(node_num).setZero();
  }

  Eigen::VectorXf col_sums = _F.colwise().sum();
  assert(!_F_next.hasNaN());
  if (_velocity_field.hasNaN())
  {
    throw std::runtime_error(fmt::format("nans detected in velocity"));
  }
  _F = _F_next;
  _F_next.setZero();
  _F_star.setZero();

}