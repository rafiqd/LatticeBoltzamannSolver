#include <chrono>
#include <thread>
#include <fmt/core.h>
#include <Eigen/Core>

#include "lbm/Boltzman.h"
#include "cnpy/cnpy.h"

int main()
{
  const int grid_width = 200;
  const int grid_height = 100;

  // Buid Lattice ======================================================================
  
  
  std::vector<std::pair<int, Eigen::Vector2f>> bcs;
  Eigen::Vector2f velocity{0.0, 0.0};
  const auto zero = Eigen::Vector2f::Zero();

  std::vector<int> solid_indices;
  for (int i = 1; i < grid_width-1; ++i) {

    auto upper_bc = Lattice2D::get1DIndex(grid_height, grid_height - 2, i);
    bcs.emplace_back(upper_bc, velocity);

    auto lower_bc = Lattice2D::get1DIndex(grid_height, 0, i);
    bcs.emplace_back(lower_bc, zero);

    auto upper_solid = Lattice2D::get1DIndex(grid_height, grid_height - 1, i);
    solid_indices.push_back(upper_solid);
  }

  solid_indices.push_back(
      Lattice2D::get1DIndex(grid_height, grid_height - 1, 0));
  solid_indices.push_back(
      Lattice2D::get1DIndex(grid_height, grid_height - 1, grid_width-1));

  std::vector<std::pair<int, int>> periodic_boundry;
  for (int i = 0; i < grid_height - 1; ++i)
  {
    auto p1 = Lattice2D::get1DIndex(grid_height, i, grid_width - 1);
    auto p1_twin = Lattice2D::get1DIndex(grid_height, i, 1);
    periodic_boundry.emplace_back(p1, p1_twin);

    auto p2 = Lattice2D::get1DIndex(grid_height, i, 0);
    auto p2_twin = Lattice2D::get1DIndex(grid_height, i, grid_width - 2);
    periodic_boundry.emplace_back(p2, p2_twin);

  }
  

  Lattice2D lattice(grid_width, grid_height, solid_indices, bcs, periodic_boundry);

  lattice.addSphere(grid_width / 4, grid_height / 2, 8);

  if (true)  // test code - delete
  {
    for (int row = 0; row < grid_height; ++row) {
      for (int col = 0; col < grid_width; ++col) {
        auto idx1d = lattice.get1DIndex(row, col);
        auto idx2d = lattice.get2DIndex(idx1d);
        assert(idx2d.first == row && idx2d.second == col);
      }
    }
  }


  // Initialize LBM ====================================================================
  Eigen::Matrix2Xf initial_velocity(2, grid_width * grid_height);
  initial_velocity.setZero();
  initial_velocity.topRows(1).setConstant(0.2);
  initial_velocity.bottomRows(1).setRandom();
  initial_velocity.bottomRows(1) *= 0.2;
  LatticeBoltzman2D9Q lbm(lattice, initial_velocity);


  // Simulate ==========================================================================
  int max_itr = 1000;
  std::string mode = "a";

  cnpy::npz_save("data/velocity.npz", "width", &grid_width, {1},
                 "w");
  cnpy::npz_save("data/velocity.npz", "height", &grid_height, {1},
                 "a");


  cnpy::npz_save("data/pressure.npz", "width", &grid_width, {1},
                 "w");
  cnpy::npz_save("data/pressure.npz", "height", &grid_height, {1},
                 "a");

  const auto &l = lbm.getLattice();
  std::vector<int> states;
  std::vector<int> dir_bc;
  std::vector<Eigen::Vector2f> dir_bc_v;
  l.saveState(states, dir_bc, dir_bc_v);
  cnpy::npz_save("data/velocity.npz", fmt::format("states"), &states[0],
                 {states.size()},"a");
  cnpy::npz_save("data/velocity.npz", fmt::format("dir_bc_idx"), &dir_bc[0],
                 {dir_bc.size()}, "a");
  cnpy::npz_save("data/velocity.npz", fmt::format("dir_bc_v"),
                 dir_bc_v[0].data(),
                 {2, dir_bc_v.size()}, "a");

  int i;
  for (i = 0; i < max_itr; ++i)
  {
    {
      // write out velocity-field
      const auto& vf = lbm.getVelocityField();
      cnpy::npz_save("data/velocity.npz", 
        fmt::format("velocity_{}", i), vf.data(),
        {static_cast<size_t>(vf.cols()), static_cast<size_t>(vf.rows())}, mode);
    }

    {
      const auto& p = lbm.getPressureField();
      cnpy::npz_save("data/pressure.npz",
        fmt::format("pressure_{}", i), p.data(),
        { static_cast<size_t>(p.size()) }, mode);
    }

    lbm.solve();
  }

  {
    // write out velocity-field
    const auto &vf = lbm.getVelocityField();
    cnpy::npz_save(
        "data/velocity.npz", fmt::format("velocity_{}", i), vf.data(),
        {static_cast<size_t>(vf.cols()), static_cast<size_t>(vf.rows())}, mode);
  }

  {
    const auto &p = lbm.getPressureField();
    cnpy::npz_save("data/pressure.npz", fmt::format("pressure_{}", i), p.data(),
                   {static_cast<size_t>(p.size())}, mode);
  }

  fmt::print("Program successfully complete!");
  return 0;
}