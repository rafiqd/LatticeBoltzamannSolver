#include "Lattice.h"

#include <fmt/core.h>

Lattice2D::Lattice2D(int width, int height, const std::vector<int> & solid_indices,
    const std::vector<std::pair<int, Eigen::Vector2f>> & dirichlet_nodes,
    const std::vector<std::pair<int, int>> &periodic_boundry_twins)
    : _width(width), _height(height), _num_nodes(width * height) {
  
  if (_height == 0 || _width == 0) {
    throw std::runtime_error(
        fmt::format("Grid height and width cannot be zero, got: [{}, {}]",
                    height, width));
  }
  _lattice = std::vector<State>(_num_nodes, State::Simulate);

  for (const auto solid_node_num : solid_indices)
  {
    if (solid_node_num >= _num_nodes || solid_node_num < 0) {
      throw std::runtime_error(fmt::format(
          "Invalid index `{}` given in the solid nodes, max index is `{}`",
          solid_node_num, _num_nodes));
    }

    _lattice[solid_node_num] = State::Solid;
  }

  for (const auto& p : periodic_boundry_twins)
  {
    const auto periodic_node = p.first;
    const auto periodic_twin = p.second;
    _lattice[periodic_node] = State::Periodic;
    auto itr = _periodic_boundry_twin_map.find(periodic_node);
    if (itr != _periodic_boundry_twin_map.end())
    {
      throw std::runtime_error(
          fmt::format("Node `{}` has already been marked as a periodic node."));
    }
    _periodic_boundry_twin_map.insert({periodic_node, periodic_twin});
  }
  
  for (auto d_node : dirichlet_nodes)
  {
    if (_lattice[d_node.first] != State::Simulate)
    {
      throw std::runtime_error(fmt::format(
          "Dirichlet node specified at a node `{}` that is marked as type `{}`, "
          "can only set nodes of type 0 with nothing on them as dirichlet BC",
          d_node.first, int(_lattice[d_node.first])));
    }

    _setDirichletBC(d_node.first, d_node.second);
  }


}