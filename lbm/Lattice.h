#pragma once

#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <Eigen/Core>


struct DirichletBC {

  enum class LatticePosition : int
  { 
    LeftWall = 1,
    RightWall = 2,
    BottomWall = 8,
    TopWall = 4,

    LeftBottomCorner = LeftWall | BottomWall,
    RightBottomCorner = RightWall | BottomWall,
    LeftTopCorner = LeftWall | TopWall,
    RightTopCorner = RightWall | TopWall,
    
    // some edge cases where a bc is surrounded by 3 or 4 walls
    LeftTopRight = LeftWall | TopWall | RightWall,
    TopRightBottom = TopWall | RightWall | BottomWall,
    RightBottomLeft = RightWall | BottomWall | LeftWall,
    BottomLeftTop = BottomWall | LeftWall | TopWall,
    All = LeftWall | TopWall | RightWall | BottomWall
  };

  Eigen::Vector2f v;
  int node_num;
  LatticePosition adjacency_info;
  DirichletBC(int node_number, Eigen::Vector2f velocity, LatticePosition adj)
      : node_num(node_number), v(velocity), adjacency_info(adj) {}
};

class Lattice2D
{
public:

  enum class State : char
  {
    Simulate = 0,
    Dirichlet = 1,
    Neuman = 2,
    Solid = 3,
    Periodic = 4
  };

  Lattice2D(
      int width, int height, const std::vector<int> &solid_indices,
      const std::vector<std::pair<int, Eigen::Vector2f>> &dirichlet_nodes,
      const std::vector<std::pair<int, int>> &periodic_boundry_twins = {});
  
  static int get1DIndex(int height, int row, int col) noexcept {
    return col * height + row;
  }

  int get1DIndex(int row, int col) const noexcept {
    return col * _height + row;
  }

  std::pair<int, int> get2DIndex(int index) const noexcept {
    int row = index % _height;
    int col = index / _height;
    return {row, col};
  }

  State getState(int node_num) const noexcept { return _lattice[node_num]; }

  /*
  void setSimulate(int node_num) noexcept {
    auto prev = _lattice[node_num];
    switch (prev) {
    case State::Simulate: {
      // nothing to do
      break;
    }
    case State::Dirichlet: {
      auto itr = std::find(
          _dirichlet.begin(), _dirichlet.end(),
          [node_num](const auto &d) { return d.node_num == node_num; });
      _dirichlet.erase(itr);
      break;
    }
    case State::Neuman: {
      // todo
      break;
    }
    case State::Solid: {

      const auto idx_2d = get2DIndex(node_num);
      const auto row = idx_2d.first;
      const auto col = idx_2d.second;

      const auto left = col - 1;
      const auto right = col + 1;
      const auto up = row + 1;
      const auto down = row - 1;

      if (left >= 0)
      {
        auto idx = get1DIndex(row, left);
        auto itr = std::find(_dirichlet.begin(), _dirichlet.end(),
                  [idx](const auto &d) { return d.node_num == idx; });
        
        itr->adjacency_info =
            DirichletBC::LatticePosition(int(itr->adjacency_info) &
            ~int(DirichletBC::LatticePosition::RightWall)); 
      }

      if (right < _) {
        auto idx = get1DIndex(row, left);
        auto itr =
            std::find(_dirichlet.begin(), _dirichlet.end(),
                      [idx](const auto &d) { return d.node_num == idx; });

        itr->adjacency_info = DirichletBC::LatticePosition(
            int(itr->adjacency_info) &
            ~int(DirichletBC::LatticePosition::RightWall));
      }




      break;
    }
    }

    _lattice[node_num] = State::Simulate;

  }

  void setSolid(int node_num) noexcept { 
    setSimulate(node_num);
    _lattice[node_num] = State::Solid;  }
  
  */

  const std::vector<DirichletBC> &
  getDirichletBoundaries() const noexcept {
    return _dirichlet;
  }

  bool isSimulate(int node_num) const noexcept {
    return _lattice[node_num] == State::Simulate;
  }

  bool isSolid(int node_num) const noexcept {
    return _lattice[node_num] == State::Solid;
  }

  bool isBoundry(int node_num) const noexcept {
    return _lattice[node_num] == State::Dirichlet ||
           _lattice[node_num] == State::Neuman;
  }

  int getTwinPeriodic(int periodic_node) const {
    const auto itr = _periodic_boundry_twin_map.find(periodic_node);
    assert(itr != _periodic_boundry_twin_map.end());
    return itr->second;
  }

  int numNodes() const noexcept { return _num_nodes; }
  int width() const noexcept { return _width; }
  int height() const noexcept { return _height; }

  void saveState(std::vector<int>& out_state, std::vector<int> & out_dirichlet_indices,
                 std::vector<Eigen::Vector2f> &out_dirichlet_velocities) const
  { 

    out_state = std::vector<int>(_lattice.size());
    for (int i = 0; i < _num_nodes; ++i)
    {
      out_state[i] = int(_lattice[i]);
    }

    out_dirichlet_indices = std::vector<int>(_dirichlet.size());
    out_dirichlet_velocities = std::vector<Eigen::Vector2f>(_dirichlet.size());
    int i = 0;
    for (const auto& node : _dirichlet)
    {
      out_dirichlet_indices[i] = node.node_num;
      out_dirichlet_velocities[i] = node.v;
      ++i;
    }
  }

  std::string printLattice(){

    std::function<std::string(int)> fn = [&](int idx) {
      return std::to_string(int(_lattice[idx]));
    };

    return printLattice(fn);
  }

  std::string printLattice(const std::function<std::string(int)> & fn)
  {
    std::stringstream ss;

    for (int row = 0; row < _height; ++row) {
      ss << "| ";
      for (int col = 0; col < _width; ++col) {
        ss << fn(get1DIndex(row, col)) << " ";
      }
      ss << " |";
      ss << std::endl;
    }

    return ss.str();
  }

  void addSphere(int x, int y, float radius)
  { 
    std::vector<Eigen::Vector2i> directions =
    {{1, 0}, {-1, 0}, {0, 1}, {0, -1}, 
      {-1, -1}, {-1, 1}, {1, 1}, {1, -1}
    };
    std::unordered_set<int> potential_borders;

    for (int col = 0; col < _width; ++col)
    {
      for (int row = 0; row < _height; ++row)
      {
        auto distance = sqrt(pow(x - col, 2) + pow(y - row, 2));
        if (round(distance) < radius)
        {
          int idx = get1DIndex(row, col);
          _lattice[idx] = State::Solid;

          for (const auto d : directions)
          {
            auto next_row = row + d.y();
            auto next_col = col + d.x();
            if (next_row > 0 && next_row < _height && next_col > 0 &&
              next_col < _width)
            {
              auto new_idx = get1DIndex(next_row, next_col);
              potential_borders.insert(new_idx);
            }
          }

        }
      }
    }

    for (const auto border_idx : potential_borders)
    {
      const auto idx_2d = get2DIndex(border_idx);
      if (_lattice[border_idx] == State::Solid)
      {
        continue;
      }

      int row = idx_2d.first;
      int col = idx_2d.second;

      bool has_free_neighbour = false;
      for (const auto d : directions) {
        auto next_row = row + d.y();
        auto next_col = col + d.x();
        if (next_row > 0 && next_row < _height && next_col > 0 &&
            next_col < _width) {
          auto new_idx = get1DIndex(next_row, next_col);
          if (_lattice[new_idx] == State::Simulate ||
            _lattice[new_idx] == State::Dirichlet ||
            _lattice[new_idx] == State::Neuman)
          {
            has_free_neighbour = true;
          }
        }
      }

      if (has_free_neighbour)
      {
        _setDirichletBC(border_idx, Eigen::Vector2f::Zero());
      }
    }

  
  }


private:
  
  void _setDirichletBC(int node_num, Eigen::Vector2f velocity) {

    const auto idx2d = get2DIndex(node_num);
    const auto row = idx2d.first;
    const auto col = idx2d.second;

    auto left = col - 1;
    auto right = col + 1;
    auto up = row + 1;
    auto down = row - 1;

    int position = 0;
    if (left < 0 || _lattice[get1DIndex(row, left)] == State::Solid) {
      position |= 1;
    }

    if (right >= _width || _lattice[get1DIndex(row, right)] == State::Solid) {
      position |= 2;
    }

    if (up >= _height || _lattice[get1DIndex(up, col)] == State::Solid) {
      position |= 4;
    }

    if (down < 0 || _lattice[get1DIndex(down, col)] == State::Solid) {
      position |= 8;
    }

    _dirichlet.emplace_back(
        node_num, velocity,
        static_cast<DirichletBC::LatticePosition>(position));
    _lattice[node_num] = State::Dirichlet;
  }

  int _height;
  int _width;
  int _num_nodes;
  std::vector<State> _lattice;
  std::vector<DirichletBC> _dirichlet;
  std::unordered_map<int, int> _periodic_boundry_twin_map;
};