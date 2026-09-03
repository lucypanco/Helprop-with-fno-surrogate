#ifndef HELPROP_KINEMATICS_H
#define HELPROP_KINEMATICS_H

#include <cmath>

#include "Unit.h"

namespace HelPropKinematics {

// A positive A denotes a nucleus; A == 0 denotes an electron or positron.
inline double energy_axis(double kinetic_energy, int A) {
  return A > 0 ? kinetic_energy / A : kinetic_energy;
}

inline double kinetic_to_momentum(double kinetic_energy, int A) {
  const double mass = A > 0 ? 0.938272 * Unit::GeV : 5.10998e-4 * Unit::GeV;
  if (A > 0)
    return std::sqrt(kinetic_energy * (kinetic_energy + 2. * mass)) * A;
  return std::sqrt(kinetic_energy * (kinetic_energy + 2. * mass));
}

inline double momentum_to_kinetic(double momentum, int A) {
  const double mass = A > 0 ? 0.938272 * Unit::GeV : 5.10998e-4 * Unit::GeV;
  if (A > 0)
    return std::sqrt(momentum / A * momentum / A + mass * mass) - mass;
  return std::sqrt(momentum * momentum + mass * mass) - mass;
}

}

#endif /* HELPROP_KINEMATICS_H */
