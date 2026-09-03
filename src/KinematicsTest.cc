#include <cassert>
#include <cmath>
#include <algorithm>
#include <iostream>

#include "kinematics.h"

using namespace HelPropKinematics;
using namespace Unit;

void check_round_trip(double kinetic_energy, int A) {
  const double momentum = kinetic_to_momentum(kinetic_energy * GeV, A);
  const double recovered = momentum_to_kinetic(momentum, A) / GeV;
  assert(std::abs(recovered - kinetic_energy) < 1e-12 * std::max(1., kinetic_energy));
}

int main() {
  check_round_trip(0.001, 0); // electron/positron kinetic energy in GeV
  check_round_trip(1., 0);
  check_round_trip(1., 1);    // proton kinetic energy per nucleon in GeV

  const double electron_momentum = kinetic_to_momentum(0.001 * GeV, 0);
  const double proton_momentum = kinetic_to_momentum(0.001 * GeV, 1);
  assert(electron_momentum < proton_momentum);
  assert(energy_axis(0.7 * GeV, 0) == 0.7 * GeV);
  assert(energy_axis(1.4 * GeV, 2) == 0.7 * GeV);

  std::cout << "kinematics checks passed\n";
  return 0;
}
