data {
  // Polynomial emulator coefficients
  int<lower=0> n_coeffs;       // number of polynomial coefficients
  int<lower=1> n_toa;          // number of TOA energy bins
  int<lower=1> n_lis;          // number of LIS energy bins
  int<lower=0> poly_degree;    // polynomial degree
  vector[n_coeffs] coeffs[n_toa, n_lis];  // polynomial coefficients

  // Energy grids (GeV)
  vector[n_toa] ETOA;
  vector[n_lis] ELIS;

  // Observed data
  int<lower=1> n_obs;          // number of observed data points
  vector[n_obs] E_obs;         // observed energy bins (GeV)
  vector[n_obs] F_obs;         // observed flux values
  vector<lower=0>[n_obs] F_err; // fractional uncertainties (e.g. 0.1 for 10%)

  // LIS flux at ELIS grid points
  vector[n_lis] F_LIS;
}

transformed data {
  // Proton rest mass in GeV (matches HelProp.cc)
  real m_proton = 0.938272;

  // Precompute momenta from kinetic energies
  vector[n_lis] pLIS;
  vector[n_toa] pTOA;
  vector[n_obs] p_obs;

  // Polynomial feature exponents: for each coefficient index k,
  // the exponents (d1, d2) such that feature_k = logD0^d1 * m^d2
  int exp_d1[n_coeffs];
  int exp_d2[n_coeffs];

  for (j in 1:n_lis)
    pLIS[j] = sqrt(ELIS[j] * (ELIS[j] + 2.0 * m_proton));
  for (i in 1:n_toa)
    pTOA[i] = sqrt(ETOA[i] * (ETOA[i] + 2.0 * m_proton));
  for (k in 1:n_obs)
    p_obs[k] = sqrt(E_obs[k] * (E_obs[k] + 2.0 * m_proton));

  // Build exponent table: ordered by total degree, then d1 descending
  {
    int idx = 1;
    for (deg in 0:poly_degree) {
      for (d1 in 0:deg) {
        exp_d1[idx] = d1;
        exp_d2[idx] = deg - d1;
        idx += 1;
      }
    }
  }
}

parameters {
  real<lower=0> D0;   // diffusion coefficient (1e22 cm^2/s)
  real m;              // co-rotation factor (dimensionless)
}

transformed parameters {
  real logD0 = log(D0);
}

model {
  // Priors
  D0 ~ lognormal(1.6, 0.7);   // center ~5, broad range
  m ~ normal(0, 1);            // centered at 0

  // Build polynomial features for current (logD0, m)
  vector[n_coeffs] phi;
  {
    real logD0_powers[poly_degree + 1];
    real m_powers[poly_degree + 1];
    logD0_powers[1] = 1.0;
    m_powers[1] = 1.0;
    for (d in 1:poly_degree) {
      logD0_powers[d + 1] = logD0_powers[d] * logD0;
      m_powers[d + 1] = m_powers[d] * m;
    }
    for (k in 1:n_coeffs)
      phi[k] = logD0_powers[exp_d1[k] + 1] * m_powers[exp_d2[k] + 1];
  }

  // Evaluate polynomial to get log(weight) for each matrix element
  // Then compute predicted TOA flux at each observed energy
  vector[n_obs] F_pred;

  for (k in 1:n_obs) {
    // Find which TOA bin this observed energy falls into (nearest neighbor)
    int i_toa = 1;
    real min_dist = fabs(log(E_obs[k]) - log(ETOA[1]));
    for (i in 2:n_toa) {
      real dist = fabs(log(E_obs[k]) - log(ETOA[i]));
      if (dist < min_dist) {
        min_dist = dist;
        i_toa = i;
      }
    }

    // Compute modulated flux: sum over LIS bins
    // F_TOA = sum_i [ weight(i_toa, i_lis) * F_LIS[i_lis] / pLIS[i_lis]^2 * pTOA[i_toa]^2 ]
    // weight is in log-space from the emulator, so exp() first
    real flux_sum = 0.0;
    for (j in 1:n_lis) {
      real log_w = 0.0;
      for (c in 1:n_coeffs)
        log_w += coeffs[i_toa, j, c] * phi[c];
      real w = exp(log_w);
      flux_sum += w * F_LIS[j] / (pLIS[j] * pLIS[j]);
    }
    F_pred[k] = flux_sum * pTOA[i_toa] * pTOA[i_toa];
  }

  // Log-normal likelihood (multiplicative errors)
  for (k in 1:n_obs) {
    real sigma = F_err[k];  // fractional uncertainty
    target += normal_lpdf(log(F_obs[k]) | log(F_pred[k]), sigma);
  }
}

generated quantities {
  // Posterior predictive flux at observed energies
  vector[n_obs] F_pred_gen;
  {
    vector[n_coeffs] phi;
    real logD0_powers[poly_degree + 1];
    real m_powers[poly_degree + 1];
    logD0_powers[1] = 1.0;
    m_powers[1] = 1.0;
    for (d in 1:poly_degree) {
      logD0_powers[d + 1] = logD0_powers[d] * logD0;
      m_powers[d + 1] = m_powers[d] * m;
    }
    for (k in 1:n_coeffs)
      phi[k] = logD0_powers[exp_d1[k] + 1] * m_powers[exp_d2[k] + 1];

    for (k in 1:n_obs) {
      int i_toa = 1;
      real min_dist = fabs(log(E_obs[k]) - log(ETOA[1]));
      for (i in 2:n_toa) {
        real dist = fabs(log(E_obs[k]) - log(ETOA[i]));
        if (dist < min_dist) {
          min_dist = dist;
          i_toa = i;
        }
      }
      real flux_sum = 0.0;
      for (j in 1:n_lis) {
        real log_w = 0.0;
        for (c in 1:n_coeffs)
          log_w += coeffs[i_toa, j, c] * phi[c];
        real w = exp(log_w);
        flux_sum += w * F_LIS[j] / (pLIS[j] * pLIS[j]);
      }
      F_pred_gen[k] = flux_sum * pTOA[i_toa] * pTOA[i_toa];
    }
  }
}
