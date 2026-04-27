data {
  // Polynomial emulator coefficients
  int<lower=1> n_params;       // number of parameters (always 4: D0, m, B0, angle)
  int<lower=1> n_coeffs;       // number of polynomial coefficients
  int<lower=1> n_toa;          // number of TOA energy bins
  int<lower=1> n_lis;          // number of LIS energy bins
  int<lower=0> poly_degree;    // polynomial degree
  vector[n_coeffs] coeffs[n_toa, n_lis];  // polynomial coefficients
  int<lower=0> exp_table[n_coeffs, n_params]; // exponent table

  // Energy grids (GeV)
  vector[n_toa] ETOA;
  vector[n_lis] ELIS;

  // Observed data
  int<lower=1> n_obs;
  vector[n_obs] E_obs;
  vector[n_obs] F_obs;
  vector<lower=0>[n_obs] F_err;

  // LIS flux at ELIS grid points
  vector[n_lis] F_LIS;

  // Fixed values and tight-prior widths for parameters not being inferred.
  // When a parameter is inferred, its prior_sigma is set to a broad value
  // and fixed_value is ignored.  When not inferred, prior_sigma is tiny
  // to effectively pin the parameter at fixed_value.
  real<lower=0> prior_mu_B0;
  real<lower=0> prior_sigma_B0;
  real prior_mu_angle;
  real<lower=0> prior_sigma_angle;
}

transformed data {
  real m_proton = 0.938272;
  vector[n_lis] pLIS;
  vector[n_toa] pTOA;
  vector[n_toa] log_ETOA;

  for (j in 1:n_lis)
    pLIS[j] = sqrt(ELIS[j] * (ELIS[j] + 2.0 * m_proton));
  for (i in 1:n_toa) {
    pTOA[i] = sqrt(ETOA[i] * (ETOA[i] + 2.0 * m_proton));
    log_ETOA[i] = log(ETOA[i]);
  }
}

parameters {
  real<lower=0> D0;
  real m_param;
  real<lower=0> B0;
  real angle;
}

transformed parameters {
  vector[n_params] theta;
  theta[1] = log(D0);
  theta[2] = m_param;
  theta[3] = B0;
  theta[4] = angle;
}

model {
  // Priors (width controlled by data for B0/angle to allow fixing them)
  D0 ~ lognormal(1.6, 0.7);
  m_param ~ normal(0, 1);
  B0 ~ normal(prior_mu_B0, prior_sigma_B0);
  angle ~ normal(prior_mu_angle, prior_sigma_angle);

  // Build polynomial features for current theta
  vector[n_coeffs] phi;
  for (k in 1:n_coeffs) {
    phi[k] = 1.0;
    for (p in 1:n_params)
      phi[k] *= theta[p] ^ exp_table[k, p];
  }

  // Compute log-predicted flux at each observed energy
  for (k in 1:n_obs) {
    real log_E_obs = log(E_obs[k]);

    // Find bracketing ETOA bins via binary search
    int i_lo;
    int i_hi;
    if (log_E_obs <= log_ETOA[1]) {
      i_lo = 1;
      i_hi = 2;
    } else if (log_E_obs >= log_ETOA[n_toa]) {
      i_lo = n_toa - 1;
      i_hi = n_toa;
    } else {
      int i_a = 1;
      int i_b = n_toa;
      while (i_b - i_a > 1) {
        int i_mid = (i_a + i_b) / 2;
        if (log_ETOA[i_mid] <= log_E_obs)
          i_a = i_mid;
        else
          i_b = i_mid;
      }
      i_lo = i_a;
      i_hi = i_b;
    }

    // Log-linear interpolation weight
    real w = (log_E_obs - log_ETOA[i_lo]) / (log_ETOA[i_hi] - log_ETOA[i_lo]);

    // Compute log(flux) at bracketing bins using log_sum_exp
    real log_flux_lo = negative_infinity();
    real log_flux_hi = negative_infinity();
    for (j in 1:n_lis) {
      real log_w_lo = 0.0;
      real log_w_hi = 0.0;
      for (c in 1:n_coeffs) {
        log_w_lo += coeffs[i_lo, j, c] * phi[c];
        log_w_hi += coeffs[i_hi, j, c] * phi[c];
      }
      real log_contrib_lo = log_w_lo + log(F_LIS[j]) - 2.0 * log(pLIS[j]);
      real log_contrib_hi = log_w_hi + log(F_LIS[j]) - 2.0 * log(pLIS[j]);
      log_flux_lo = log_sum_exp(log_flux_lo, log_contrib_lo);
      log_flux_hi = log_sum_exp(log_flux_hi, log_contrib_hi);
    }
    log_flux_lo += 2.0 * log(pTOA[i_lo]);
    log_flux_hi += 2.0 * log(pTOA[i_hi]);

    real log_F_pred = (1 - w) * log_flux_lo + w * log_flux_hi;

    // Log-normal likelihood
    target += normal_lpdf(log(F_obs[k]) | log_F_pred, F_err[k]);
  }
}

generated quantities {
  // Posterior predictive flux at observed energies
  vector[n_obs] F_pred_gen;

  {
    vector[n_coeffs] phi;
    for (k in 1:n_coeffs) {
      phi[k] = 1.0;
      for (p in 1:n_params)
        phi[k] *= theta[p] ^ exp_table[k, p];
    }

    for (k in 1:n_obs) {
      real log_E_obs = log(E_obs[k]);

      int i_lo;
      int i_hi;
      if (log_E_obs <= log_ETOA[1]) {
        i_lo = 1;
        i_hi = 2;
      } else if (log_E_obs >= log_ETOA[n_toa]) {
        i_lo = n_toa - 1;
        i_hi = n_toa;
      } else {
        int i_a = 1;
        int i_b = n_toa;
        while (i_b - i_a > 1) {
          int i_mid = (i_a + i_b) / 2;
          if (log_ETOA[i_mid] <= log_E_obs)
            i_a = i_mid;
          else
            i_b = i_mid;
        }
        i_lo = i_a;
        i_hi = i_b;
      }

      real w = (log_E_obs - log_ETOA[i_lo]) / (log_ETOA[i_hi] - log_ETOA[i_lo]);

      real log_flux_lo = negative_infinity();
      real log_flux_hi = negative_infinity();
      for (j in 1:n_lis) {
        real log_w_lo = 0.0;
        real log_w_hi = 0.0;
        for (c in 1:n_coeffs) {
          log_w_lo += coeffs[i_lo, j, c] * phi[c];
          log_w_hi += coeffs[i_hi, j, c] * phi[c];
        }
        real log_contrib_lo = log_w_lo + log(F_LIS[j]) - 2.0 * log(pLIS[j]);
        real log_contrib_hi = log_w_hi + log(F_LIS[j]) - 2.0 * log(pLIS[j]);
        log_flux_lo = log_sum_exp(log_flux_lo, log_contrib_lo);
        log_flux_hi = log_sum_exp(log_flux_hi, log_contrib_hi);
      }
      log_flux_lo += 2.0 * log(pTOA[i_lo]);
      log_flux_hi += 2.0 * log(pTOA[i_hi]);

      F_pred_gen[k] = exp((1 - w) * log_flux_lo + w * log_flux_hi);
    }
  }
}
