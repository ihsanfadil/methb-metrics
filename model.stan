functions {

  real effective_dose(real d, real lambda, int t) {
    if (d <= 0 || t <= 1) return 0.0;
    return d * (1.0 - exp(-lambda * (t - 1)));
  }

  real power_dose_effect(real alpha_p, real beta, real d_eff) {
    if (d_eff <= 0) return 1.0;
    return 1.0 + exp(alpha_p) * pow(d_eff, beta);
  }

  vector methb_model(
    int    ndays,
    vector dose,
    real   logit_total_rate,
    real   alpha_p,
    real   beta,
    real   lambda,
    real   steady_state
  ) {
    vector[ndays] methb;
    real total_rate = inv_logit(logit_total_rate);
    real prod_rate  = steady_state * total_rate;
    real red_rate   = (1.0 - steady_state) * total_rate;

    methb[1] = steady_state;
    for (t in 2:ndays) {
      real d_eff     = effective_dose(dose[t - 1], lambda, t);
      real dose_mult = power_dose_effect(alpha_p, beta, d_eff);
      methb[t] =
        methb[t - 1]
        + prod_rate * dose_mult * (1.0 - methb[t - 1])
        - red_rate  * methb[t - 1];
    }
    return methb;
  }
}

data {
  int<lower=1> n_patients;
  int<lower=1> n_sites;

  array[n_patients] int<lower=1> ndays;
  array[n_patients] int<lower=1> nobs;

  vector<lower=0, upper=1>[sum(nobs)] methb;
  vector<lower=0>[sum(ndays)]         dose;

  array[sum(nobs)] int<lower=1, upper=sum(ndays)> ind_obs;
  array[sum(nobs)] int<lower=1, upper=n_patients> patient_id;
  array[sum(nobs)] int<lower=1, upper=n_sites>    site_id;
  array[n_patients] int<lower=1, upper=n_sites>   site_id_patient;

  vector[n_patients] age_c;
  vector[n_patients] age_rcs1;
  vector[n_patients] age_rcs2;
  array[n_patients] int<lower=0, upper=1> has_age;

  vector[n_patients] sex_c;

  vector[n_patients] hbday0_c;
  array[n_patients] int<lower=0, upper=1> has_hb;
}

parameters {

  real logit_total_rate;

  real alpha_pop;

  real beta_age;
  real beta_age_rcs1;
  real beta_age_rcs2;
  real beta_sex;
  real beta_hb;

  real beta;

  real<lower=0>      sigma_eta;
  vector[n_patients] z_eta;

  real log_lambda;

  vector[n_sites] delta_obs;

  real<lower=0> sigma;

  vector<lower=0, upper=1>[n_patients] steady_state;
}

transformed parameters {
  vector[sum(ndays)] pred_methb;

  real<lower=0> lambda = exp(log_lambda);

  vector[n_patients] eta     = z_eta * sigma_eta;
  vector[n_patients] alpha_p;

  for (p in 1:n_patients) {
    alpha_p[p] = alpha_pop
      + beta_age      * age_c[p]    * has_age[p]
      + beta_age_rcs1 * age_rcs1[p] * has_age[p]
      + beta_age_rcs2 * age_rcs2[p] * has_age[p]
      + beta_sex      * sex_c[p]
      + beta_hb       * hbday0_c[p] * has_hb[p]
      + eta[p];
  }

  {
    int pos = 1;
    for (p in 1:n_patients) {
      vector[ndays[p]] tmp = methb_model(
        ndays[p],
        dose[pos : pos + ndays[p] - 1],
        logit_total_rate,
        alpha_p[p],
        beta,
        lambda,
        steady_state[p]
      );
      pred_methb[pos : pos + ndays[p] - 1] = tmp;
      pos += ndays[p];
    }
  }
}

model {

  logit_total_rate ~ normal(-2.20, 0.50);

  alpha_pop ~ normal(2.59, 0.50);

  beta_age      ~ normal(0, 0.5);
  beta_age_rcs1 ~ normal(0, 0.5);
  beta_age_rcs2 ~ normal(0, 0.5);
  beta_sex      ~ normal(0, 0.5);
  beta_hb       ~ normal(0, 0.5);

  beta ~ normal(1.00, 0.50);

  sigma_eta ~ normal(1.0, 0.30);
  z_eta     ~ normal(0, 1);

  log_lambda ~ normal(-0.69, 0.35);

  delta_obs ~ normal(0, 0.40);
  if (n_sites > 1)
    sum(delta_obs) ~ normal(0, 0.20 * n_sites);

  sigma ~ normal(0, 0.25);

  steady_state ~ lognormal(log(0.01), 0.35);

  for (i in 1:sum(nobs)) {
    real pred   = pred_methb[ind_obs[i]];
    real pred_c = fmax(1e-9, pred);
    real mu_s   = fmax(pred_c * exp(delta_obs[site_id[i]]), 1e-6);
    methb[i] ~ lognormal(log(mu_s), sigma);
  }
}

generated quantities {

  vector[sum(ndays)] pred_methb_gq = pred_methb;
}
