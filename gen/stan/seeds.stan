// The seeds random-effect logistic regression
//
// log_tau is declared UNCONSTRAINED and the Jacobian of tau = exp(log_tau) is added by hand,
// exactly as samplingdb's Seeds does, so Stan samples in the same coordinates the target uses
// and its log_prob matches the torch one term for term. Declaring real<lower=0> tau instead
// would leave Stan to add the Jacobian and would sample in a different parameterisation.
data {
  int<lower=1> n_groups;
  array[n_groups] int<lower=0> r;
  array[n_groups] int<lower=0> n;
  vector[n_groups] x1;
  vector[n_groups] x2;
  real<lower=0> alpha;              // Gamma shape on tau
  real<lower=0> beta;               // Gamma rate on tau
  real<lower=0> coefficient_scale;  // prior sd of a_0, a_1, a_2, a_12
}
parameters {
  vector[4] a;                      // a_0, a_1, a_2, a_12
  vector[n_groups] b;
  real log_tau;
}
model {
  real tau = exp(log_tau);
  vector[n_groups] logits = a[1] + a[2] * x1 + a[3] * x2 + a[4] * (x1 .* x2) + b;

  // Gamma prior on tau, plus log|dtau/dlog_tau| = log_tau
  target += alpha * log(beta) - lgamma(alpha) + (alpha - 1) * log_tau - beta * tau + log_tau;
  target += -0.5 * dot_self(a / coefficient_scale)
            - 4 * (log(coefficient_scale) + 0.5 * log(2 * pi()));
  target += 0.5 * n_groups * log_tau - 0.5 * tau * dot_self(b)
            - n_groups * 0.5 * log(2 * pi());
  for (i in 1:n_groups) {
    target += r[i] * logits[i] - n[i] * log1p_exp(logits[i])
              + lchoose(n[i], r[i]);
  }
}
