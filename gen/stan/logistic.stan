// Bayesian logistic regression
//
// The parameters are unconstrained, so this is a direct transcription of samplingdb's
// LogisticRegression.log_prob with no Jacobian term. Written with explicit target += rather
// than sampling statements so that Stan's log_prob matches the torch one CONSTANT INCLUDED,
// which is what gen/stan/generate.py asserts.
data {
  int<lower=1> n_data;
  int<lower=1> dim;
  matrix[n_data, dim] features;   // standardised, bias column first
  array[n_data] int<lower=0, upper=1> labels;
  real<lower=0> sigma_w;
}
parameters {
  vector[dim] w;
}
model {
  vector[n_data] logits = features * w;
  target += -0.5 * dot_self(w / sigma_w) - dim * (log(sigma_w) + 0.5 * log(2 * pi()));
  target += dot_product(to_vector(labels), logits) - sum(log1p_exp(logits));
}
