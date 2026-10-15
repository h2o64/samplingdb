// Log Gaussian Cox process on the Finnish pine saplings
//
// The Cholesky factor is passed in as data rather than rebuilt here, so Stan and torch use the
// same matrix to the last bit. whitened selects between the two parameterisations, which are
// different geometries carrying different spec_ids: direct samples the field itself, whitened
// samples e with field = L e + mu.
data {
  int<lower=1> dim;
  matrix[dim, dim] cholesky;        // lower-triangular factor of the kernel
  vector[dim] mu;
  vector[dim] counts;
  real<lower=0> area;
  int<lower=0, upper=1> whitened;
}
parameters {
  vector[dim] x;
}
transformed parameters {
  vector[dim] field = whitened == 1 ? cholesky * x + mu : x;
}
model {
  if (whitened == 1) {
    target += -0.5 * dot_self(x);
  } else {
    target += -0.5 * dot_self(mdivide_left_tri_low(cholesky, x - mu));
  }
  target += dot_product(field, counts) - area * sum(exp(field));
}
