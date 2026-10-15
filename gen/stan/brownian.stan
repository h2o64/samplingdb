// Brownian motion with unknown scales and a missing middle
//
// Both scales are declared as their logarithms, with the Jacobians folded in: a LogNormal(0, s)
// prior on a scale becomes N(0, s^2) on its log, which is what the target term below is. Same
// reason as seeds.stan -- Stan then samples in samplingdb's own coordinates.
data {
  int<lower=1> n_steps;
  vector[n_steps] y;
  array[n_steps] int<lower=0, upper=1> observed;
  real<lower=0> prior_log_scale;    // 2, the scale of the LogNormal priors
}
parameters {
  real log_inn;
  real log_obs;
  vector[n_steps] locations;
}
model {
  real inn = exp(log_inn);
  real obs = exp(log_obs);

  // LogNormal(0, prior_log_scale) on each scale, Jacobian included
  target += -0.5 * square(log_inn / prior_log_scale) - 0.5 * square(log_obs / prior_log_scale)
            - 2 * log(prior_log_scale) - log(2 * pi());

  // The random walk, the first step from zero
  target += -0.5 * square(locations[1] / inn) - log_inn - 0.5 * log(2 * pi());
  for (i in 2:n_steps) {
    target += -0.5 * square((locations[i] - locations[i - 1]) / inn) - log_inn
              - 0.5 * log(2 * pi());
  }

  // The observations, only where they exist
  for (i in 1:n_steps) {
    if (observed[i] == 1) {
      target += -0.5 * square((locations[i] - y[i]) / obs) - log_obs - 0.5 * log(2 * pi());
    }
  }
}
