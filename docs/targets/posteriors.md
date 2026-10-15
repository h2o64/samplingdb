# Bayesian posteriors (Tier 2)

Seven posteriors, each in an unconstrained parameterisation with its log-Jacobian written out by
hand. They work in float64. None has a known `log Z`, and none had a reference ensemble in the
literature (Blessing et al. 2024 report only ELBO and EUBO on them), so the references are this
library's Stan runs.

| Target | dim | `name` | Model |
| --- | --- | --- | --- |
| `LogisticRegression('ionosphere')` | 35 | `ionosphere` | Bayesian logistic regression, `w ~ N(0, I)` |
| `LogisticRegression('sonar')` | 61 | `sonar` | same |
| `LogisticRegression('german_credit')` | 25 | `german-credit` | same |
| `LogisticRegression('breast_cancer')` | 31 | `breast-cancer` | same |
| `Seeds` | 26 | `seeds` | random-effect binomial regression on the seeds data of Crowder (1978) |
| `Brownian` | 32 | `brownian` | `inference_gym`'s Brownian motion with two unknown scales and ten unobserved middle steps |
| `LGCP` | 1600 | `lgcp`, `lgcp-whitened` | log Gaussian Cox process on the 126 Finnish pine saplings, 40 x 40 grid |

The datasets are fetched from `DenisBless/variational_sampling_methods` at a pinned commit and
checksum.

## Models

**Logistic regression.** `log p(w) + sum_i [y_i s_i - softplus(s_i)]` with `s_i = w . u_i`. Features
are standardised, a bias column is prepended, so coordinate 0 is the intercept, and `sigma_w = 1`.
The same documented model is used for all four datasets. The hub implements German Credit and
Breast Cancer differently (features scaled without centring, prior scale 10, then the prior
omitted), so its reference-draw files for those two come from another density.

**Seeds.** `tau ~ Gamma(0.01, 0.01)`, `a_0, a_1, a_2, a_12 ~ N(0, 10^2)`, `b_i ~ N(0, 1 / tau)`,
and `r_i ~ Binomial(N_i, sigmoid(a_0 + a_1 x1_i + a_2 x2_i + a_12 x1_i x2_i + b_i))`. Coordinates
are `[a_0, a_1, a_2, a_12, b_1 .. b_21, log tau]`. The vague prior on `tau` makes `log tau` the neck
of a funnel.

**Brownian.** `alpha_inn, alpha_obs ~ LogNormal(0, 2)`, `x_1 ~ N(0, alpha_inn)`,
`x_i ~ N(x_{i-1}, alpha_inn)`, and `y_i ~ N(x_i, alpha_obs)` observed for `i` in 1..10 and 21..30.
Coordinates are `[log alpha_inn, log alpha_obs, x_1 .. x_30]`. The two scales form the second
funnel of the tier, and the middle ten steps are constrained only by the random walk.

**LGCP.** `log N(x; mu, K) + sum_i [x_i c_i - a exp(x_i)]` with an exponential kernel
`K = sigma^2 exp(-|m - n| / (40 beta))`, `sigma^2 = 1.91`, `beta = 1/33`,
`mu = log(126) - sigma^2 / 2` and `a = 1/1600`. `LGCP()` samples `x` directly, the setting of
Blessing et al. 2024. `LGCP(whitened=True)` samples `e` with `x = L e + mu`; it is a different
geometry with its own name. Both have a Stan reference set, but the direct one needs a dense
1600 x 1600 triangular solve per gradient: its run took 29 hours, against 30 minutes for the
whitened one.

## Metric panel

Every posterior reports moments relative to the reference spread:

| Metric | What it compares |
| --- | --- |
| `max_mean_err`, `mean_mean_err` | posterior means, in reference standard deviations, worst and average over coordinates |
| `max_std_ratio_err` | the worst ratio of standard deviations, minus one, both divided by `n` |
| `w1_first_coordinate` | 1D Wasserstein on coordinate 0 |

`return_hist=True` gives the per-coordinate `mean`, `ref_mean`, `std` and `ref_std`.

Each posterior then adds the quantities it is judged on, because an average over coordinates hides
the hard one:

| Target | Metrics |
| --- | --- |
| logistic regressions | `predictive_log_likelihood_err` on the log pointwise predictive density `mean_i log sum_s w_s p(y_i \| x_s)`, `predictive_accuracy_err` on the accuracy of the posterior predictive thresholded at one half |
| `Seeds` | `log_tau_mean_err` and `log_tau_w1` on the neck of the funnel, `max_fixed_effect_err` on the four fixed effects |
| `Brownian` | `scale_mean_err` on the two scales, `missing_middle_mean_err` on the unobserved steps, `observed_mean_err` on the observed ones |
| `LGCP` | `total_intensity_err` on `sum_i a exp(x_i)`, which the 126 observed pines pin; `neighbour_correlation_err` on the covariance of adjacent cells about the reference's mean field, which the kernel's correlation length sets |

Each of these compares the samples with the reference draws; the logistic and LGCP entries are the
absolute difference between the two sets' values. Read each number against its floor, the same
metric between two halves of the reference set. The funnels (Seeds, Brownian) have percent-level
floors where the logistic regressions are sub-percent.

The predictive density averages each data point's likelihood over the samples before the
logarithm (the lppd of Gelman, Hwang and Vehtari 2014, divided by the number of data points), so it
converges as the sample count grows.

## Splits

| Split | How it is built |
| --- | --- |
| test | Stan, `gen/stan/generate.py`, published as `stan/<name>_<spec_id>.npy` and downloaded on first use. The Stan models are written in this library's coordinates and checked against the torch density before sampling. See [Reference sets](../reference-sets.md#stan-posteriors). |
| test, `brownian` | Exact draws, `gen/exact/brownian.py`, published as `exact/brownian_<spec_id>.npy`. See [Reference sets](../reference-sets.md#exact-sampler-brownian-motion). |
| train | One 500-step MALA chain per sample, seed 0, from an independent `N(0, prior_scale^2 I)` draw, which is the prior of the logistic regressions; `LGCP` starts from its Gaussian prior `N(mu, K)`. It is informal: useful as a warm start, not for evaluation. The observed data on the target is this tier's real training data. |

Brownian's exact reference gives scale means of 0.1156 (innovation) and 0.1127 (observation).
`inference_gym`'s published ground truth gives 0.11985 and 0.10105, up to 0.31 posterior standard
deviations away, so read the panel against this library's reference.
