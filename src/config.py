"""Module-level parameters for every experiment, as module-level variables.

No external market data is used anywhere in this repository: every
experiment evaluates the contour-integral quantile method on analytically
specified working densities (Schadner, "A general technique for explicit
quantile representations"), so there is no universe, sample period, or
API key to configure here.
"""

# --- Shared numerical precision (Appendix D.1: "24-digit arithmetic") ---
CONTOUR_DPS = 24          # mpmath precision for the contour integral itself
NEWTON_REFERENCE_DPS = 30  # mpmath precision for Newton-method reference quantiles
CONTROL_CHECK_DPS = 50     # precision for CDF-mass / density-normalization control checks
GL_POINTS = 20             # points per Gauss-Legendre cell (Appendix D.1)
TRUNCATION_LOG_ORDER = 60.0  # truncate where neglected mass/integrand is of order e^-60

# --- exp_1: GIG quantile construction ---
GIG_TEST_PARAMS = [
    # (a, b, sigma) grid spanning a < 1, a = 1, a > 1 and small/large b
    (-0.5, 1.0, 1.0),
    (0.5, 2.0, 1.0),
    (1.0, 1.0, 2.0),
    (2.5, 0.5, 1.0),
    (-2.0, 3.0, 1.0),
]
GIG_PROB_LEVELS = [0.01, 0.05, 0.2, 0.4, 0.6, 0.8, 0.95, 0.99]
GIG_LEFT_TRUNCATION_RULE = "x = b / 130"  # paper_pipeline implementation_notes, exp_1

# --- exp_2: Generalized Beta (GB) quantile construction ---
GB_BETA_TEST_PARAMS = [(0.3, 0.6), (2.5, 3.0), (0.5, 0.5), (4.0, 1.5)]
GB_FAMILY_TEST_PARAMS = [
    # (a, b, c, rho, sigma): GB1 (rho<1), GB2 (rho=1), beta, beta-prime cases
    (2.0, 3.0, 1.0, 0.5, 1.0),
    (1.5, 2.0, 2.0, 1.0, 1.0),  # GB2
    (0.8, 1.2, 1.0, 0.0, 1.0),  # beta-like (rho=0 degenerates map; handled in module)
]
GB_PROB_LEVELS = [0.01, 0.05, 0.2, 0.4, 0.6, 0.8, 0.95, 0.99]
# GB2 direct-integral regularization order m = floor(1/(c*b)) (methodology, exp_2)

# --- exp_3: Generalized Gamma (GG) and GEV quantile construction ---
GG_TEST_PARAMS = [(0.4, 1.0), (1.0, 1.0), (2.5, 0.5), (1.0, 2.0)]  # (a, c)
GEV_TEST_PARAMS = [(-0.3, 1.0, 0.0), (0.0, 1.0, 0.0), (0.5, 2.0, 1.0)]  # (a, sigma, mu)
GG_GEV_PROB_LEVELS = [0.01, 0.05, 0.2, 0.4, 0.6, 0.8, 0.95, 0.99]

# --- exp_4: Generalized Hyperbolic (GH) quantile construction ---
GH_TEST_PARAMS = [
    (1.0, 1.5, 0.5),    # hyperbolic
    (-0.5, 2.0, 1.0),   # normal inverse Gaussian (NIG)
    (0.0, 1.0, 0.0),
]
GH_PROB_LEVELS = [0.01, 0.05, 0.2, 0.4, 0.6, 0.8, 0.95, 0.99]

# --- exp_5: Normal and log-normal quantile construction ---
LOGNORMAL_TEST_PARAMS = [(1.0, 0.25), (2.0, 0.5), (1.0, 1.0)]  # (sigma=median, a=log-sd)
NORMAL_PROB_LEVELS = [0.01, 0.05, 0.2, 0.4, 0.6, 0.8, 0.95, 0.99]

# --- exp_6: Student's t and Cauchy quantile construction ---
STUDENT_T_TEST_DOF = [0.5, 1.0, 2.0, 5.0, 30.0]  # a=1 is Cauchy; a=2 has closed form
STUDENT_T_PROB_LEVELS = [0.01, 0.05, 0.2, 0.4, 0.6, 0.8, 0.95, 0.99]

# --- exp_7: Variance Gamma (VG) quantile construction ---
VG_TEST_PARAMS = [(-0.3, 0.15), (1.5, 0.2), (0.2, -0.1)]  # (a, d)
VG_PROB_LEVELS = [0.01, 0.05, 0.2, 0.4, 0.6, 0.8, 0.95, 0.99]

# --- exp_8: Black-Scholes implied volatility via contour integral ---
# 42-case grid (methodology/implementation_notes, exp_8)
BS_LOG_MONEYNESS_GRID = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0]
BS_TOTAL_VOL_GRID = [0.1, 0.2, 0.4, 0.8, 1.5, 3.0]
BS_MATURITIES = [0.25, 1.0, 4.0]  # years, assigned cyclically across the grid
BS_QUAD_ABS_TOL = 2e-12  # scipy.integrate.quad epsabs (implementation_notes, exp_8)
BS_QUAD_REL_TOL = 2e-12  # scipy.integrate.quad epsrel
BS_REFERENCE_DPS = 100    # mpmath precision for high-precision reference prices
BS_INDEPENDENT_CHECK_DPS = [35, 50]  # independent mpmath checks at these precisions

# --- exp_9: Bachelier (normal) implied volatility via contour integral ---
BACHELIER_TEST_CASES = [
    # (m = |F-K|, Vbar = normalized time value, T)
    (1.0, 0.05, 1.0),
    (1.0, 0.2, 1.0),
    (2.0, 0.3, 0.5),
    (0.5, 0.01, 2.0),
]
BACHELIER_QUAD_ABS_TOL = 1e-30  # mpmath quadrature tolerance (not specified numerically in the paper)

# --- exp_10: cross-law numerical verification ---
# Twelve working laws and eight probability levels (methodology, exp_10).
VERIFICATION_PROB_LEVELS = [0.01, 0.05, 0.2, 0.4, 0.6, 0.8, 0.95, 0.99]
VERIFICATION_LAWS = [
    {"name": "beta(0.3,0.6)", "family": "beta", "params": {"a": 0.3, "b": 0.6}},
    {"name": "beta(2.5,3)", "family": "beta", "params": {"a": 2.5, "b": 3.0}},
    {"name": "gamma(a=0.4)", "family": "gamma", "params": {"a": 0.4}},
    {"name": "gamma(a=1)[exponential/GEV]", "family": "gamma", "params": {"a": 1.0}},
    {"name": "GIG(a=-0.5,b=1)[inverse Gaussian]", "family": "gig", "params": {"a": -0.5, "b": 1.0}},
    {"name": "GH(a=1,b=1.5,d=0.5)[hyperbolic]", "family": "gh", "params": {"a": 1.0, "b": 1.5, "d": 0.5}},
    {"name": "GH(a=-0.5,b=2,d=1)[NIG]", "family": "gh", "params": {"a": -0.5, "b": 2.0, "d": 1.0}},
    {"name": "normal", "family": "normal", "params": {}},
    {"name": "student_t(a=0.5)", "family": "student_t", "params": {"a": 0.5}},
    {"name": "student_t(a=5)", "family": "student_t", "params": {"a": 5.0}},
    {"name": "VG(a=-0.3,d=0.15)[unbounded at 0]", "family": "vg", "params": {"a": -0.3, "d": 0.15}},
    {"name": "VG(a=1.5,d=0.2)[bounded at 0]", "family": "vg", "params": {"a": 1.5, "d": 0.2}},
]
EXPECTED_MAX_RELATIVE_ERROR = 1e-17  # expected_outcomes, exp_10
