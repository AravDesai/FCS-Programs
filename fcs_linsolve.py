"""
fcs_linsolve.py -- linear least-squares decomposition of a correlation curve
onto a fixed basis.

Solves, in a weighted least-squares sense,

    min_d  || W (A d - G) ||^2          (optionally subject to d >= 0)

where

    G : (m,)    measured curve, m lag times
    A : (m, n)  design matrix, one basis vector per column
    d : (n,)    amplitudes returned by the fit
    W : (m, m)  diagonal weight matrix, W_ii = 1 / sigma_i

The solver half of this module is model-agnostic: it knows nothing about
diffusion and never builds A.  The basis half is a port of the MATLAB
`basismodel` and does know where the columns come from.  The two halves talk
only through the matrix A.

A is always evaluated on the lag vector the data actually have.  Nothing here
resamples, interpolates or assumes a lag grid, so a caller with multi-tau data
must rebuild A for each dataset rather than reusing one matrix across curves
whose lag vectors differ.

The non-negativity constraint may be applied per column, so the basis
amplitudes can be held at d >= 0 while the baseline column stays free to go
negative.

Conditioning is reported, never acted on.  When the basis is fine enough that
neighbouring columns are nearly parallel, `cond` will be large and the
amplitudes will be correspondingly unstable; that is a fact about the data and
the chosen grid, and this module states it rather than quietly altering the
problem to hide it.

STATUS: bare-bones.  Search "OPEN QUESTION" for decisions still outstanding.

Conventions followed from the rest of the suite:
  - module-level constants for runtime-adjustable settings, leading underscore
  - explicit warnings rather than silent fallbacks
  - clarity over speed
"""

import warnings

import numpy as np


# --------------------------------------------------------------------------
# Module-level settings
#
# The three _TAUFACTOR_* values are the intended fcs_prefs targets; see the
# note at the foot of this file.
# --------------------------------------------------------------------------

# Singular values below _RCOND * max(singular value) are treated as zero when
# inverting.  None defers to numpy's own default.
_RCOND = None

# Warn when the weighted normal-equation matrix is worse conditioned than this.
# Purely diagnostic; nothing changes automatically.
_COND_WARN = 1.0e8

# Relative tolerance below which a bounded amplitude counts as pinned at its
# bound rather than free.  Pinned parameters get no error bar.
_BOUND_ACTIVE_TOL = 1.0e-10

# Defaults for solve_coordinate_descent.
#
# _CD_MIN_DCHI2 is the improvement below which a move counts as not worth
# making.  1.0 is the conventional scale: adding one free parameter to a fit
# lowers chi2 by about 1 through noise alone, so a move buying less than that
# is not buying anything real.
#
# _CD_PATIENCE exists because the improvement sequence is NOT monotonic --
# measured runs go 3.6, 5.3, 2.1, 1.7, 1.3, 1.8, 1.3 -- so a single small step
# is not evidence of having arrived.  Requiring several consecutive small steps
# avoids stopping on a plateau the search then walks off.
# _CD_TARGET_CHI2_RED is the primary stop when a real sigma is supplied: once
# the fit agrees with the data to within the noise there is nothing left to
# find, and any further column is fitting noise.  It is scale free, which the
# chi2 threshold is not -- a curve sitting on a large offset starts with a large
# chi2, so its improvements stay above an absolute threshold for far longer and
# the search runs on, adding columns.  Measured: the same curve plus a constant
# 0.05 ran 278 rounds instead of 8 and shifted <N> by 6%.
_CD_MAX_ITER = 500
_CD_MIN_DCHI2 = 1.0
_CD_PATIENCE = 3
_CD_TARGET_CHI2_RED = 1.0

# Default taufactor grid.  Deliberately fine: the spread of the returned
# amplitudes is the quantity of interest, so the grid must not presuppose which
# discrete decay times are correct.
#
# taufactor = 1 is a characteristic lag time of 1 microsecond, so the span
# below covers 1 us to 1 ms at 20 points per decade (61 columns).  The two
# component times of that column are 5.385e-07 s and 2.083e-06 s, which bracket
# the 1 us anchor.
#
# All three are fcs_prefs targets under the "Linear decomposition" section.
_TAUFACTOR_MIN = 1.0e0
_TAUFACTOR_MAX = 1.0e3
_TAUFACTOR_PER_DECADE = 20

# Units of the lag vector handed to build_basis.  The MATLAB basis constants
# are in seconds; anything else is converted on the way in.
#
# fcs_corr holds lag times in SECONDS: it carries an explicit tau_ms = 1e3*tau
# conversion, which only makes sense if tau is already in seconds.  Both
# representations therefore circulate in the suite, and a millisecond vector
# handed to build_basis without saying so would still fit -- every recovered
# taufactor would simply be wrong by three decades, with no other symptom.
# Call sites should pass t_units explicitly rather than lean on this default.
_BASIS_T_UNITS = "s"

_T_UNIT_TO_SECONDS = {
    "s": 1.0,
    "ms": 1.0e-3,
    "us": 1.0e-6,
    "ns": 1.0e-9,
}


# --------------------------------------------------------------------------
# Result container
# --------------------------------------------------------------------------

class LinearFitResult(object):
    """Everything the solver knows about one decomposition.

    Attributes
    ----------
    d : (n,) ndarray
        Best-fit amplitudes.
    d_err : (n,) ndarray
        One-sigma uncertainties.  NaN for any amplitude pinned at a bound.
    d_cov : (n, n) ndarray
        Covariance matrix.  Rows/columns of pinned parameters are NaN.
    G_fit, residuals, residuals_weighted : (m,) ndarrays
    chi2 : float
        Sum of squared weighted residuals.
    n_free : int
        Number of amplitudes not pinned at a bound.
    dof : int
        m - n_free.
    chi2_red : float
        chi2 / dof.
    pinned : (n,) bool ndarray
        True where a lower bound is active.
    cond : float
        Condition number of the weighted normal-equation matrix over the free
        columns.  Large values mean the amplitudes are strongly degenerate.
    rank : int
        Numerical rank of the weighted design matrix.
    sigma_used : (m,) ndarray or None
    error_rescaled : bool
        Whether d_err was multiplied by sqrt(chi2_red) (Birge rescaling).
    method : str
        'lstsq' or 'bounded'.
    warnings : list of str
        Human-readable notes.  Always inspect this.
    """

    def __init__(self, **kw):
        for key, value in kw.items():
            setattr(self, key, value)

    def __repr__(self):
        return ("<LinearFitResult n=%d m=%d method=%s chi2_red=%.4g "
                "cond=%.3g warnings=%d>"
                % (len(self.d), len(self.G_fit), self.method, self.chi2_red,
                   self.cond, len(self.warnings)))

    def summary(self, max_rows=12):
        """Multi-line human-readable report."""
        n = len(self.d)
        lines = []
        lines.append("Linear decomposition (%s)" % self.method)
        lines.append("  data points m      : %d" % len(self.G_fit))
        lines.append("  basis vectors n    : %d" % n)
        lines.append("  weighting          : %s"
                     % ("per-point sigma" if self.sigma_used is not None
                        else "UNIT (unweighted)"))
        lines.append("  free amplitudes    : %d" % self.n_free)
        lines.append("  dof                : %d" % self.dof)
        lines.append("  chi2               : %.6g" % self.chi2)
        lines.append("  chi2_red           : %.6g" % self.chi2_red)
        lines.append("  condition number   : %.4g" % self.cond)
        lines.append("  rank               : %d of %d" % (self.rank, n))
        lines.append("  pinned at bound    : %d" % int(np.sum(self.pinned)))
        lines.append("  errors rescaled    : %s" % self.error_rescaled)
        if n <= max_rows:
            lines.append("  amplitudes:")
            for i in range(n):
                if self.pinned[i]:
                    lines.append("    d[%3d] = %14.6g   (pinned, no error)"
                                 % (i, self.d[i]))
                else:
                    lines.append("    d[%3d] = %14.6g +/- %-14.6g"
                                 % (i, self.d[i], self.d_err[i]))
        else:
            lines.append("  amplitudes: %d values, sum = %.6g, max = %.6g "
                         "at index %d" % (n, float(np.sum(self.d)),
                                          float(np.max(self.d)),
                                          int(np.argmax(self.d))))
        if self.warnings:
            lines.append("  warnings:")
            for w in self.warnings:
                lines.append("    ! " + w)
        return "\n".join(lines)


# --------------------------------------------------------------------------
# Input validation
# --------------------------------------------------------------------------

def _as_vector(x, name):
    arr = np.asarray(x, dtype=float).ravel()
    if arr.size == 0:
        raise ValueError("%s is empty" % name)
    if not np.all(np.isfinite(arr)):
        raise ValueError("%s contains non-finite values (NaN or inf)" % name)
    return arr


def _as_matrix(x, name):
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError("%s must be 2-D, got %d dimensions" % (name, arr.ndim))
    if arr.size == 0:
        raise ValueError("%s is empty" % name)
    if not np.all(np.isfinite(arr)):
        raise ValueError("%s contains non-finite values (NaN or inf)" % name)
    return arr


# --------------------------------------------------------------------------
# Core solver
# --------------------------------------------------------------------------

def _free_covariance(A_w, free, notes):
    """Covariance of the unpinned amplitudes, plus the condition number.

    Shared by both solvers so that an amplitude and its error mean the same
    thing whichever route produced them.  Returns an (n, n) array whose pinned
    rows and columns are NaN.
    """
    n = A_w.shape[1]
    d_cov = np.full((n, n), np.nan)
    idx = np.where(free)[0]
    if idx.size == 0:
        return d_cov, np.inf
    Af = A_w[:, free]
    gram = Af.T.dot(Af)
    try:
        cond = float(np.linalg.cond(gram))
    except np.linalg.LinAlgError:
        cond = np.inf
    if not np.isfinite(cond) or cond > _COND_WARN:
        notes.append("Normal-equation matrix is ill conditioned "
                     "(cond = %.4g). Neighbouring basis vectors are nearly "
                     "parallel over the measured lag range, so individual "
                     "amplitudes are strongly degenerate and their errors "
                     "are large and highly correlated." % cond)
    d_cov[np.ix_(idx, idx)] = np.linalg.pinv(
        gram, rcond=(1e-15 if _RCOND is None else _RCOND))
    return d_cov, cond


def solve_linear(G, A, sigma=None, nonneg=False, rescale_errors="auto"):
    """Least-squares decomposition of G onto the columns of A.

    Parameters
    ----------
    G : array_like, shape (m,)
    A : array_like, shape (m, n)
        Column j is basis vector j evaluated at the same m abscissae as G, in
        the same order.
    sigma : array_like, shape (m,), optional
        One-sigma uncertainty on each element of G.  If None, unit weights are
        used and a warning is recorded.
    nonneg : bool or array_like of bool, shape (n,)
        False leaves every amplitude free.  True constrains every amplitude to
        d >= 0.  A boolean mask constrains only the columns where it is True,
        which is how a free (possibly negative) baseline column is kept
        unconstrained while the basis amplitudes are held non-negative.
    rescale_errors : {"auto", True, False}
        Whether to multiply d_err by sqrt(chi2_red) (Birge rescaling).  "auto"
        rescales only when sigma is None.  Made explicit rather than defaulted
        because the rest of the suite is currently inconsistent on this point.

    Returns
    -------
    LinearFitResult
    """
    notes = []

    G = _as_vector(G, "G")
    A = _as_matrix(A, "A")

    if sigma is not None:
        sigma = _as_vector(sigma, "sigma")
        if np.any(sigma <= 0.0):
            raise ValueError("sigma must be strictly positive; got min %.6g"
                             % float(np.min(sigma)))

    m = G.size
    if A.shape[0] != m:
        raise ValueError("A has %d rows but G has %d elements"
                         % (A.shape[0], m))
    if sigma is not None and sigma.size != m:
        raise ValueError("sigma has %d elements but G has %d"
                         % (sigma.size, m))
    n = A.shape[1]

    # ---- bounds ----
    if isinstance(nonneg, (bool, np.bool_)):
        bounded = np.full(n, bool(nonneg))
    else:
        bounded = np.asarray(nonneg, dtype=bool).ravel()
        if bounded.size != n:
            raise ValueError("nonneg mask has %d elements but A has %d columns"
                             % (bounded.size, n))
    any_bounded = bool(np.any(bounded))

    if n > m:
        notes.append("n (%d) exceeds m (%d): the system is underdetermined "
                     "and the unconstrained solution is not unique." % (n, m))

    # ---- weighting ----
    if sigma is None:
        notes.append("No sigma supplied; using unit weights. chi2_red is then "
                     "an estimate of the noise variance, not a goodness of fit.")
        A_w = A.copy()
        G_w = G.copy()
    else:
        A_w = A / sigma[:, None]
        G_w = G / sigma

    # ---- conditioning diagnostics ----
    sv = np.linalg.svd(A_w, compute_uv=False)
    tol = (np.finfo(float).eps * max(m, n) * sv[0]) if _RCOND is None \
        else (_RCOND * sv[0])
    rank = int(np.sum(sv > tol))
    if rank < n:
        notes.append("Weighted design matrix is rank deficient: rank %d of %d "
                     "columns. Basis vectors are linearly dependent over the "
                     "measured lag range." % (rank, n))

    # ---- solve ----
    if any_bounded:
        from scipy.optimize import lsq_linear
        lb = np.where(bounded, 0.0, -np.inf)
        ub = np.full(n, np.inf)
        method = "bvls" if m >= n else "trf"
        sol = lsq_linear(A_w, G_w, bounds=(lb, ub), method=method)
        d = sol.x
        method_name = "bounded"
    else:
        d, _res, _rk, _sv = np.linalg.lstsq(A_w, G_w, rcond=_RCOND)
        method_name = "lstsq"

    # ---- which parameters are free? ----
    pinned = np.zeros(n, dtype=bool)
    if any_bounded:
        scale = max(float(np.max(np.abs(d))) if np.any(d) else 0.0, 1.0)
        # BVLS can land a hair below the bound (order 1e-17 relative).  Snap
        # those to exactly zero so downstream code never sees a negative
        # amplitude; anything larger than round-off is a real failure and is
        # reported rather than quietly cleaned up.
        below = bounded & (d < 0.0)
        if np.any(below):
            worst = float(np.min(d[below]))
            if abs(worst) <= _BOUND_ACTIVE_TOL * scale:
                d = d.copy()
                d[below] = 0.0
            else:
                notes.append("Solver returned an amplitude of %.6g below the "
                             "d >= 0 bound, well beyond round-off. The result "
                             "is not trustworthy." % worst)
        pinned = bounded & (d <= _BOUND_ACTIVE_TOL * scale)
        if np.all(pinned):
            notes.append("Every constrained amplitude was pinned to zero.")
        elif np.any(pinned):
            notes.append("%d of %d amplitudes pinned at the d >= 0 boundary. "
                         "Their errors are undefined and the errors on the "
                         "remaining amplitudes are conditional on this active "
                         "set." % (int(np.sum(pinned)), n))
    free = ~pinned
    n_free = int(np.sum(free))

    # ---- residuals and chi2 ----
    G_fit = A.dot(d)
    resid = G - G_fit
    resid_w = resid if sigma is None else resid / sigma
    chi2 = float(np.dot(resid_w, resid_w))

    # ---- covariance over the free columns ----
    d_cov, cond = _free_covariance(A_w, free, notes)

    dof = m - n_free
    if dof <= 0:
        notes.append("Non-positive degrees of freedom (%d); chi2_red is "
                     "meaningless." % dof)
        chi2_red = np.nan
    else:
        chi2_red = chi2 / float(dof)

    # ---- error bars, with an explicit rescaling decision ----
    if rescale_errors == "auto":
        do_rescale = (sigma is None)
    elif rescale_errors in (True, False):
        do_rescale = bool(rescale_errors)
    else:
        raise ValueError("rescale_errors must be 'auto', True or False")

    if do_rescale and np.isfinite(chi2_red) and chi2_red > 0.0:
        d_cov = d_cov * chi2_red
        notes.append("Parameter errors rescaled by sqrt(chi2_red) = %.4g."
                     % np.sqrt(chi2_red))
    elif do_rescale:
        do_rescale = False
        notes.append("Error rescaling requested but chi2_red is not usable; "
                     "errors left unscaled.")

    with np.errstate(invalid="ignore"):
        d_err = np.sqrt(np.diag(d_cov))

    for note in notes:
        warnings.warn("fcs_linsolve: " + note, RuntimeWarning, stacklevel=2)

    return LinearFitResult(
        d=d, d_err=d_err, d_cov=d_cov,
        G_fit=G_fit, residuals=resid, residuals_weighted=resid_w,
        chi2=chi2, n_free=n_free, dof=dof, chi2_red=chi2_red,
        pinned=pinned, cond=cond, rank=rank, sigma_used=sigma,
        error_rescaled=do_rescale, method=method_name, warnings=notes,
    )


def solve_coordinate_descent(G, A, sigma=None, nonneg=False,
                             max_iter=None, min_dchi2=None, patience=None,
                             target_chi2_red=None, polish=False,
                             rescale_errors="auto"):
    """Greedy coordinate descent: free ONE amplitude per round, best move wins.

    Every round, each column is asked how much it could lower chi2 if it alone
    were allowed to move while the others stay at their CURRENT values.  The
    single best move is taken, and the round repeats.  A column already carrying
    amplitude is eligible again on any later round, so an early choice is
    revisited and corrected rather than frozen.

    Why this exists alongside solve_linear
    --------------------------------------
    On a fine grid the joint solve is not merely ill conditioned, it is
    hopeless: with 65 lag points and 602 columns it returns every column
    non-zero and a negative degrees of freedom.  Measured on that problem, this
    routine reaches a LOWER chi2 (72.99 against 73.37) with 27 non-zero columns
    instead of 601.  It never forms the full normal-equation matrix, so the
    conditioning of the whole basis never enters; each move involves one column.

    What it is not
    --------------
    It does not run to a fixed point and should not.  Measured quality is best
    in the first handful of rounds and DEGRADES with further iteration -- chi2
    per degree of freedom went 1.53, 1.26 at round 5, then 3.09 by round 500 --
    because later rounds spend degrees of freedom on columns the data do not
    support.  Stopping is therefore on the size of the improvement, not on the
    solution having stopped moving: the step size is not monotonic and has
    plateaus that a movement threshold mistakes for convergence.

    Parameters
    ----------
    G, A, sigma, nonneg, rescale_errors
        As for :func:`solve_linear`.
    max_iter : int, optional
        Hard cap on rounds.  Default _CD_MAX_ITER.
    min_dchi2 : float, optional
        A round improving chi2 by less than this counts as stalled.  Default
        _CD_MIN_DCHI2.
    patience : int, optional
        Consecutive stalled rounds required before stopping.  Default
        _CD_PATIENCE.
    target_chi2_red : float or None, optional
        Stop as soon as chi2 per degree of freedom falls to this.  Default
        _CD_TARGET_CHI2_RED.  Applied only when a sigma was supplied, since
        without one chi2_red measures the noise rather than the fit.  Pass None
        to rely on the min_dchi2 rule alone.
    polish : bool
        After selection stops, re-solve ALL selected amplitudes jointly on the
        chosen support.  The support is small by then, so this subproblem is
        well conditioned.  Off by default: it changes the amplitudes the
        selection arrived at, and that should be an explicit choice.

    Returns
    -------
    LinearFitResult, with the extra attributes ``n_iter``, ``converged``,
    ``chi2_history`` and ``selection_order``.
    """
    notes = []

    G = _as_vector(G, "G")
    A = _as_matrix(A, "A")
    if sigma is not None:
        sigma = _as_vector(sigma, "sigma")
        if np.any(sigma <= 0.0):
            raise ValueError("sigma must be strictly positive; got min %.6g"
                             % float(np.min(sigma)))
    m = G.size
    if A.shape[0] != m:
        raise ValueError("A has %d rows but G has %d elements"
                         % (A.shape[0], m))
    if sigma is not None and sigma.size != m:
        raise ValueError("sigma has %d elements but G has %d"
                         % (sigma.size, m))
    n = A.shape[1]

    max_iter = _CD_MAX_ITER if max_iter is None else int(max_iter)
    min_dchi2 = _CD_MIN_DCHI2 if min_dchi2 is None else float(min_dchi2)
    patience = _CD_PATIENCE if patience is None else int(patience)
    if target_chi2_red is None:
        target_chi2_red = _CD_TARGET_CHI2_RED
    if target_chi2_red is not None:
        target_chi2_red = float(target_chi2_red)
        if target_chi2_red <= 0.0:
            raise ValueError("target_chi2_red must be > 0")
    if max_iter < 1:
        raise ValueError("max_iter must be >= 1")
    if min_dchi2 < 0.0:
        raise ValueError("min_dchi2 must be >= 0")
    if patience < 1:
        raise ValueError("patience must be >= 1")

    if isinstance(nonneg, (bool, np.bool_)):
        bounded = np.full(n, bool(nonneg))
    else:
        bounded = np.asarray(nonneg, dtype=bool).ravel()
        if bounded.size != n:
            raise ValueError("nonneg mask has %d elements but A has %d columns"
                             % (bounded.size, n))

    if sigma is None:
        notes.append("No sigma supplied; using unit weights. chi2_red is then "
                     "an estimate of the noise variance, not a goodness of fit.")
        A_w = A.copy()
        G_w = G.copy()
    else:
        A_w = A / sigma[:, None]
        G_w = G / sigma

    # A column with no weighted length is invisible to the data and would
    # accept unlimited amplitude at no cost in chi2.  Exclude it from selection
    # and say so, rather than letting it collect a meaningless amplitude.
    col2 = np.einsum("ij,ij->j", A_w, A_w)
    usable = col2 > 0.0
    if not np.all(usable):
        notes.append("%d of %d columns have zero weighted length over the "
                     "measured lag range and were excluded from selection."
                     % (int(np.sum(~usable)), n))
    if not np.any(usable):
        raise ValueError("no column has any weighted length; check A and sigma")
    col2_safe = np.where(usable, col2, 1.0)

    # ---- the loop ----
    d = np.zeros(n)
    r = A_w.dot(d) - G_w
    chi2 = float(r.dot(r))
    chi2_history = [chi2]
    selection_order = []
    stalled = 0
    converged = False
    n_iter = 0

    for n_iter in range(1, max_iter + 1):
        num = A_w.T.dot(r)                       # gradient direction per column
        trial = d - num / col2_safe              # 1-D minimiser for each column
        np.maximum(trial, 0.0, out=trial, where=bounded)
        move = trial - d
        # Exact change in chi2 from moving column j alone, no refit needed.
        dchi2 = 2.0 * move * num + (move ** 2) * col2
        dchi2[~usable] = 0.0

        j = int(np.argmin(dchi2))
        gain = -float(dchi2[j])
        if gain <= 0.0:
            converged = True
            notes.append("No column can lower chi2 further; stopped after "
                         "%d rounds." % n_iter)
            break

        delta = float(move[j])
        d[j] = trial[j]
        r = r + delta * A_w[:, j]
        chi2 = float(r.dot(r))
        chi2_history.append(chi2)
        if j not in selection_order:
            selection_order.append(j)

        if target_chi2_red is not None and sigma is not None:
            n_now = int(np.sum(~(bounded & (d <= 0.0))))
            dof_now = m - n_now
            if dof_now > 0 and chi2 / float(dof_now) <= target_chi2_red:
                converged = True
                notes.append("Stopped at chi2_red = %.4g, at or below the "
                             "target of %.4g, after %d rounds."
                             % (chi2 / float(dof_now), target_chi2_red, n_iter))
                break

        stalled = stalled + 1 if gain < min_dchi2 else 0
        if stalled >= patience:
            converged = True
            break

    if not converged:
        notes.append("Hit the %d-round cap without stalling; the result is "
                     "where the search had reached, not where it settles."
                     % max_iter)

    # ---- optional joint re-solve on the selected support ----
    if polish:
        support = np.where(d != 0.0)[0]
        if support.size:
            from scipy.optimize import lsq_linear
            lb = np.where(bounded[support], 0.0, -np.inf)
            sub = A_w[:, support]
            sol = lsq_linear(sub, G_w, bounds=(lb, np.full(support.size, np.inf)),
                             method="bvls" if m >= support.size else "trf")
            d = np.zeros(n)
            d[support] = sol.x
            r = A_w.dot(d) - G_w
            chi2 = float(r.dot(r))
            chi2_history.append(chi2)
            notes.append("Amplitudes re-solved jointly on the %d selected "
                         "columns after selection stopped." % support.size)

    # ---- book-keeping, matching solve_linear ----
    scale = max(float(np.max(np.abs(d))) if np.any(d) else 0.0, 1.0)
    below = bounded & (d < 0.0)
    if np.any(below):
        d[below] = 0.0
    pinned = bounded & (d <= _BOUND_ACTIVE_TOL * scale)
    free = ~pinned
    n_free = int(np.sum(free))

    G_fit = A.dot(d)
    resid = G - G_fit
    resid_w = resid if sigma is None else resid / sigma
    chi2 = float(np.dot(resid_w, resid_w))

    sv = np.linalg.svd(A_w, compute_uv=False)
    tol = (np.finfo(float).eps * max(m, n) * sv[0]) if _RCOND is None \
        else (_RCOND * sv[0])
    rank = int(np.sum(sv > tol))

    d_cov, cond = _free_covariance(A_w, free, notes)

    dof = m - n_free
    if dof <= 0:
        notes.append("Non-positive degrees of freedom (%d); chi2_red is "
                     "meaningless." % dof)
        chi2_red = np.nan
    else:
        chi2_red = chi2 / float(dof)

    if rescale_errors == "auto":
        do_rescale = (sigma is None)
    elif rescale_errors in (True, False):
        do_rescale = bool(rescale_errors)
    else:
        raise ValueError("rescale_errors must be 'auto', True or False")
    if do_rescale and np.isfinite(chi2_red) and chi2_red > 0.0:
        d_cov = d_cov * chi2_red
        notes.append("Parameter errors rescaled by sqrt(chi2_red) = %.4g."
                     % np.sqrt(chi2_red))
    elif do_rescale:
        do_rescale = False

    with np.errstate(invalid="ignore"):
        d_err = np.sqrt(np.diag(d_cov))

    for note in notes:
        warnings.warn("fcs_linsolve: " + note, RuntimeWarning, stacklevel=2)

    return LinearFitResult(
        d=d, d_err=d_err, d_cov=d_cov,
        G_fit=G_fit, residuals=resid, residuals_weighted=resid_w,
        chi2=chi2, n_free=n_free, dof=dof, chi2_red=chi2_red,
        pinned=pinned, cond=cond, rank=rank, sigma_used=sigma,
        error_rescaled=do_rescale, method="coordinate_descent",
        warnings=notes,
        n_iter=n_iter, converged=converged,
        chi2_history=np.asarray(chi2_history),
        selection_order=np.asarray(selection_order, dtype=int),
    )


def apply_mask(G, A, sigma=None, mask=None):
    """Restrict G, A and sigma to the rows where mask is True.

    Fit-window selection lives here so it is not duplicated at every call site.
    """
    G = _as_vector(G, "G")
    A = _as_matrix(A, "A")
    if mask is None:
        return G, A, sigma
    mask = np.asarray(mask, dtype=bool).ravel()
    if mask.size != G.size:
        raise ValueError("mask has %d elements but G has %d"
                         % (mask.size, G.size))
    if not np.any(mask):
        raise ValueError("mask selects no points")
    return (G[mask], A[mask, :],
            None if sigma is None else _as_vector(sigma, "sigma")[mask])


# --------------------------------------------------------------------------
# Basis generation
# --------------------------------------------------------------------------
#
# Port of the MATLAB `basismodel(taufactor)`.  One basis vector is the shape of
# the curve produced by a single species whose characteristic decay time is
# `taufactor` times a hardcoded reference time.  The two-term functional form
# and every numerical constant below are taken verbatim from that script.
#
# Two deliberate departures from the MATLAB:
#   1. t is supplied by the caller instead of a hardcoded logspace, so the basis
#      is evaluated on the measured lag times.
#   2. The additive `offset` term is NOT baked into each column.  A baseline is
#      fitted instead as a separate constant column (the tau_D -> infinity
#      column), so the reconstructed curve has one baseline rather than one per
#      basis vector scaled by its amplitude.

_BASIS_REF_TAU = 5.416343166e-05      # MATLAB fitreftau
_BASIS_US_SCALE = 1.0e-06             # numerator of MATLAB usconversion
_BASIS_TAU1 = 2.916511434e-05         # MATLAB t1 coefficient
_BASIS_TAU2 = 1.128422659e-04         # MATLAB t2 coefficient
_BASIS_F1 = 0.6820770377              # MATLAB f1 (f2 = 1 - f1)
_BASIS_K = 19.90748158                # MATLAB k
_BASIS_G0 = 1.0                       # MATLAB g0, forced to 1 so the column
                                      # carries shape only and the amplitude
                                      # lives entirely in d
_BASIS_MATLAB_OFFSET = -9.573317357e-05   # MATLAB offset, retained for the
                                          # reference check only; not used in
                                          # the basis columns

# Value of the baseline column.  Ones by default: the basis columns are order
# unity at short lag, so a column of ones is on the same scale and costs
# nothing in conditioning.
_BASELINE_VALUE = 1.0


def basis_component_times(taufactors):
    """Return the two component decay times, in seconds, for each taufactor.

    At taufactor = 1 these are roughly 5.385e-07 s and 2.083e-06 s.
    """
    tf = np.asarray(taufactors, dtype=float).ravel()
    usconversion = _BASIS_US_SCALE / _BASIS_REF_TAU
    return tf * usconversion * _BASIS_TAU1, tf * usconversion * _BASIS_TAU2


def build_basis(t, taufactors, t_units=None, include_baseline=True):
    """Populate the columns of A from a list of taufactors.

    Parameters
    ----------
    t : array_like, shape (m,)
        Lag times, one per element of G, in the same order.
    taufactors : array_like, shape (n_tau,)
        Column j is basismodel(taufactors[j]) evaluated at t.
    t_units : {'s', 'ms', 'us', 'ns'}, optional
        Units of t.  Defaults to the module constant _BASIS_T_UNITS.  The basis
        constants are in seconds; t is converted on the way in.
    include_baseline : bool
        Append a constant column as the LAST column, fitting a free baseline.

    Returns
    -------
    A : (m, n_tau + 1) ndarray if include_baseline else (m, n_tau)
    """
    t = _as_vector(t, "t")
    if np.any(t < 0.0):
        raise ValueError("t contains negative lag times")

    units = _BASIS_T_UNITS if t_units is None else str(t_units)
    if units not in _T_UNIT_TO_SECONDS:
        raise ValueError("unknown t_units %r; expected one of %s"
                         % (units, sorted(_T_UNIT_TO_SECONDS)))
    t_sec = t * _T_UNIT_TO_SECONDS[units]

    tf = np.asarray(taufactors, dtype=float).ravel()
    if tf.size == 0:
        raise ValueError("taufactors is empty")
    if not np.all(np.isfinite(tf)):
        raise ValueError("taufactors contains non-finite values")
    if np.any(tf <= 0.0):
        raise ValueError("taufactors must be strictly positive; got min %.6g"
                         % float(np.min(tf)))

    t1, t2 = basis_component_times(tf)          # each (n_tau,)
    ksq = _BASIS_K ** 2

    r1 = t_sec[:, None] / t1[None, :]           # r1[i, j] = t[i] / t1[j]
    r2 = t_sec[:, None] / t2[None, :]

    d1 = (1.0 + r1) ** (-1.0) * (1.0 + r1 / ksq) ** (-0.5)
    d2 = (1.0 + r2) ** (-1.0) * (1.0 + r2 / ksq) ** (-0.5)

    f1 = _BASIS_F1
    A = _BASIS_G0 * (f1 * d1 + (1.0 - f1) * d2)

    if include_baseline:
        A = np.hstack([A, np.full((t.size, 1), _BASELINE_VALUE)])

    if not np.all(np.isfinite(A)):
        raise ValueError("basis evaluation produced non-finite values; check t "
                         "and taufactors")
    return A


def basis_vector(t, taufactor, t_units=None):
    """Evaluate a single basis curve at lag times t.  No baseline column."""
    return build_basis(t, [taufactor], t_units=t_units,
                       include_baseline=False)[:, 0]


def log_taufactor_grid(tf_min=None, tf_max=None, per_decade=None):
    """Log-spaced taufactor grid, defaulting to the module constants.

    Spacing is set per decade rather than by a total count so that widening the
    span does not silently coarsen the grid.
    """
    tf_min = _TAUFACTOR_MIN if tf_min is None else float(tf_min)
    tf_max = _TAUFACTOR_MAX if tf_max is None else float(tf_max)
    per_decade = _TAUFACTOR_PER_DECADE if per_decade is None else int(per_decade)
    if tf_min <= 0.0 or tf_max <= 0.0:
        raise ValueError("taufactor bounds must be strictly positive")
    if tf_max <= tf_min:
        raise ValueError("tf_max must exceed tf_min")
    if per_decade < 1:
        raise ValueError("per_decade must be >= 1")
    decades = np.log10(tf_max / tf_min)
    n = int(round(per_decade * decades)) + 1
    return np.logspace(np.log10(tf_min), np.log10(tf_max), n)


def basis_nonneg_mask(n_tau, include_baseline=True):
    """Bounds mask constraining the basis amplitudes but not the baseline.

    The MATLAB reference offset is negative, so a baseline forced to be >= 0
    would be pinned at zero and the bias would be absorbed by the longest-time
    basis vectors.  Pass the result of this as solve_linear's `nonneg`.
    """
    mask = np.ones(int(n_tau) + (1 if include_baseline else 0), dtype=bool)
    if include_baseline:
        mask[-1] = False
    return mask


def baseline_index(n_tau):
    """Index of the baseline column in a matrix built with include_baseline."""
    return int(n_tau)


# --------------------------------------------------------------------------
# Self-test
# --------------------------------------------------------------------------

def _matlab_reference(taufactor, tvec):
    """Line-by-line transcription of basismodel, for verification only."""
    fitreftau = 0.00005416343166
    usconversion = 1e-06 / fitreftau
    t1 = taufactor * usconversion * 0.00002916511434
    f1 = 0.6820770377
    f2 = 1 - f1
    t2 = taufactor * usconversion * 0.0001128422659
    k = 19.90748158
    offset = -0.00009573317357
    g0 = 1
    d1 = ((1 + (tvec / t1)) ** (-1)) * ((1 + (tvec / (t1 * k ** 2))) ** (-0.5))
    d2 = ((1 + (tvec / t2)) ** (-1)) * ((1 + (tvec / (t2 * k ** 2))) ** (-0.5))
    return offset + g0 * (f1 * d1 + f2 * d2)


def _self_test():
    rng = np.random.RandomState(20260828)

    print("=" * 70)
    print("TEST 1: basis columns match MATLAB reference (minus its offset)")
    t_ml = np.logspace(-6, -1, 100)
    tf_list = [0.1, 1.0, 7.3, 250.0]
    A_b = build_basis(t_ml, tf_list, t_units="s", include_baseline=False)
    for j, tf in enumerate(tf_list):
        ref = _matlab_reference(tf, t_ml) - _BASIS_MATLAB_OFFSET
        err = np.max(np.abs(A_b[:, j] - ref))
        print("  taufactor %9.4g  max abs deviation %.3e" % (tf, err))
        assert err < 1e-14
    print("  taufactor=1 component times: t1=%.6e s  t2=%.6e s"
          % tuple(float(x[0]) for x in basis_component_times([1.0])))

    print("=" * 70)
    print("TEST 2: unit conversion")
    A_s = build_basis(t_ml, [5.0], t_units="s", include_baseline=False)
    A_ms = build_basis(t_ml * 1e3, [5.0], t_units="ms", include_baseline=False)
    assert np.allclose(A_s, A_ms, rtol=0, atol=1e-15)
    print("  s and ms paths agree to %.2e" % float(np.max(np.abs(A_s - A_ms))))

    print("=" * 70)
    print("TEST 3: baseline column and bounds mask")
    A_bl = build_basis(t_ml, tf_list, t_units="s", include_baseline=True)
    assert A_bl.shape == (100, 5)
    assert np.all(A_bl[:, -1] == _BASELINE_VALUE)
    assert baseline_index(len(tf_list)) == 4
    assert basis_nonneg_mask(len(tf_list)).tolist() == [1, 1, 1, 1, 0]
    print("  shape %s, baseline at index %d, unconstrained in mask"
          % (A_bl.shape, baseline_index(len(tf_list))))

    print("=" * 70)
    print("TEST 4: default grid")
    grid = log_taufactor_grid()
    decades = np.log10(_TAUFACTOR_MAX / _TAUFACTOR_MIN)
    print("  %d taufactors from %.4g to %.4g (%d per decade)"
          % (grid.size, grid[0], grid[-1], _TAUFACTOR_PER_DECADE))
    # Derived from the constants rather than hardcoded, so a preference change
    # does not break the self-test.
    assert grid.size == int(round(_TAUFACTOR_PER_DECADE * decades)) + 1
    assert np.isclose(grid[0], _TAUFACTOR_MIN)
    assert np.isclose(grid[-1], _TAUFACTOR_MAX)

    print("=" * 70)
    print("TEST 5: recovery with a free baseline")
    t5 = np.logspace(-6, -1, 128)
    tf5 = np.array([2.0, 20.0, 400.0])
    A5 = build_basis(t5, tf5, t_units="s")
    d5_true = np.array([0.35, 0.20, 0.06, -0.004])   # last is the baseline
    G5c = A5.dot(d5_true)
    s5 = 0.002 * (1.0 + np.abs(G5c))
    G5 = G5c + s5 * rng.randn(t5.size)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r5 = solve_linear(G5, A5, sigma=s5, nonneg=basis_nonneg_mask(tf5.size))
    print(r5.summary())
    dev = np.abs(r5.d - d5_true) / r5.d_err
    print("  deviation from truth, in sigma: %s"
          % np.array2string(dev, precision=2))
    assert np.all(dev < 4.0)
    assert r5.d[-1] < 0.0, "baseline should be free to go negative"
    assert 0.4 < r5.chi2_red < 2.0
    assert r5.n_free == 4 and r5.dof == 124

    print("=" * 70)
    print("TEST 6: unweighted path engages auto rescaling")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r6 = solve_linear(G5, A5, sigma=None)
    assert r6.error_rescaled and r6.sigma_used is None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r6b = solve_linear(G5, A5, sigma=s5, rescale_errors=True)
    assert r6b.error_rescaled
    print("  auto-on-unweighted and explicit override both behave")

    print("=" * 70)
    print("TEST 7: fine grid, constraint only, no regularisation")
    grid7 = log_taufactor_grid(1.0, 1.0e4, 20)
    A7 = build_basis(t5, grid7, t_units="s")
    truth = np.zeros(grid7.size + 1)
    truth[np.argmin(np.abs(grid7 - 30.0))] = 0.40
    truth[-1] = -0.003
    G7c = A7.dot(truth)
    s7 = 0.002 * (1.0 + np.abs(G7c))
    G7 = G7c + s7 * rng.randn(t5.size)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r7 = solve_linear(G7, A7, sigma=s7, nonneg=basis_nonneg_mask(grid7.size))
    amp = r7.d[:grid7.size]
    live = np.where(amp > 0.01 * amp.max())[0]
    print("  %d of %d columns non-zero, at taufactors %s"
          % (live.size, grid7.size,
             np.array2string(grid7[live], precision=2)))
    print("  amplitude sum %.4f (truth 0.40), baseline %+.5f (truth -0.00300)"
          % (amp.sum(), r7.d[-1]))
    print("  chi2_red %.3f, cond %.4g" % (r7.chi2_red, r7.cond))
    assert np.all(amp >= 0.0)
    assert 0.4 < r7.chi2_red < 2.0

    print("=" * 70)
    print("TEST 8: coordinate descent on the regime the joint solve cannot handle")
    t8 = np.logspace(-5, -1, 65)                 # 65 lag points
    g8 = log_taufactor_grid(1e-2, 1e4, 100)      # 601 basis vectors
    A8 = build_basis(t8, g8, t_units="s")
    tr8 = np.zeros(g8.size + 1)
    tr8[int(np.argmin(np.abs(g8 - 30.0)))] = 0.05
    tr8[-1] = -0.0005
    s8 = 0.0008 * (1.0 + np.abs(A8.dot(tr8)))
    G8 = A8.dot(tr8) + s8 * rng.randn(t8.size)
    mask8 = basis_nonneg_mask(g8.size)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        joint = solve_linear(G8, A8, sigma=s8, nonneg=mask8)
        cdr = solve_coordinate_descent(G8, A8, sigma=s8, nonneg=mask8)
    print("  joint solve : %3d non-zero, dof %5d, chi2 %8.3f, chi2_red %s"
          % (int(np.sum(joint.d[:g8.size] > 0)), joint.dof, joint.chi2,
             "nan" if not np.isfinite(joint.chi2_red) else "%.3f" % joint.chi2_red))
    print("  coord descent: %3d non-zero, dof %5d, chi2 %8.3f, chi2_red %.3f"
          % (int(np.sum(cdr.d[:g8.size] > 0)), cdr.dof, cdr.chi2, cdr.chi2_red))
    print("  rounds %d, converged %s, columns selected %d"
          % (cdr.n_iter, cdr.converged, cdr.selection_order.size))
    assert cdr.n_free < joint.n_free, "CD must be sparser than the joint solve"
    assert cdr.dof > 0, "CD must leave positive degrees of freedom"
    assert np.all(cdr.d[:g8.size] >= 0.0)
    assert cdr.chi2_history[-1] <= cdr.chi2_history[0]
    assert cdr.chi2_history.size == cdr.n_iter + 1
    peak = int(np.argmax(cdr.d[:g8.size]))
    print("  largest amplitude at taufactor %.4g (truth 30)" % g8[peak])
    assert 3.0 < g8[peak] < 300.0, "peak should be within a decade of the truth"

    print("=" * 70)
    print("TEST 9: stopping rule, patience, and the polish option")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        loose = solve_coordinate_descent(G8, A8, sigma=s8, nonneg=mask8,
                                         min_dchi2=50.0)
        tight = solve_coordinate_descent(G8, A8, sigma=s8, nonneg=mask8,
                                         min_dchi2=0.01, max_iter=400)
        capped = solve_coordinate_descent(G8, A8, sigma=s8, nonneg=mask8,
                                          min_dchi2=0.0, max_iter=5)
        pol = solve_coordinate_descent(G8, A8, sigma=s8, nonneg=mask8,
                                       polish=True)
    print("  min_dchi2=50   -> %2d rounds, %2d non-zero" % (loose.n_iter, loose.n_free))
    print("  min_dchi2=0.01 -> %2d rounds, %2d non-zero" % (tight.n_iter, tight.n_free))
    assert loose.n_iter < tight.n_iter, "a looser threshold must stop sooner"
    assert capped.n_iter == 5 and not capped.converged, "the cap must be honoured"
    assert pol.chi2 <= cdr.chi2 + 1e-9, "polishing must not worsen chi2"
    print("  cap honoured (%d rounds, converged=%s); polish chi2 %.3f vs %.3f"
          % (capped.n_iter, capped.converged, pol.chi2, cdr.chi2))
    for bad in (
        lambda: solve_coordinate_descent(G8, A8, sigma=s8, max_iter=0),
        lambda: solve_coordinate_descent(G8, A8, sigma=s8, patience=0),
        lambda: solve_coordinate_descent(G8, A8, sigma=s8, min_dchi2=-1.0),
        lambda: solve_coordinate_descent(G8, A8, sigma=s8,
                                         nonneg=np.ones(3, dtype=bool)),
    ):
        try:
            bad()
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError")
    print("  invalid settings rejected")

    print("=" * 70)
    print("TEST 10: rank deficiency is reported")
    A_dup = np.hstack([A5, A5[:, [0]]])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r8 = solve_linear(G5, A_dup, sigma=s5)
    print("  rank %d of %d columns" % (r8.rank, A_dup.shape[1]))
    assert r8.rank == 4

    print("=" * 70)
    print("TEST 11: masking and input validation")
    Gm, Am, sm = apply_mask(G5, A5, s5, t5 > 1e-5)
    assert Gm.size == Am.shape[0] == sm.size == int(np.sum(t5 > 1e-5))
    for bad in (
        lambda: solve_linear(G5, A5[:-1, :], sigma=s5),
        lambda: solve_linear(G5, A5, sigma=np.zeros(t5.size)),
        lambda: solve_linear(G5, A5, rescale_errors="maybe"),
        lambda: solve_linear(G5, A5, nonneg=np.ones(2, dtype=bool)),
        lambda: build_basis(t5, []),
        lambda: build_basis(t5, [1.0, -2.0]),
        lambda: build_basis([-1e-6, 1e-6], [1.0]),
        lambda: build_basis(t5, [1.0], t_units="fortnights"),
        lambda: log_taufactor_grid(10.0, 1.0),
        lambda: apply_mask(G5, A5, s5, np.zeros(t5.size, dtype=bool)),
    ):
        try:
            bad()
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError")
    print("  all rejected")

    print("=" * 70)
    print("ALL SELF-TESTS PASSED")


if __name__ == "__main__":
    _self_test()
