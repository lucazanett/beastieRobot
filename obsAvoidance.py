"""
Geometric constraints from Mousavi et al. (ICEE 2013):
Linearized (tangent) half-space constraints for probabilistic obstacle
avoidance using expanded error ellipses.

This module implements the steps:
1) Compute an error ellipse from a 2x2 covariance Σ and collision probability δ
2) Expand the ellipse axes by vehicle & obstacle radii
3) Build the tangent half-space at the intersection of the ellipse and the line
   from obstacle center to the (predicted) vehicle position
4) Return linear constraints of the form A x <= b to be fed into an optimizer

Notation (matching the paper):
- δ: collision probability level used to define the ellipse level set
- k = -2 ln(δ)
- Σ = T diag(λ1, λ2) T^T (eigendecomposition, λ1 ≥ λ2 ≥ 0)
- a = sqrt(k λ1), b = sqrt(k λ2) are the semi-axes
- a_exp = a + r_vehicle + r_obstacle; b_exp similarly
- Ellipse shape matrix: Q = T diag(1/a_exp^2, 1/b_exp^2) T^T
- Tangent at p* on the ellipse has outward normal n = Q (p* - c)
  and the half-space is chosen so the reference point is feasible.

Author: ChatGPT — Python/Numpy implementation
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple
import numpy as np

ArrayLike = Sequence[float]


@dataclass
class ObstacleState:
    """Predicted obstacle state at a time step.

    Attributes
    ----------
    center : np.ndarray
        2D vector (x, y) for the obstacle's mean position c.
    cov : np.ndarray
        2x2 covariance matrix Σ for obstacle position uncertainty.
    r_obs : float
        Obstacle radius.
    """
    center: np.ndarray
    cov: np.ndarray
    r_obs: float


def _ensure_array(x: ArrayLike, shape: Tuple[int, ...]) -> np.ndarray:
    arr = np.asarray(x, dtype=float).reshape(shape)
    return arr


def error_ellipse_axes_from_cov(
    Sigma: np.ndarray,
    delta: float,
) -> Tuple[np.ndarray, float, float, np.ndarray]:
    """Compute eigenvectors and semi-axes (a, b) of the 2D error ellipse.

    Parameters
    ----------
    Sigma : (2,2) ndarray
        Covariance matrix of the Gaussian position.
    delta : float
        Collision probability level δ in (0,1). The ellipse level is set by
        k = -2 ln(δ).

    Returns
    -------
    T : (2,2) ndarray
        Columns are eigenvectors (principal directions).
    a : float
        Semi-major axis length (>= b).
    b : float
        Semi-minor axis length.
    lambdas : (2,) ndarray
        Sorted eigenvalues [λ1, λ2] (λ1 >= λ2).
    """
    Sigma = _ensure_array(Sigma, (2, 2))
    if not np.isfinite(Sigma).all():
        raise ValueError("Sigma contains non-finite values")

    # Regularize if needed to ensure SPD
    eps = 1e-12
    Sigma = 0.5 * (Sigma + Sigma.T)
    w, V = np.linalg.eigh(Sigma + eps * np.eye(2))
    # Sort descending by eigenvalue so λ1 >= λ2 and first column is major axis
    order = np.argsort(w)[::-1]
    lambdas = w[order]
    T = V[:, order]

    if np.any(lambdas < 0):
        raise ValueError("Sigma is not positive semidefinite after regularization")

    if not (0.0 < delta < 1.0):
        raise ValueError("delta must be in (0,1)")

    k = -2.0 * np.log(delta)
    # Semi-axes for the contour x^T Σ^{-1} x = k
    a = float(np.sqrt(k * lambdas[0]))
    b = float(np.sqrt(k * lambdas[1]))
    return T, a, b, lambdas


def expand_axes(a: float, b: float, r_vehicle: float, r_obstacle: float) -> Tuple[float, float]:
    """Inflate the ellipse semi-axes by vehicle and obstacle radii.

    Returns
    -------
    a_exp, b_exp : floats
    """
    a_exp = float(a + r_vehicle + r_obstacle)
    b_exp = float(b + r_vehicle + r_obstacle)
    return a_exp, b_exp


def ellipse_shape_matrix(T: np.ndarray, a: float, b: float) -> np.ndarray:
    """Return the shape matrix Q s.t. (x-c)^T Q (x-c) = 1 is the ellipse.

    Q = T diag(1/a^2, 1/b^2) T^T
    """
    inv_axes2 = np.diag([1.0 / (a * a), 1.0 / (b * b)])
    Q = T @ inv_axes2 @ T.T
    return Q


def tangent_halfspace(
    x_ref: ArrayLike,
    center: ArrayLike,
    Q: np.ndarray,
    ensure_satisfied_by: ArrayLike | None = None,
) -> Tuple[np.ndarray, float, np.ndarray]:
    """Build A, b for the tangent half-space at the ellipse point toward x_ref.

    Given ellipse (x-c)^T Q (x-c) = 1 and a reference point x_ref, we:
    - Shoot a ray from c to x_ref, intersect with the ellipse at p*.
    - Compute outward normal n = Q (p* - c).
    - Choose inequality orientation so a chosen point is feasible.

    Parameters
    ----------
    x_ref : (2,) array-like
        The reference position (typically current/predicted vehicle position).
    center : (2,) array-like
        Ellipse center c.
    Q : (2,2) ndarray
        Ellipse shape matrix.
    ensure_satisfied_by : (2,) array-like or None
        If provided, orient inequality so this point satisfies A x <= b.
        Defaults to x_ref if None.

    Returns
    -------
    A : (1,2) ndarray
        Row vector for the linear constraint A x <= b.
    b : float
        Right-hand side scalar.
    p_star : (2,) ndarray
        The ellipse point where the tangent is taken.
    """
    x_ref = _ensure_array(x_ref, (2,))
    c = _ensure_array(center, (2,))
    Q = _ensure_array(Q, (2, 2))

    d = x_ref - c
    denom = float(d.T @ Q @ d)

    # Handle degenerate case: reference at center or nearly so
    if denom <= 1e-14:
        # Pick direction of largest curvature (major axis in transformed space)
        # Use eigenvectors of Q (same as T but inverted axes). Use first column of eigenvectors.
        w, V = np.linalg.eigh(Q)
        # Smallest eigenvalue of Q corresponds to largest axis of ellipse, but we only need a direction
        v_dir = V[:, 0]
        d = v_dir
        denom = float(d.T @ Q @ d)
        if denom <= 1e-14:
            raise RuntimeError("Cannot find a valid direction for tangent construction.")

    t = 1.0 / np.sqrt(denom)
    p_star = c + t * d
    n = Q @ (p_star - c)  # outward normal

    A = n.reshape(1, 2)
    b = float(n @ p_star)

    test_point = _ensure_array(ensure_satisfied_by, (2,)) if ensure_satisfied_by is not None else x_ref
    # If test point does NOT satisfy A x <= b, flip the inequality
    if float(A @ test_point) > b:
        A = -A
        b = -b

    return A, b, p_star


def obstacle_halfspace_from_gaussian(
    x_vehicle: ArrayLike,
    center: ArrayLike,
    Sigma: np.ndarray,
    delta: float,
    r_vehicle: float,
    r_obstacle: float,
) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray, Tuple[float, float]]:
    """Convenience wrapper: Σ, δ, radii -> (A,b) for one obstacle.

    Returns
    -------
    A, b : define A x <= b
    Q : ellipse shape matrix used
    T : eigenvectors of Σ
    (a_exp, b_exp) : expanded semi-axes
    """
    T, a, b_axis, _ = error_ellipse_axes_from_cov(Sigma, delta)
    a_exp, b_exp = expand_axes(a, b_axis, r_vehicle, r_obstacle)
    Q = ellipse_shape_matrix(T, a_exp, b_exp)
    A, b, p_star = tangent_halfspace(x_vehicle, center, Q)
    return A, b, Q, T, (a_exp, b_exp)


def batch_constraints(
    x_vehicle_seq: Iterable[ArrayLike],
    obstacles_seq: Iterable[Iterable[ObstacleState]],
    Sigmas_seq: Iterable[Iterable[np.ndarray]],
    deltas_seq: Iterable[Iterable[float]] | float,
    r_vehicle: float,
) -> Tuple[List[np.ndarray], List[float]]:
    """Build constraints over a horizon and multiple obstacles.

    Parameters
    ----------
    x_vehicle_seq : iterable over time of (2,) vehicle positions (reference guesses)
    obstacles_seq : iterable over time of iterable of ObstacleState (centers & radii)
    Sigmas_seq : iterable over time of iterable of 2x2 covariance matrices per obstacle
    deltas_seq : iterable over time of iterable of δ per obstacle, or a scalar δ broadcast to all
    r_vehicle : vehicle radius

    Returns
    -------
    A_list : list of (1,2) arrays
    b_list : list of scalars
    """
    A_list: List[np.ndarray] = []
    b_list: List[float] = []

    # Broadcast delta if a scalar is provided
    def _delta_for_time(t_idx: int, num_obs: int) -> List[float]:
        if isinstance(deltas_seq, (float, int)):
            return [float(deltas_seq)] * num_obs
        return [float(d) for d in deltas_seq[t_idx]]

    for t_idx, (x_v, obs_list, Sigma_list) in enumerate(zip(x_vehicle_seq, obstacles_seq, Sigmas_seq)):
        deltas_t = _delta_for_time(t_idx, len(list(obs_list)))
        for (obs, Sigma, delt) in zip(obs_list, Sigma_list, deltas_t):
            A, b, _, _, _ = obstacle_halfspace_from_gaussian(
                x_vehicle=x_v,
                center=obs.center,
                Sigma=Sigma,
                delta=delt,
                r_vehicle=r_vehicle,
                r_obstacle=obs.r_obs,
            )
            A_list.append(A)
            b_list.append(b)
    return A_list, b_list


# ------------------------------
# Example usage / quick test
# ------------------------------
if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)

    # Example obstacle
    c = np.array([1.0, 0.0])
    Sigma = np.array([[0.5, 0.1], [0.1, 0.2]])  # covariance
    delta = 0.7                   # as in the paper's example
    r_vehicle = 0.3
    r_obstacle = 0.2

    # Vehicle reference position
    x_v = np.array([1.0, -2.0])

    A, b, Q, T, axes = obstacle_halfspace_from_gaussian(
        x_vehicle=x_v,
        center=c,
        Sigma=Sigma,
        delta=delta,
        r_vehicle=r_vehicle,
        r_obstacle=r_obstacle,
    )

    print("Constraint A x <= b:")
    print("A =", A)
    print("b =", b)
    print("Expanded semi-axes (a_exp, b_exp) =", axes)

    # Check that x_v satisfies the inequality
    lhs = float(A @ x_v)
    print("A x_v =", lhs, "<=", b, "?", lhs <= b + 1e-9)

    # A point slightly inside along the ray should violate the constraint
    # Move from p* towards center c to an interior point
    # (we can reconstruct p* from the tangent computation again)
    _, _, p_star = tangent_halfspace(x_v, c, Q)
    interior = p_star * 0.9  # towards the center
    print("A interior =", float(A @ interior), "<=", b)

    # Optional quick plot if matplotlib is available
    try:
        import matplotlib.pyplot as plt

        # Ellipse contour
        theta = np.linspace(0, 2 * np.pi, 200)
        a_exp, b_exp = axes
        ellipse_pts = (T @ np.vstack([a_exp * np.cos(theta), b_exp * np.sin(theta)])).T + c

        # Tangent line sampling
        n = A.reshape(2)
        # Points x with n^T x = b
        # Parametrize line as x = x0 + t * t_dir where t_dir is perpendicular to n
        if abs(n[1]) > 1e-12:
            x0 = np.array([0.0, b / n[1]])
        else:
            x0 = np.array([b / n[0], 0.0])
        t_dir = np.array([-n[1], n[0]])
        line_pts = np.array([x0 + s * t_dir for s in np.linspace(-50, 50, 100)])

        plt.figure()
        plt.plot(ellipse_pts[:, 0], ellipse_pts[:, 1], label="Expanded ellipse")
        plt.plot([c[0]], [c[1]], "o", label="Obstacle center")
        plt.plot([x_v[0]], [x_v[1]], "x", label="Vehicle ref")
        plt.plot(line_pts[:, 0], line_pts[:, 1], "--", label="Tangent")
        plt.axis("equal")
        plt.legend()
        plt.title("Geometric constraint: tangent half-space")
        plt.show()
    except Exception as e:
        pass
