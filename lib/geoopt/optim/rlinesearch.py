"""Riemannian Line Search.

This module implements line search on Riemannian manifolds using geoopt.
This module uses the same syntax as a Torch optimizer
"""

from scipy.optimize.linesearch import scalar_search_wolfe2, scalar_search_armijo
import warnings
import torch
from .mixin import OptimMixin
from ..tensor import ManifoldParameter, ManifoldTensor
from ..manifolds import Euclidean


__all__ = ["RiemannianLineSearch"]


class LineSearchWarning(RuntimeWarning):
    pass


class RiemannianLineSearch(OptimMixin, torch.optim.Optimizer):
    r"""Riemannian line search optimizer.

    We try to minimize objective :math:`f\colon M\to \mathbb{R}`, in a search
    direction :math:`\eta`.
    This is done by minimizing the line search objective

    .. math::

        \phi(\alpha) = f(R_x(\alpha\eta)),

    where :math:`R_x` is the retraction at :math:`x`.
    Its derivative is given by

    .. math::

        \phi'(\alpha) = \langle\mathrm{grad} f(R_x(\alpha\eta)),\,
        \mathcal T_{\alpha\eta}(\eta) \rangle_{R_x(\alpha\eta)},

    where :math:`\mathcal T_\xi(\eta)` denotes the vector transport of :math:`\eta`
    to the point :math:`R_x(\xi)`.

    The search direction :math:`\eta` is defined recursively by

    .. math::

        \eta_{k+1} = -\mathrm{grad} f(R_{x_k}(\alpha_k\eta_k))
        + \beta \mathcal T_{\alpha_k\eta_k}(\eta_k)

    Here :math:`\beta` is the scale parameter. If :math:`\beta=0` this is steepest
    descent, other choices are Riemannian version of Fletcher-Reeves and
    Polak-Ribière scale parameters.

    Common conditions to accept the new point are the Armijo /
    sufficient decrease condition:

    .. math::

        \phi(\alpha)\leq \phi(0)+c_1\alpha\phi'(0)

    And additionally the curvature / (strong) Wolfe condition

    .. math::

        \phi'(\alpha)\geq c_2\phi'(0)

    The Wolfe conditions are more restrictive, but guarantee that search direction
    :math:`\eta` is a descent direction.

    The constants :math:`c_1` and :math:`c_2` satisfy :math:`c_1\in (0,1)`
    and :math:`c_2\in (c_1,1)`.

    Parameters
    ----------
    params : iterable
        iterable of parameters to optimize or dicts defining
        parameter groups
    line_search_method : ('wolfe', 'armijo', or callable)
        Which line_search_method to use. If callable it should be any method
        of signature `(phi, derphi, **kwargs) -> step_size`,
        where phi is scalar line search objective, and derphi is its derivative.
        If no suitable step size can be found, the method should return `None`.
        The following arguments are always passed in `**kwargs`:
        * **phi0:** float, Value of phi at 0
        * **old_phi0:** float, Value of phi at previous point
        * **derphi0:** float, Value derphi at 0
        * **old_derphi0:** float, Value of derphi at previous point
        * **old_step_size:** float, Stepsize at previous point
        If any of these arguments are undefined, they default to `None`.
        Additional arguments can be supplied through the `line_search_params` parameter
    line_search_params : dict
        Extra parameters to pass to `line_search_method`, for
        the parameters available to strong Wolfe see :meth:`strong_wolfe_line_search`.
        For Armijo backtracking parameters see :meth:`armijo_backtracking`.
    cg_method : ('steepest', 'fr', 'pr', or callable)
        Method used to compute the conjugate gradient scale parameter beta.
        If 'steepest', set the scale parameter to zero, which is equivalent
        to doing steepest descent. Use 'fr' for Fletcher-Reeves, or 'pr' for
        Polak-Ribière (NB: this setting requires an additional vector transport).
        If callable, it should be a function of signature
        `(params, states, **kwargs) -> beta`,
        where params are the parameters of this optimizer,
        states are the states associated to the parameters (self._states),
        and beta is a float giving the scale parameter. The keyword
        arguments are specified in optional parameter `cg_kwargs`.

    Other Paremeters
    ----------------
    compute_derphi : bool, optional
        If True, compute the derivative of the line search objective phi
        for every trial step_size alpha. If alpha is not zero, this requires
        a vector transport and an extra gradient computation. This is always set
        True if `line_search_method='wolfe'` and False if `'armijo'`, but needs
        to be manually set for a user implemented line search method.
    transport_grad : bool, optional
        If True, the transport of the gradient to the new point is computed
        at the end of every step. Set to `True` if Polak-Ribière is used, otherwise
        defaults to `False`.
    transport_search_direction: bool, optional
        If True, transport the search direction to new point at end of every step.
        Set to False if steepest descent is used, True Otherwise.
    fallback_stepsize : float
        fallback_stepsize to take if no point can be found satisfying
        line search conditions. See also :meth:`step` (default: 1)
    stabilize : int
        Stabilize parameters if they are off-manifold due to numerical
        reasons every `stabilize` steps (default: `None` -- no stabilize)
    cg_kwargs : dict
        Additional parameters to pass to the method used to compute the
        conjugate gradient scale parameter.

    Attributes
    ----------
    last_step_size : int or `None`
        Last step size taken. If `None` no suitable step size was
        found, and consequently no step was taken.
    step_size_history : List[int or `None`]
        List of all step sizes taken so far.
    line_search_method : callable
    line_search_params : dict
    cg_method : callable
    cg_kwargs : dict
    fallback_stepsize : float
    """

    def __init__(
        self,
        params,
        line_search_method="armijo",
        line_search_params=None,
        cg_method="steepest",
        cg_kwargs=None,
        compute_derphi=True,
        transport_grad=False,
        transport_search_direction=True,
        fallback_stepsize=1,
        stabilize=None,
    ):
        defaults = dict(
            line_search_method=line_search_method,
            line_search_params=line_search_params,
            cg_method=cg_method,
            cg_kwargs=cg_kwargs,
            compute_derphi=compute_derphi,
            transport_grad=transport_grad,
            transport_search_direction=transport_search_direction,
            fallback_stepsize=fallback_stepsize,
            stabilize=stabilize,
        )
        super(RiemannianLineSearch, self).__init__(
            params, defaults, stabilize=stabilize
        )
        self._params = []
        for group in self.param_groups:
            group.setdefault("step", 0)
            self._params.extend(group["params"])
        if len(self.param_groups) > 1:
            warning_string = """Multiple parameter groups detected.
            Line search parameters will be taken from first group.
            """
            warnings.warn(warning_string, UserWarning)

        self.compute_derphi = self.param_groups[0]["compute_derphi"]
        ls_method = self.param_groups[0]["line_search_method"]
        if ls_method == "wolfe":
            self.line_search_method = strong_wolfe_line_search
            self.compute_derphi = True
        elif ls_method == "armijo":
            self.line_search_method = armijo_backtracking
            self.compute_derphi = False
        elif callable(ls_method):
            self.line_search_method = ls_method
        else:
            raise ValueError(f"Unrecognized line search method '{ls_method}'")

        self.cg_kwargs = self.param_groups[0]["cg_kwargs"]
        if self.cg_kwargs is None:
            self.cg_kwargs = dict()
        self.transport_grad = self.param_groups[0]["transport_grad"]
        self.transport_search_direction = self.param_groups[0][
            "transport_search_direction"
        ]
        cg_method = self.param_groups[0]["cg_method"]
        if cg_method in ("steepest", "constant"):
            self.cg_method = cg_constant
            self.transport_search_direction = False
        elif cg_method in ("fr", "fletcher-reeves"):
            self.cg_method = cg_fletcher_reeves
        elif cg_method in ("pr", "polak-ribiere"):
            self.cg_method = cg_polak_ribiere
            self.transport_grad = True
        elif callable(cg_method):
            self.cg_method = cg_method
        else:
            raise ValueError(f"Unrecognized scale parameter method '{cg_method}'")

        self.line_search_params = self.param_groups[0]["line_search_params"]
        if self.line_search_params is None:
            self.line_search_params = dict()
        self.fallback_stepsize = self.param_groups[0]["fallback_stepsize"]

        self.old_phi0 = None
        self.old_derphi0 = None
        self.step_size_history = []
        self.last_step_size = None
        self._last_step = None
        self._grads_computed = False
        self.prev_loss = None
        self.closure = None
        self._step_size_dic = dict()

    def _phi(self, step_size):
        """Compute the line search objective, and store its derivatives in the state."""

        if step_size in self._step_size_dic:
            return self._step_size_dic[step_size]

        param_copy = [param.clone() for param in self._params]

        for point in self._params:
            state = self.state[point]
            if "search_direction" not in state:  # this shouldn't be possible actually
                raise ValueError("Search direction for parameter not computed.")
            if isinstance(point, (ManifoldParameter, ManifoldTensor)):
                manifold = point.manifold
            else:  # Use euclidean manifold
                manifold = Euclidean()

            search_direction = state["search_direction"]

            # compute retract and transport in search direction
            if self.compute_derphi:
                new_point, search_transported = manifold.retr_transp(
                    point, step_size * search_direction, search_direction
                )
                # This should not have any effect, but it does
                new_point = manifold.projx(new_point)
                state["search_transported"] = manifold.proju(
                    new_point, search_transported
                )
            else:
                new_point = manifold.retr(point, step_size * search_direction)

            with torch.no_grad():
                point.copy_(new_point)

        # recompute loss at new point
        phi = self.closure()

        if self.compute_derphi:
            # Store new gradients in state
            for point in self._params:
                grad = point.grad
                if grad is None:
                    continue
                if isinstance(point, (ManifoldParameter, ManifoldTensor)):
                    manifold = point.manifold
                else:  # Use euclidean manifold
                    manifold = Euclidean()

                state = self.state[point]

                # project gradient onto tangent space
                grad = manifold.egrad2rgrad(point, grad)

                state["new_grad"] = grad
                state["der_phi"] = torch.sum(
                    manifold.inner(point, grad, state["search_transported"])
                ).item()

            self._grads_computed = True

        # roll back parameters to before step, save new point is state
        with torch.no_grad():
            for point, old_point in zip(self._params, param_copy):
                state = self.state[point]
                state["new_point"] = point.clone()

                point.copy_(old_point)

        self._step_size_dic[step_size] = phi
        self._last_step = step_size
        return phi

    def _derphi(self, step_size):
        """Compute derivative of phi.

        The derivative of phi is given by computing inner
        product between all tensor gradients at target point and those at source point.
        The source gradients are transported to the target point, and both gradients are
        projected.
        """

        if not self.compute_derphi:
            raise ValueError("Cannot call _derphi if self.compute_derphi=False!")

        # Call _phi to compute gradients; Does nothing if _phi was
        # already called with this stepsize during this step
        self._phi(step_size)

        derphi = 0
        for point in self._params:
            state = self.state[point]
            if "der_phi" not in state:
                continue

            derphi += state["der_phi"]

        return derphi

    def _init_loss(self, recompute_gradients=False):
        """Compute loss, gradients and search direction at start of line search.

        Use `recompute_gradients=True` if gradients have changed between
        consecutive calls of `step`.
        """

        if recompute_gradients or (not self._grads_computed):
            loss = self.closure()
            reuse_grads = False
        else:
            loss = self.prev_loss
            reuse_grads = True

        self._step_size_dic = dict()

        for point in self._params:
            state = self.state[point]
            if isinstance(point, (ManifoldParameter, ManifoldTensor)):
                manifold = point.manifold
            else:  # Use euclidean manifold
                manifold = Euclidean()
            if reuse_grads:
                grad = state["new_grad"]
            else:
                grad = point.grad
                grad = manifold.egrad2rgrad(point, grad)

            if "grad" in state:
                state["prev_grad"] = state["grad"]
                state["prev_grad_norm_squared"] = torch.sum(
                    manifold.inner(point, state["grad"])
                ).item()
            state["grad"] = grad

        derphi0 = self._compute_search_direction()

        self._grads_computed = True

        return loss, derphi0

    def _compute_search_direction(self):
        """Compute the search direction.

        If the direction is not a descent direction, revert to steepest descent.
        """

        first_time = False
        for point in self._params:
            state = self.state[point]
            if "search_direction" not in state:
                state["search_direction"] = -state["grad"]
                first_time = True

        if not first_time:
            beta = self.cg_method(self._params, self.state, **self.cg_kwargs)

            for point in self._params:
                state = self.state[point]

                if beta != 0:
                    state["search_direction"] = (
                        -state["grad"] + beta * state["search_transported"]
                    )
                else:
                    state["search_direction"] = -state["grad"]

        # Deriphative of phi at zero is inner product grad and search direction
        derphi0 = 0
        for point in self._params:
            state = self.state[point]
            if isinstance(point, (ManifoldParameter, ManifoldTensor)):
                manifold = point.manifold
            else:  # Use euclidean manifold
                manifold = Euclidean()
            derphi0 += torch.sum(
                manifold.inner(point, state["grad"], state["search_direction"])
            ).item()

        # If search direction is not a descent direction, revert to gradient
        if derphi0 >= 0:
            derphi0 = 0
            for point in self._params:
                state = self.state[point]
                if isinstance(point, (ManifoldParameter, ManifoldTensor)):
                    manifold = point.manifold
                else:  # Use euclidean manifold
                    manifold = Euclidean()
                derphi0 -= torch.sum(manifold.inner(point, state["grad"])).item()
                state["search_direction"] = -state["grad"]

        return derphi0

    def step(self, closure, force_step=False, recompute_gradients=False, no_step=False):
        """Do a linesearch step.

        Parameters
        ----------
        closure : callable
            A closure that reevaluates the model and returns the loss.
        force_step : bool (optional)
            If `True`, take a unit step of size `self.fallback_stepsize`
            if no suitable step size can be found.
            If `False`, no step is taken in this situation. (default: `False`)
        recompute_gradients : bool (optional)
            If True, recompute the gradients. Use this if the parameters
            have changed in between consecutive steps. (default: False)
        no_step : bool (optional)
            If True, just compute step size and do not perform the step.
            (default: False)
        """

        self.closure = closure
        phi0, derphi0 = self._init_loss(recompute_gradients=recompute_gradients)
        self._step_size_dic = dict()

        phi_information = {
            "phi0": phi0,
            "derphi0": derphi0,
            "old_phi0": self.old_phi0,
            "old_derphi0": self.old_derphi0,
            "old_step_size": self.last_step_size,
        }
        step_size = self.line_search_method(
            self._phi, self._derphi, **phi_information, **self.line_search_params
        )

        self.step_size_history.append(step_size)
        self.last_step_size = step_size

        self.old_phi0 = phi0
        self.old_derphi0 = derphi0

        # Ensure that the last step for which we computed the closure coincides with
        # proposed step size, so that we can reuse the gradients and retract.
        # This is very rare, and should only happen if force_step=True and no stepsize
        # was found, or for user-defined linesearch methods.
        if self._last_step != step_size or not self.compute_derphi:
            self._grads_computed = False
        redo_retract = self._last_step != step_size

        # If it fails to find a good step, probably we have convergence
        if step_size is None:
            if force_step:
                step_size = self.fallback_stepsize
                self._grads_computed = False
            elif (
                self.last_step_size is None
            ):  # Warn if step_size is None twice in a row
                warning_string = """No suitable step size could be found, and no step
                was taken. Call `step` with `force_step=True` to take a step anyway.
                """
                warnings.warn(warning_string, LineSearchWarning)

        for point in self._params:
            if step_size is None or no_step:
                continue
            state = self.state[point]
            if "search_direction" not in state:
                continue
            if isinstance(point, (ManifoldParameter, ManifoldTensor)):
                manifold = point.manifold
            else:  # Use euclidean manifold
                manifold = Euclidean()

            search_direction = state["search_direction"]

            # Compute retract if suggested step size is not the last one we tried (rare)
            if redo_retract:
                new_point = manifold.retr(point, step_size * search_direction)
            else:
                new_point = state["new_point"]

            # Use retract to perform the step, and transport the search direction
            if self.transport_search_direction:
                search_transported = manifold.transp_follow_retr(
                    point, step_size * search_direction, search_direction
                )
                state["search_transported"] = search_transported

            if self.transport_grad:
                grad_transport = manifold.transp_follow_retr(
                    point, step_size * search_direction, state["grad"]
                )
                state["grad_transported"] = grad_transport

            with torch.no_grad():  # Take suggested step
                point.copy_(new_point)

        for group in self.param_groups:
            group["step"] += 1
            if (
                group["stabilize"] is not None
                and group["step"] % group["stabilize"] == 0
            ):
                self.stabilize_group(group)

        # Update loss value
        if step_size is not None:
            new_loss = self._phi(step_size)
            self.prev_loss = new_loss
        else:
            new_loss = self.prev_loss
        return new_loss

    def stabilize_group(self, group):
        for p in group["params"]:
            if not isinstance(p, (ManifoldParameter, ManifoldTensor)):
                continue
            state = self.state[p]
            if not state:  # due to None grads
                continue
            manifold = p.manifold
            p.copy_(manifold.projx(p))


#################################################################################
# Conjugate gradient scale factor
#################################################################################


def cg_constant(params, states, alpha=0, **kwargs):
    """Constant scale parameter. If alpha=0, then this is steepest descent."""

    return alpha


def cg_fletcher_reeves(params, states, **kwargs):
    r"""Fletcher-Reeves scale parameter.

    This is given by

    .. math::
        \beta_{k+1}^{FR} = \frac{\langle\nabla f(x_{k+1},\,
        \nabla f(x_{k+1}\rangle_{x_{k+1}}
        {\langle\nabla f(x_k),\nabla f(x_k)\rangle_{x_k}}
    """
    numerator = 0
    denominator = 0
    for point in params:
        state = states[point]

        # Can't compute beta, probably first step hasn't been taken yet
        if "prev_grad_norm_squared" not in state:
            return 0

        if isinstance(point, (ManifoldParameter, ManifoldTensor)):
            manifold = point.manifold
        else:  # Use euclidean manifold
            manifold = Euclidean()
        numerator += torch.sum(manifold.inner(point, state["grad"])).item()
        denominator += state["prev_grad_norm_squared"]

    if denominator == 0:
        return 0
    else:
        return numerator / denominator


def cg_polak_ribiere(params, states, **kwargs):
    r"""Polak-Ribière scale parameter.

    This is given by

    .. math::
        \beta_{k+1}^{PR} = \frac{\langle\nabla f(x_{k+1}
        ,\,\nabla f(x_{k+1})-\mathcal T_{\alpha_k\eta_k}\nabla f(x_k)\rangle_{x_{k+1}}}
        {\langle\nabla f(x_k),\,\nabla f(x_k)\rangle_{x_k}}
    """

    numerator = 0
    denominator = 0
    for point in params:
        state = states[point]

        # Can't compute beta, probably first step hasn't been taken yet.
        if "grad_transported" not in state:
            return 0

        if isinstance(point, (ManifoldParameter, ManifoldTensor)):
            manifold = point.manifold
        else:  # Use euclidean manifold
            manifold = Euclidean()
        numerator += torch.sum(
            manifold.inner(
                point, state["grad"], state["grad"] - state["grad_transported"]
            )
        ).item()
        denominator += state["prev_grad_norm_squared"]

    if denominator == 0:
        return 0
    else:
        return numerator / denominator


#################################################################################
# Line search methods
#################################################################################


def strong_wolfe_line_search(
    phi,
    derphi,
    phi0=None,
    old_phi0=None,
    derphi0=None,
    c1=1e-4,
    c2=0.9,
    amax=None,
    **kwargs,
):
    """
    Scalar line search method to find step size satisfying strong Wolfe conditions.

    Parameters
    ----------
    c1 : float, optional
        Parameter for Armijo condition rule.
    c2 : float, optional
        Parameter for curvature condition rule.
    amax : float, optional
        Maximum step size

    Returns
    -------
    step_size : float
        The next step size
    """

    step_size, _, _, _ = scalar_search_wolfe2(
        phi,
        derphi,
        phi0=phi0,
        old_phi0=old_phi0,
        c1=c1,
        c2=c2,
        amax=amax,
    )

    return step_size


def armijo_backtracking(
    phi,
    derphi,
    phi0=None,
    derphi0=None,
    old_phi0=None,
    c1=1e-4,
    amin=0,
    amax=None,
    **kwargs,
):
    """Scalar line search method to find step size satisfying Armijo conditions.

    Parameters
    ----------
    c1 : float, optional
        Parameter for Armijo condition rule.
    amax, amin : float, optional
        Maxmimum and minimum step size
    """

    # TODO: Allow different schemes to choose initial step size

    if old_phi0 is not None and derphi0 != 0:
        alpha0 = 1.01 * 2 * (phi0 - old_phi0) / derphi0
    else:
        alpha0 = 1.0
    if alpha0 <= 0:
        alpha0 = 1.0
    if amax is not None:
        alpha0 = min(alpha0, amax)

    step_size, _ = scalar_search_armijo(
        phi, phi0, derphi0, c1=c1, alpha0=alpha0, amin=amin
    )

    return step_size
