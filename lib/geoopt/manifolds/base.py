import abc
import torch.nn
import itertools
from typing import Optional, Tuple, Union

__all__ = ["Manifold", "ScalingInfo"]


class ScalingInfo(object):
    """
    Scaling info for each argument that requires rescaling.

    .. code:: python

        scaled_value = value * scaling ** power if power != 0 else value

    For results it is not always required to set powers of scaling, then it is no-op.

    The convention for this info is the following. The output of a function is either a tuple or a single object.
    In any case, outputs are treated as positionals. Function inputs, in contrast, are treated by keywords.
    It is a common practice to maintain function signature when overriding, so this way may be considered
    as a sufficient in this particular scenario. The only required info for formula above is ``power``.
    """

    # marks method to be not working with Scaled wrapper
    NotCompatible = object()
    __slots__ = ["kwargs", "results"]

    def __init__(self, *results: float, **kwargs: float):
        self.results = results
        self.kwargs = kwargs


class ScalingStorage(dict):
    """
    Helper class to make implementation transparent.

    This is just a dictionary with additional overriden ``__call__``
    for more explicit and elegant API to declare members. A usage example may be found in :class:`Manifold`.

    Methods that require rescaling when wrapped into :class:`Scaled` should be defined as follows

    1. Regular methods like ``dist``, ``dist2``, ``expmap``, ``retr`` etc. that are already present in the base class
    do not require registration, it has already happened in the base :class:`Manifold` class.

    2. New methods (like in :class:`PoincareBall`) should be treated with care.

    .. code-block:: python

        class PoincareBall(Manifold):
            # make a class copy of __scaling__ info. Default methods are already present there
            __scaling__ = Manifold.__scaling__.copy()
            ... # here come regular implementation of the required methods

            @__scaling__(ScalingInfo(1))  # rescale output according to rule `out * scaling ** 1`
            def dist0(self, x: torch.Tensor, *, dim=-1, keepdim=False):
                return math.dist0(x, c=self.c, dim=dim, keepdim=keepdim)

            @__scaling__(ScalingInfo(u=-1))  # rescale argument `u` according to the rule `out * scaling ** -1`
            def expmap0(self, u: torch.Tensor, *, dim=-1, project=True):
                res = math.expmap0(u, c=self.c, dim=dim)
                if project:
                    return math.project(res, c=self.c, dim=dim)
                else:
                    return res
            ... # other special methods implementation

    3. Some methods are not compliant with the above rescaling rules. We should mark them as `NotCompatible`

    .. code-block:: python

            # continuation of the PoincareBall definition
            @__scaling__(ScalingInfo.NotCompatible)
            def mobius_fn_apply(
                self, fn: callable, x: torch.Tensor, *args, dim=-1, project=True, **kwargs
            ):
                res = math.mobius_fn_apply(fn, x, *args, c=self.c, dim=dim, **kwargs)
                if project:
                    return math.project(res, c=self.c, dim=dim)
                else:
                    return res
    """

    def __call__(self, scaling_info: ScalingInfo, *aliases):
        def register(fn):
            self[fn.__name__] = scaling_info
            for alias in aliases:
                self[alias] = scaling_info
            return fn

        return register

    def copy(self):
        return self.__class__(self)


class Manifold(torch.nn.Module, metaclass=abc.ABCMeta):
    __scaling__ = ScalingStorage()  # will be filled along with implementation below
    name = None
    ndim = None
    reversible = None

    forward = NotImplemented

    def __init__(self, **kwargs):
        super().__init__()

    @property
    def device(self) -> Optional[torch.device]:
        """
        Manifold device.

        Returns
        -------
        Optional[torch.device]
        """
        p = next(itertools.chain(self.buffers(), self.parameters()), None)
        if p is not None:
            return p.device
        else:
            return None

    @property
    def dtype(self) -> Optional[torch.dtype]:
        """
        Manifold dtype.

        Returns
        -------
        Optional[torch.dtype]
        """

        p = next(itertools.chain(self.buffers(), self.parameters()), None)
        if p is not None:
            return p.dtype
        else:
            return None

    def check_point(
        self, x: torch.Tensor, *, explain=False
    ) -> Union[Tuple[bool, Optional[str]], bool]:
        """
        Check if point is valid to be used with the manifold.

        Parameters
        ----------
        x : torch.Tensor
            point on the manifold
        explain: bool
            return an additional information on check

        Returns
        -------
        bool
            boolean indicating if tensor is valid and reason of failure if False

        Notes
        -----
        This check is compatible to what optimizer expects, last dimensions are treated as manifold dimensions
        """
        ok, reason = self._check_shape(x.shape, "x")
        if explain:
            return ok, reason
        else:
            return ok

    def assert_check_point(self, x: torch.Tensor):
        """
        Check if point is valid to be used with the manifold and raise an error with informative message on failure.

        Parameters
        ----------
        x : torch.Tensor
            point on the manifold

        Notes
        -----
        This check is compatible to what optimizer expects, last dimensions are treated as manifold dimensions
        """

        ok, reason = self._check_shape(x.shape, "x")
        if not ok:
            raise ValueError(
                "`x` seems to be not valid "
                "tensor for {} manifold.\nerror: {}".format(self.name, reason)
            )

    def check_vector(self, u: torch.Tensor, *, explain=False):
        """
        Check if vector is valid to be used with the manifold.

        Parameters
        ----------
        u : torch.Tensor
            vector on the tangent plane
        explain: bool
            return an additional information on check

        Returns
        -------
        bool
            boolean indicating if tensor is valid and reason of failure if False

        Notes
        -----
        This check is compatible to what optimizer expects, last dimensions are treated as manifold dimensions
        """
        ok, reason = self._check_shape(u.shape, "u")
        if explain:
            return ok, reason
        else:
            return ok

    def assert_check_vector(self, u: torch.Tensor):
        """
        Check if vector is valid to be used with the manifold and raise an error with informative message on failure.

        Parameters
        ----------
        u : torch.Tensor
            vector on the tangent plane

        Notes
        -----
        This check is compatible to what optimizer expects, last dimensions are treated as manifold dimensions
        """

        ok, reason = self._check_shape(u.shape, "u")
        if not ok:
            raise ValueError(
                "`u` seems to be not valid "
                "tensor for {} manifold.\nerror: {}".format(self.name, reason)
            )

    def check_point_on_manifold(
        self, x: torch.Tensor, *, explain=False, atol=1e-5, rtol=1e-5
    ) -> Union[Tuple[bool, Optional[str]], bool]:
        """
        Check if point :math:`x` is lying on the manifold.

        Parameters
        ----------
        x : torch.Tensor
            point on the manifold
        atol: float
            absolute tolerance as in :func:`numpy.allclose`
        rtol: float
            relative tolerance as in :func:`numpy.allclose`
        explain: bool
            return an additional information on check

        Returns
        -------
        bool
            boolean indicating if tensor is valid and reason of failure if False

        Notes
        -----
        This check is compatible to what optimizer expects, last dimensions are treated as manifold dimensions
        """
        ok, reason = self._check_shape(x.shape, "x")
        if ok:
            ok, reason = self._check_point_on_manifold(x, atol=atol, rtol=rtol)
        if explain:
            return ok, reason
        else:
            return ok

    def assert_check_point_on_manifold(self, x: torch.Tensor, *, atol=1e-5, rtol=1e-5):
        """
        Check if point :math`x` is lying on the manifold and raise an error with informative message on failure.

        Parameters
        ----------
        x : torch.Tensor
            point on the manifold
        atol: float
            absolute tolerance as in :func:`numpy.allclose`
        rtol: float
            relative tolerance as in :func:`numpy.allclose`
        """
        self.assert_check_point(x)
        ok, reason = self._check_point_on_manifold(x, atol=atol, rtol=rtol)
        if not ok:
            raise ValueError(
                "`x` seems to be a tensor "
                "not lying on {} manifold.\nerror: {}".format(self.name, reason)
            )

    def check_vector_on_tangent(
        self,
        x: torch.Tensor,
        u: torch.Tensor,
        *,
        ok_point=False,
        explain=False,
        atol=1e-5,
        rtol=1e-5
    ) -> Union[Tuple[bool, Optional[str]], bool]:
        """
        Check if :math:`u` is lying on the tangent space to x.

        Parameters
        ----------
        x : torch.Tensor
            point on the manifold
        u : torch.Tensor
            vector on the tangent space to :math:`x`
        atol: float
            absolute tolerance as in :func:`numpy.allclose`
        rtol: float
            relative tolerance as in :func:`numpy.allclose`
        explain: bool
            return an additional information on check
        ok_point: bool
            is a check for point required?

        Returns
        -------
        bool
            boolean indicating if tensor is valid and reason of failure if False
        """
        if not ok_point:
            ok, reason = self._check_shape(x.shape, "x")
            if ok:
                ok, reason = self._check_shape(u.shape, "u")
            if ok:
                ok, reason = self._check_point_on_manifold(x, atol=atol, rtol=rtol)
        else:
            ok = True
            reason = None
        if ok:
            ok, reason = self._check_vector_on_tangent(x, u, atol=atol, rtol=rtol)
        if explain:
            return ok, reason
        else:
            return ok

    def assert_check_vector_on_tangent(
        self, x: torch.Tensor, u: torch.Tensor, *, ok_point=False, atol=1e-5, rtol=1e-5
    ):
        """
        Check if u :math:`u` is lying on the tangent space to x and raise an error on fail.

        Parameters
        ----------
        x : torch.Tensor
            point on the manifold
        u : torch.Tensor
            vector on the tangent space to :math:`x`
        atol: float
            absolute tolerance as in :func:`numpy.allclose`
        rtol: float
            relative tolerance as in :func:`numpy.allclose`
        ok_point: bool
            is a check for point required?
        """
        if not ok_point:
            ok, reason = self._check_shape(x.shape, "x")
            if ok:
                ok, reason = self._check_shape(u.shape, "u")
            if ok:
                ok, reason = self._check_point_on_manifold(x, atol=atol, rtol=rtol)
        else:
            ok = True
            reason = None
        if ok:
            ok, reason = self._check_vector_on_tangent(x, u, atol=atol, rtol=rtol)
        if not ok:
            raise ValueError(
                "`u` seems to be a tensor "
                "not lying on tangent space to `x` for {} manifold.\nerror: {}".format(
                    self.name, reason
                )
            )

    @__scaling__(ScalingInfo(1))
    def dist(self, x: torch.Tensor, y: torch.Tensor, *, keepdim=False) -> torch.Tensor:
        """
        Compute distance between 2 points on the manifold that is the shortest path along geodesics.

        Parameters
        ----------
        x : torch.Tensor
            point on the manifold
        y : torch.Tensor
            point on the manifold
        keepdim : bool
            keep the last dim?

        Returns
        -------
        torch.Tensor
            distance between two points
        """
        raise NotImplementedError

    @__scaling__(ScalingInfo(2))
    def dist2(self, x: torch.Tensor, y: torch.Tensor, *, keepdim=False) -> torch.Tensor:
        """
        Compute squared distance between 2 points on the manifold that is the shortest path along geodesics.

        Parameters
        ----------
        x : torch.Tensor
            point on the manifold
        y : torch.Tensor
            point on the manifold
        keepdim : bool
            keep the last dim?

        Returns
        -------
        torch.Tensor
            squared distance between two points
        """
        return self.dist(x, y, keepdim=keepdim) ** 2

    @abc.abstractmethod
    @__scaling__(ScalingInfo(u=-1))
    def retr(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        Perform a retraction from point :math:`x` with given direction :math:`u`.

        Parameters
        ----------
        x : torch.Tensor
            point on the manifold
        u : torch.Tensor
            tangent vector at point :math:`x`

        Returns
        -------
        torch.Tensor
            transported point
        """
        raise NotImplementedError

    @abc.abstractmethod
    @__scaling__(ScalingInfo(u=-1))
    def expmap(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        r"""
        Perform an exponential map :math:`\operatorname{Exp}_x(u)`.

        Parameters
        ----------
        x : torch.Tensor
            point on the manifold
        u : torch.Tensor
            tangent vector at point :math:`x`

        Returns
        -------
        torch.Tensor
            transported point
        """
        raise NotImplementedError

    @__scaling__(ScalingInfo(1))
    def logmap(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        r"""
        Perform an logarithmic map :math:`\operatorname{Log}_{x}(y)`.

        Parameters
        ----------
        x : torch.Tensor
            point on the manifold
        y : torch.Tensor
            point on the manifold

        Returns
        -------
        torch.Tensor
            tangent vector
        """
        raise NotImplementedError

    @__scaling__(ScalingInfo(u=-1))
    def expmap_transp(
        self, x: torch.Tensor, u: torch.Tensor, v: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform an exponential map and vector transport from point :math:`x` with given direction :math:`u`.

        Parameters
        ----------
        x : torch.Tensor
            point on the manifold
        u : torch.Tensor
            tangent vector at point :math:`x`
        v : torch.Tensor
            tangent vector at point :math:`x` to be transported

        Returns
        -------
        torch.Tensor
            transported point
        """
        y = self.expmap(x, u)
        v_transp = self.transp(x, y, v)
        return y, v_transp

    @__scaling__(ScalingInfo(u=-1))
    def retr_transp(
        self, x: torch.Tensor, u: torch.Tensor, v: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform a retraction + vector transport at once.

        Parameters
        ----------
        x : torch.Tensor
            point on the manifold
        u : torch.Tensor
            tangent vector at point :math:`x`
        v : torch.Tensor
            tangent vector at point :math:`x` to be transported

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            transported point and vectors

        Notes
        -----
        Sometimes this is a far more optimal way to preform retraction + vector transport
        """
        y = self.retr(x, u)
        v_transp = self.transp(x, y, v)
        return y, v_transp

    @__scaling__(ScalingInfo(u=-1))
    def transp_follow_retr(
        self, x: torch.Tensor, u: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        r"""
        Perform vector transport following :math:`u`: :math:`\mathfrak{T}_{x\to\operatorname{retr}(x, u)}(v)`.

        This operation is sometimes is much more simpler and can be optimized.

        Parameters
        ----------
        x : torch.Tensor
            point on the manifold
        u : torch.Tensor
            tangent vector at point :math:`x`
        v : torch.Tensor
            tangent vector at point :math:`x` to be transported

        Returns
        -------
        torch.Tensor
            transported tensor
        """
        y = self.retr(x, u)
        return self.transp(x, y, v)

    @__scaling__(ScalingInfo(u=-1))
    def transp_follow_expmap(
        self, x: torch.Tensor, u: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        r"""
        Perform vector transport following :math:`u`: :math:`\mathfrak{T}_{x\to\operatorname{Exp}(x, u)}(v)`.

        Here, :math:`\operatorname{Exp}` is the best possible approximation of the true exponential map.
        There are cases when the exact variant is hard or impossible implement, therefore a
        fallback, non-exact, implementation is used.

        Parameters
        ----------
        x : torch.Tensor
            point on the manifold
        u : torch.Tensor
            tangent vector at point :math:`x`
        v : torch.Tensor
            tangent vector at point :math:`x` to be transported

        Returns
        -------
        torch.Tensor
            transported tensor
        """
        y = self.expmap(x, u)
        return self.transp(x, y, v)

    def transp(self, x: torch.Tensor, y: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        r"""
        Perform vector transport :math:`\mathfrak{T}_{x\to y}(v)`.

        Parameters
        ----------
        x : torch.Tensor
            start point on the manifold
        y : torch.Tensor
            target point on the manifold
        v : torch.Tensor
            tangent vector at point :math:`x`

        Returns
        -------
        torch.Tensor
           transported tensor
        """
        raise NotImplementedError

    @abc.abstractmethod
    def inner(
        self, x: torch.Tensor, u: torch.Tensor, v: torch.Tensor = None, *, keepdim=False
    ) -> torch.Tensor:
        """
        Inner product for tangent vectors at point :math:`x`.

        Parameters
        ----------
        x : torch.Tensor
            point on the manifold
        u : torch.Tensor
            tangent vector at point :math:`x`
        v : Optional[torch.Tensor]
            tangent vector at point :math:`x`
        keepdim : bool
            keep the last dim?

        Returns
        -------
        torch.Tensor
            inner product (broadcasted)
        """
        raise NotImplementedError

    def component_inner(
        self, x: torch.Tensor, u: torch.Tensor, v: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Inner product for tangent vectors at point :math:`x` according to components of the manifold.

        The result of the function is same as ``inner`` with ``keepdim=True`` for
        all the manifolds except ProductManifold. For this manifold it acts different way
        computing inner product for each component and then building an output correctly
        tiling and reshaping the result.

        Parameters
        ----------
        x : torch.Tensor
            point on the manifold
        u : torch.Tensor
            tangent vector at point :math:`x`
        v : Optional[torch.Tensor]
            tangent vector at point :math:`x`

        Returns
        -------
        torch.Tensor
            inner product component wise (broadcasted)

        Notes
        -----
        The purpose of this method is better adaptive properties in optimization since ProductManifold
        will "hide" the structure in public API.
        """
        return self.inner(x, u, v, keepdim=True)

    def norm(self, x: torch.Tensor, u: torch.Tensor, *, keepdim=False) -> torch.Tensor:
        """
        Norm of a tangent vector at point :math:`x`.

        Parameters
        ----------
        x : torch.Tensor
            point on the manifold
        u : torch.Tensor
            tangent vector at point :math:`x`
        keepdim : bool
            keep the last dim?

        Returns
        -------
        torch.Tensor
            inner product (broadcasted)
        """
        return self.inner(x, u, keepdim=keepdim) ** 0.5

    @abc.abstractmethod
    def proju(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        Project vector :math:`u` on a tangent space for :math:`x`, usually is the same as :meth:`egrad2rgrad`.

        Parameters
        ----------
        x torch.Tensor
            point on the manifold
        u torch.Tensor
            vector to be projected

        Returns
        -------
        torch.Tensor
            projected vector
        """
        raise NotImplementedError

    @abc.abstractmethod
    def egrad2rgrad(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        Transform gradient computed using autodiff to the correct Riemannian gradient for the point :math:`x`.

        Parameters
        ----------
        x torch.Tensor
            point on the manifold
        u torch.Tensor
            gradient to be projected

        Returns
        -------
        torch.Tensor
            grad vector in the Riemannian manifold
        """
        raise NotImplementedError

    @abc.abstractmethod
    def projx(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project point :math:`x` on the manifold.

        Parameters
        ----------
        x torch.Tensor
            point to be projected

        Returns
        -------
        torch.Tensor
            projected point
        """
        raise NotImplementedError

    def _check_shape(
        self, shape: Tuple[int], name: str
    ) -> Union[Tuple[bool, Optional[str]], bool]:
        """
        Util to check shape.

        Exhaustive implementation for checking if
        a given point has valid dimension size,
        shape, etc. It should return boolean and
        a reason of failure if check is not passed

        Parameters
        ----------
        shape : Tuple[int]
            shape of point on the manifold
        name : str
            name to be present in errors

        Returns
        -------
        bool, str or None
            check result and the reason of fail if any
        """
        ok = len(shape) >= self.ndim
        if not ok:
            reason = "'{}' on the {} requires more than {} dim".format(
                name, self, self.ndim
            )
        else:
            reason = None
        return ok, reason

    def _assert_check_shape(self, shape: Tuple[int], name: str):
        """
        Util to check shape and raise an error if needed.

        Exhaustive implementation for checking if
        a given point has valid dimension size,
        shape, etc. It will raise a ValueError if check is not passed

        Parameters
        ----------
        shape : tuple
            shape of point on the manifold
        name : str
            name to be present in errors

        Raises
        ------
        ValueError
        """
        ok, reason = self._check_shape(shape, name)
        if not ok:
            raise ValueError(reason)

    @abc.abstractmethod
    def _check_point_on_manifold(
        self, x: torch.Tensor, *, atol=1e-5, rtol=1e-5
    ) -> Union[Tuple[bool, Optional[str]], bool]:
        """
        Util to check point lies on the manifold.

        Exhaustive implementation for checking if
        a given point lies on the manifold. It
        should return boolean and a reason of
        failure if check is not passed. You can
        assume assert_check_point is already
        passed beforehand

        Parameters
        ----------
        x torch.Tensor
            point on the manifold
        atol: float
            absolute tolerance as in :func:`numpy.allclose`
        rtol: float
            relative tolerance as in :func:`numpy.allclose`

        Returns
        -------
        bool, str or None
            check result and the reason of fail if any
        """
        # return True, None
        raise NotImplementedError

    @abc.abstractmethod
    def _check_vector_on_tangent(
        self, x: torch.Tensor, u: torch.Tensor, *, atol=1e-5, rtol=1e-5
    ) -> Union[Tuple[bool, Optional[str]], bool]:
        """
        Util to check a vector belongs to the tangent space of a point.

        Exhaustive implementation for checking if
        a given point lies in the tangent space at x
        of the manifold. It should return a boolean
        indicating whether the test was passed
        and a reason of failure if check is not passed.
        You can assume assert_check_point is already
        passed beforehand

        Parameters
        ----------
        x torch.Tensor
        u torch.Tensor
        atol : float
            absolute tolerance
        rtol :
            relative tolerance

        Returns
        -------
        bool, str or None
            check result and the reason of fail if any
        """
        # return True, None
        raise NotImplementedError

    def extra_repr(self):
        return ""

    def __repr__(self):
        extra = self.extra_repr()
        if extra:
            return self.name + "({}) manifold".format(extra)
        else:
            return self.name + " manifold"

    def unpack_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Construct a point on the manifold.

        This method should help to work with product and compound manifolds.
        Internally all points on the manifold are stored in an intuitive format.
        However, there might be cases, when this representation is simpler or more efficient to store in
        a different way that is hard to use in practice.

        Parameters
        ----------
        tensor : torch.Tensor

        Returns
        -------
        torch.Tensor
        """
        return tensor

    def pack_point(self, *tensors: torch.Tensor) -> torch.Tensor:
        """
        Construct a tensor representation of a manifold point.

        In case of regular manifolds this will return the same tensor. However, for e.g. Product manifold
        this function will pack all non-batch dimensions.

        Parameters
        ----------
        tensors : Tuple[torch.Tensor]

        Returns
        -------
        torch.Tensor
        """
        if len(tensors) != 1:
            raise ValueError("1 tensor expected, got {}".format(len(tensors)))
        return tensors[0]

    def random(self, *size, dtype=None, device=None, **kwargs) -> torch.Tensor:
        """
        Random sampling on the manifold.

        The exact implementation depends on manifold and usually does not follow all
        assumptions about uniform measure, etc.
        """
        raise NotImplementedError

    def origin(
        self,
        *size: Union[int, Tuple[int]],
        dtype=None,
        device=None,
        seed: Optional[int] = 42
    ) -> torch.Tensor:
        """
        Create some reasonable point on the manifold in a deterministic way.

        For some manifolds there may exist e.g. zero vector or some analogy.
        In case it is possible to define this special point, this point is returned with the desired size.
        In other case, the returned point is sampled on the manifold in a deterministic way.

        Parameters
        ----------
        size : Union[int, Tuple[int]]
            the desired shape
        device : torch.device
            the desired device
        dtype : torch.dtype
            the desired dtype
        seed : Optional[int]
            A parameter controlling deterministic randomness for manifolds that do not provide ``.origin``,
            but provide ``.random``. (default: 42)

        Returns
        -------
        torch.Tensor
        """
        if seed is not None:
            # we promise pseudorandom behaviour but do not want to modify global seed
            state = torch.random.get_rng_state()
            torch.random.manual_seed(seed)
            try:
                return self.random(*size, dtype=dtype, device=device)
            finally:
                torch.random.set_rng_state(state)
        else:
            return self.random(*size, dtype=dtype, device=device)
