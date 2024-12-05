import torch.nn
from .manifolds import Euclidean, Manifold
from .docutils import insert_docs
import functools
from typing import Union, Tuple
import copy

__all__ = ["ManifoldTensor", "ManifoldParameter"]


class ManifoldTensor(torch.Tensor):
    """Same as :class:`torch.Tensor` that has information about its manifold.

    Other Parameters
    ----------------
    manifold : :class:`geoopt.Manifold`
        A manifold for the tensor, (default: :class:`geoopt.Euclidean`)
    """

    try:
        # https://github.com/pytorch/pytorch/issues/46159#issuecomment-707817037
        from torch._C import _disabled_torch_function_impl  # noqa

        __torch_function__ = _disabled_torch_function_impl

    except ImportError:
        pass

    def __new__(
        cls, *args, manifold: Manifold = Euclidean(), requires_grad=False, **kwargs
    ):
        if len(args) == 1 and isinstance(args[0], torch.Tensor):
            data = args[0].data
        else:
            data = torch.Tensor(*args, **kwargs)
        if kwargs.get("device") is not None:
            data.data = data.data.to(kwargs.get("device"))
        with torch.no_grad():
            manifold.assert_check_point(data)
        instance = torch.Tensor._make_subclass(cls, data, requires_grad)
        instance.manifold = manifold
        return instance

    manifold: Manifold

    def proj_(self) -> torch.Tensor:
        """
        Inplace projection to the manifold.

        Returns
        -------
        tensor
            same instance
        """
        return self.copy_(self.manifold.projx(self))

    @insert_docs(Manifold.retr.__doc__, r"\s+x : .+\n.+", "")
    def retr(self, u: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.manifold.retr(self, u=u, **kwargs)

    @insert_docs(Manifold.expmap.__doc__, r"\s+x : .+\n.+", "")
    def expmap(self, u: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.manifold.expmap(self, u=u, **kwargs)

    @insert_docs(Manifold.inner.__doc__, r"\s+x : .+\n.+", "")
    def inner(self, u: torch.Tensor, v: torch.Tensor = None, **kwargs) -> torch.Tensor:
        return self.manifold.inner(self, u=u, v=v, **kwargs)

    @insert_docs(Manifold.proju.__doc__, r"\s+x : .+\n.+", "")
    def proju(self, u: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.manifold.proju(self, u, **kwargs)

    @insert_docs(Manifold.transp.__doc__, r"\s+x : .+\n.+", "")
    def transp(self, y: torch.Tensor, v: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.manifold.transp(self, y, v, **kwargs)

    @insert_docs(Manifold.retr_transp.__doc__, r"\s+x : .+\n.+", "")
    def retr_transp(
        self, u: torch.Tensor, v: torch.Tensor, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.manifold.retr_transp(self, u, v, **kwargs)

    @insert_docs(Manifold.expmap_transp.__doc__, r"\s+x : .+\n.+", "")
    def expmap_transp(self, u: torch.Tensor, v: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.manifold.expmap_transp(self, u, v, **kwargs)

    @insert_docs(Manifold.transp_follow_expmap.__doc__, r"\s+x : .+\n.+", "")
    def transp_follow_expmap(
        self, u: torch.Tensor, v: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        return self.manifold.transp_follow_expmap(self, u, v, **kwargs)

    @insert_docs(Manifold.transp_follow_retr.__doc__, r"\s+x : .+\n.+", "")
    def transp_follow_retr(
        self, u: torch.Tensor, v: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        return self.manifold.transp_follow_retr(self, u, v, **kwargs)

    def dist(
        self, other: torch.Tensor, p: Union[int, float, bool, str] = 2, **kwargs
    ) -> torch.Tensor:
        """
        Return euclidean  or geodesic distance between points on the manifold. Allows broadcasting.

        Parameters
        ----------
        other : tensor
        p : str|int
            The norm to use. The default behaviour is not changed and is just euclidean distance.
            To compute geodesic distance, :attr:`p` should be set to ``"g"``

        Returns
        -------
        scalar
        """
        if p == "g":
            return self.manifold.dist(self, other, **kwargs)
        else:
            return super().dist(other)

    @insert_docs(Manifold.logmap.__doc__, r"\s+x : .+\n.+", "")
    def logmap(self, y: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.manifold.logmap(self, y, **kwargs)

    def __repr__(self):
        return "Tensor on {} containing:\n".format(
            self.manifold
        ) + torch.Tensor.__repr__(self)

    # noinspection PyUnresolvedReferences
    def __reduce_ex__(self, proto):
        build, proto = super(ManifoldTensor, self).__reduce_ex__(proto)
        new_build = functools.partial(_rebuild_manifold_tensor, build_fn=build)
        new_proto = proto + (dict(), self.__class__, self.manifold, self.requires_grad)
        return new_build, new_proto

    @insert_docs(Manifold.unpack_tensor.__doc__, r"\s+tensor : .+\n.+", "")
    def unpack_tensor(self) -> Union[torch.Tensor, Tuple[torch.Tensor]]:
        return self.manifold.unpack_tensor(self)

    def __deepcopy__(self, memo):
        if id(self) in memo:
            return memo[id(self)]
        else:
            result = type(self)(
                self.data.clone(memory_format=torch.preserve_format),
                manifold=copy.deepcopy(self.manifold, memo=memo),
                requires_grad=self.requires_grad,
            )
            memo[id(self)] = result
            return result


class ManifoldParameter(ManifoldTensor, torch.nn.Parameter):
    """Same as :class:`torch.nn.Parameter` that has information about its manifold.

    It should be used within :class:`torch.nn.Module` to be recognized
    in parameter collection.

    Other Parameters
    ----------------
    manifold : :class:`geoopt.Manifold` (optional)
        A manifold for the tensor if ``data`` is not a :class:`geoopt.ManifoldTensor`
    """

    def __new__(cls, data=None, manifold=None, requires_grad=True):
        if data is None:
            data = ManifoldTensor(manifold=manifold or Euclidean())
        elif not isinstance(data, ManifoldTensor):
            data = ManifoldTensor(data, manifold=manifold or Euclidean())
        else:
            if manifold is not None and data.manifold != manifold:
                raise ValueError(
                    "Manifolds do not match: {}, {}".format(data.manifold, manifold)
                )
        instance = ManifoldTensor._make_subclass(cls, data, requires_grad)
        instance.manifold = data.manifold
        return instance

    def __repr__(self):
        return "Parameter on {} containing:\n".format(
            self.manifold
        ) + torch.Tensor.__repr__(self)


def _rebuild_manifold_tensor(*args, build_fn):
    tensor = build_fn(*args[:-4])
    return args[-3](tensor, manifold=args[-2], requires_grad=args[-1])
