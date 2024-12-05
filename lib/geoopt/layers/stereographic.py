import torch
import geoopt


class Distance2StereographicHyperplanes(torch.nn.Module):
    """Distances to Stereographic hyperplanes.

    This layer can be used as a feature extractor in deep learning.

    Examples
    --------
    >>> ball = geoopt.Stereographic(-1)

    >>> layer = torch.nn.Sequential(
    ...    Distance2StereographicHyperplanes(2, 10, ball=ball),
    ...    torch.nn.Linear(10, 32),
    ...    torch.nn.ReLU(),
    ...    torch.nn.Linear(32, 64),
    ... )
    >>> input = ball.random_normal(100, 2)
    >>> layer(input).shape
    torch.Size([100, 64])

    >>> layer = torch.nn.Sequential(
    ...    Distance2StereographicHyperplanes(2, 10, ball=ball, ndim=2),
    ...    torch.nn.Conv2d(10, 32, 3),
    ...    torch.nn.ReLU(),
    ...    torch.nn.Conv2d(32, 64, 3),
    ... )
    >>> input = ball.random_normal(100, 12, 12, 2).permute(0, 3, 1, 2) # BxCxHxW
    >>> input.shape
    torch.Size([100, 2, 12, 12])
    >>> layer(input).shape
    torch.Size([100, 64, 8, 8])
    """

    def __init__(
        self,
        plane_shape: int,
        num_planes: int,
        signed=True,
        squared=False,
        *,
        ball,
        init_std=1.0,
        ndim=0,
    ):
        super().__init__()
        self.ndim = ndim
        self.signed = signed
        self.squared = squared
        # Do not forget to save Manifold instance to the Module
        self.ball = ball
        self.plane_shape = geoopt.utils.size2shape(plane_shape)
        self.num_planes = num_planes

        # In a layer we create Manifold Parameters in the same way we do it for
        # regular pytorch Parameters, there is no difference. But geoopt optimizer
        # will recognize the manifold and adjust to it
        self.points = geoopt.ManifoldParameter(
            torch.empty(num_planes, plane_shape), manifold=self.ball
        )
        self.init_std = init_std
        self.reset_parameters()

    def forward(self, input_p):
        input_p = input_p.unsqueeze(-self.ndim - 1)
        points = self.points.permute(1, 0)
        points = points.view(points.shape + (1,) * self.ndim)

        distance = self.ball.dist2plane(
            x=input_p, p=points, a=points, signed=self.signed, dim=-self.ndim - 2
        )
        if self.squared and self.signed:
            sign = distance.sign()
            distance = distance**2 * sign
        elif self.squared:
            distance = distance**2
        return distance

    def extra_repr(self):
        return (
            f"ndim={self.ndim}, "
            f"plane_shape={self.plane_shape}, "
            f"num_planes={self.num_planes}"
        )

    @torch.no_grad()
    def reset_parameters(self):
        self.points.set_(self.ball.random_normal(*self.points.shape, std=self.init_std))
