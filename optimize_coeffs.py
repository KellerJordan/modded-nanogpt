import argparse
from functools import partial

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
import seaborn as sns
import sympy as sp

DEFAULT_EPS = 5e-2
DEFAULT_PRECISION = 4

gamma_ = sp.Symbol("gamma", interval=(5/4, sp.S.Infinity), left_open=True, right_open=True)
l_ = sp.Symbol("l", interval=(0, 1), left_open=False, right_open=True)
r_ = sp.Symbol("r", interval=(0, 1), left_open=False, right_open=True)
x_ = sp.Symbol("x", real=True)

fp_ = [-(1 + r_), -(1 - l_), 0, 1 - l_, 1 + r_]
iterator_ = x_ + gamma_ * (x_ - fp_[0])*(x_ - fp_[1])*(x_ - fp_[2])*(x_ - fp_[3])*(x_ - fp_[4])
iterator_simplified_ = sp.collect(sp.expand(iterator_), x_)

abc_iterator_jax = jax.jit(lambda x, a, b, c: a*x + b*x**3 + c*x**5)
glr_iterator_jax = sp.lambdify((x_, gamma_, l_, r_), iterator_simplified_, "jax")

a_, b_, c_ = sp.Poly(iterator_simplified_, x_).coeffs()[::-1]
a_jax = sp.lambdify((gamma_, l_, r_), a_, "jax")
b_jax = sp.lambdify((gamma_, l_, r_), b_, "jax")
c_jax = sp.lambdify((gamma_, l_, r_), c_, "jax")


def abc_to_glr_reparam(a: float, b: float, c: float, verbose: bool = False):
    iterator_fn = a*x_ + b*x_**3 + c*x_**5
    iterator_roots = sp.nroots(iterator_fn - x_)
    if verbose:
        print(iterator_roots)
    iterator_roots_real = [root.evalf() for root in iterator_roots if root.is_real]
    iterator_roots = sorted(iterator_roots_real)
    return float(c), float(1 - iterator_roots[-2]), float(iterator_roots[-1] - 1)


@partial(jax.jit, static_argnames=("decimals",))
def glr_to_abc_reparam(gamma: float, l: float, r: float, decimals: int = 4):
    abc = jnp.stack([a_jax(gamma, l, r), b_jax(gamma, l, r), c_jax(gamma, l, r)])
    return abc + jax.lax.stop_gradient(jnp.round(abc, decimals) - abc)


def loss(
    x: jax.Array,
    params: jax.Array,
    eps: float = DEFAULT_EPS,
    precision: int = DEFAULT_PRECISION,
    enable_contraction_aux_loss: bool = True,
    enable_flatness_aux_loss: bool = False,
):
    def scan_body_fn(y: jax.Array, glr: jax.Array):
        gamma, l, r = glr

        # The peak of the previous iteration should be at most 1 + r - eps
        # to prevent singular values from blowing up
        intermediate_loss = jnp.clip(y.max() - (1 + r - eps), min=0)

        a, b, c = glr_to_abc_reparam(gamma, l, r, precision)
        new_y = abc_iterator_jax(y, a, b, c)

        # The iterator must not cross the a-axis
        # to prevent singular values from switching signs
        intermediate_loss += jnp.clip(eps - jnp.amin(jnp.where(y > 0.5, new_y, jnp.inf)), min=0)

        return new_y, intermediate_loss
    y, intermediate_losses = jax.lax.scan(scan_body_fn, x, params)

    # This auxiliary loss term encourages the contraction of the
    # attractor basins of the iterators
    aesthetic_aux_loss = (
        jnp.clip(params[1:,2] - params[:-1,2], min=0).sum()
        + jnp.clip(params[1:,1] - params[:-1,1], min=0).sum()
        + jnp.clip(params[1:,0] - params[:-1,0], min=0).sum()
    )

    # This auxiliary loss term encourages the flatness of the composite curve
    # Taken from @YouJiacheng's code here: https://gist.github.com/YouJiacheng/393c90cbdc23b09d5688815ba382288b
    y_max = jnp.amax(y)
    y_min = jnp.amin(jnp.where(x > 0.05, y, jnp.inf))
    diff_ratio = (y_max - y_min) / jnp.clip(y_max, min=1e-3)

    loss1 = jnp.sqrt(jnp.mean((y - 1) ** 2))
    loss2 = (
        intermediate_losses.mean()
        + jnp.int32(enable_contraction_aux_loss) * aesthetic_aux_loss
        + jnp.int32(enable_flatness_aux_loss) * diff_ratio
    )
    return loss1 + loss2


loss_and_grad_fn = jax.jit(jax.value_and_grad(loss, argnums=1), static_argnums=(2, 3, 4, 5))


@partial(jax.jit, static_argnums=(2, 3, 4, 5, 6))
def train(
    x: jax.Array,
    params: jax.Array,
    learning_rate: float = 0.001,
    num_steps: int = 10_000,
    eps: float = DEFAULT_EPS,
    precision: int = DEFAULT_PRECISION,
    enable_contraction_aux_loss: bool = True,
    enable_flatness_aux_loss: bool = False,
):
    optimizer = optax.chain(
        # can also use optax.contrib.muon
        optax.adam(learning_rate=learning_rate),
        optax.clip_by_global_norm(max_norm=1.),
    )
    opt_state = optimizer.init(params)

    def body_fn(values: tuple[jax.Array, optax.OptState], _):
        params, opt_state = values
        loss, grad = loss_and_grad_fn(
            x,
            params,
            eps,
            precision,
            enable_contraction_aux_loss,
            enable_flatness_aux_loss,
        )
        updates, opt_state = optimizer.update(grad, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return (new_params, opt_state), (params, loss)
    (params, _), (historical_params, losses) = jax.lax.scan(body_fn, (params, opt_state), length=num_steps)
    return params, historical_params, losses


def plot_ns_iterator_fns(ax, params: jax.Array):
    max_r = params[:,-1].max().item() + 0.1
    x_space = jnp.linspace(-0.1, 1 + max_r, 100)

    palette = sns.color_palette("Blues", n_colors=params.shape[0])

    sns.lineplot(x=x_space, y=x_space, label='y=x', color="black", linestyle="--", alpha=0.25, ax=ax)

    for idx, glr in enumerate(params):
        gamma, l, r = glr
        a, b, c = glr_to_abc_reparam(gamma, l, r, DEFAULT_PRECISION)
        y_ = abc_iterator_jax(x_space, a, b, c)
        label = f"Iteration {idx+1}"
        # a, b, c = abc_reparametrize(gamma, r)
        # label = f"Iteration {idx+1}; ${a}x {b}x^3 + {c}x^5$"
        sns.lineplot(x=x_space, y=y_, label=label, color=palette[idx], ax=ax)

    ax.set_xlim(-0.1, 1 + max_r)
    ax.set_ylim(-0.1, 1 + max_r)
    ax.grid()
    ax.legend(loc="lower center")


def plot_ns_iteration_overall(axes, params: jax.Array, ref_num_iters: int = 5):
    x0 = jnp.concat([
        jnp.linspace(0, 1, 512),
        jnp.linspace(0, 0.01, 256),
        jnp.linspace(0, 0.001, 256),
    ])

    y_kj = [x0]
    n_iterations = params.shape[0]
    for _ in range(max(5, n_iterations)):
        y_kj.append(abc_iterator_jax(y_kj[-1], kj_a, kj_b, kj_c))

    def scan_fn(y, glr):
        gamma, l, r = glr
        a, b, c = glr_to_abc_reparam(gamma, l, r, DEFAULT_PRECISION)
        y = abc_iterator_jax(y, a, b, c)
        return y, None
    y2, _ = jax.lax.scan(scan_fn, x0, params)

    def plot_ns_iteration_overall_helper(ax, max_x=1.):
        sns.lineplot(x=x0, y=y_kj[n_iterations], label=f"Keller-Jordan {n_iterations}-steps", linestyle="--", ax=ax)
        if n_iterations != ref_num_iters:
            sns.lineplot(x=x0, y=y_kj[ref_num_iters], label=f"Keller-Jordan {ref_num_iters}-steps", linestyle="--", ax=ax)
        sns.lineplot(x=x0, y=y2, label=f"Optimized {best_params.shape[0]}-steps", color="black", ax=ax)
        ax.set_xlim(-max_x*0.01, max_x)
        ax.grid()
        ax.legend(loc="lower right")

    if isinstance(axes, plt.Axes):
        plot_ns_iteration_overall_helper(axes)
    else:
        plot_ns_iteration_overall_helper(axes[0], max_x=1.)
        plot_ns_iteration_overall_helper(axes[1], max_x=0.01)
        plot_ns_iteration_overall_helper(axes[2], max_x=0.001)


def plot_iterators(params: jax.Array, ref_num_iters: int = 5, savefile: str = "muon_ns_iterators.png"):
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()
    plot_ns_iterator_fns(axes[0], params)
    plot_ns_iteration_overall(axes[1:], params, ref_num_iters)
    plt.tight_layout()
    plt.savefig(savefile)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_ns_iters", help="Number of Newton-Schulz iterations", type=int, default=5
    )
    parser.add_argument(
        "--num_train_steps", help="Number of training steps", type=int, default=10_000
    )
    parser.add_argument(
        "--learning_rate", help="Learning rate", type=float, default=0.001
    )
    parser.add_argument(
        "--precision", help="Number of decimals in the coefficients", type=int, default=DEFAULT_PRECISION
    )
    parser.add_argument(
        "--eps", help="Epsilon", type=float, default=DEFAULT_EPS
    )
    parser.add_argument(
        "--enable_contraction_aux_loss", help="Enable contraction auxiliary loss", action="store_true", default=True
    )
    parser.add_argument(
        "--enable_flatness_aux_loss", help="Enable flatness auxiliary loss", action="store_true", default=False
    )
    args = parser.parse_args()

    # Reparametrize Keller Jordan's a-b-c coefficients to gamma-l-r
    kj_a, kj_b, kj_c = 3.4445, -4.7750, 2.0315
    kj_gamma, kj_inner_radius, kj_outer_radius = abc_to_glr_reparam(kj_a, kj_b, kj_c)
    # Check if the reparametrization is correct
    kj_abc = glr_to_abc_reparam(kj_gamma, kj_inner_radius, kj_outer_radius, decimals=4)
    assert jnp.allclose(kj_abc, jnp.array([kj_a, kj_b, kj_c]), atol=1e-4)

    x = jnp.concat([
        # The extra 0.1 is there to account for numerical instability
        jnp.linspace(0, 1.1, 2**10),
        # Gradients typically have low stable rank (i.e. most of the singular values are close to 0).
        # To simulate that, we add a couple more points near 0.
        jnp.linspace(0, 0.1, 2**9),
    ])
    init_params = jnp.array([[kj_gamma, kj_inner_radius, kj_outer_radius]]*args.num_ns_iters)

    trained_params, historical_params, losses = train(
        x=x,
        params=init_params,
        learning_rate=args.learning_rate,
        num_steps=args.num_train_steps,
        eps=args.eps,
        precision=args.precision,
        enable_contraction_aux_loss=args.enable_contraction_aux_loss,
        enable_flatness_aux_loss=args.enable_flatness_aux_loss,
    )

    best_params: jax.Array = historical_params[jnp.nanargmin(losses)]

    for gamma, l, r in best_params:
        a, b, c = glr_to_abc_reparam(gamma, l, r, args.precision)
        print(f"({a:.4f}, {b:.4f}, {c:.4f})")

    plot_iterators(best_params)
