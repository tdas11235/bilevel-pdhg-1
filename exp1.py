import torch
from grp_prox import GroupProx
from pdhg import PDHGSolver
from bilevel import Bilevel
from constraints import Inequality, Equality

torch.manual_seed(0)
dtype = torch.float64

n = 4
m = 2
c = torch.tensor([1.0, -0.5, 0.2, -0.1], dtype=dtype)

l = -torch.ones(n, dtype=dtype)
u = torch.ones(n, dtype=dtype)

z_l = torch.tensor([0.1], dtype=dtype)
z_u = torch.tensor([2.0], dtype=dtype)

# base constraint matrix
A0 = torch.tensor([
    [1.0, 0.0, 1.0, 0.0],
    [0.0, 1.0, 0.0, 1.0],
], dtype=dtype)
b0 = torch.tensor([0.5, 0.5], dtype=dtype)


def A_fn(z):
    return z[0] * A0

def b_fn(z):
    return b0


ineq_map = Inequality(
    A_fn=A_fn,
    b_fn=b_fn,
)

eq_map = None

groups = [
    torch.tensor([0, 1]),
    torch.tensor([2, 3]),
]

p = 2.0
q = 1.0          # group lasso
mu = 0.3

prox = GroupProx(
    groups=groups,
    tau=0.1,
    p=p,
    q=q,
    mu=mu,
    l=l,
    u=u,
    eta=0.5,
    eps=1e-6,
    max_iter=5,
    detach_internal=False
)

solver = PDHGSolver(
    prox=prox,
    ineq_map=ineq_map,
    eq_map=eq_map,
    c=c,
    tau=0.1,
    sigma=0.9,
    theta=1.0,
    max_iter=20,
)

bilevel = Bilevel(
    solver=solver,
    prox=prox,
    c=c,
    z_init=torch.tensor([1.5], dtype=dtype),
    z_l=z_l,
    z_u=z_u,
    z_lr=0.1,
)

x0 = torch.zeros(n, dtype=dtype)
y1_0 = torch.zeros(m, dtype=dtype)
y2_0 = None

history, x_final, y1_final, _ = bilevel.solve(
    x0=x0,
    y1_0=y1_0,
    y2_0=y2_0,
    num_steps=100,
    verbose=True,
)
