import torch
from grp_prox import GroupProx
from pdhg import PDHGSolver
from bilevel import Bilevel
from constraints import Inequality, Equality


torch.manual_seed(0)
dtype = torch.float64

n = 200
n_groups = 20
group_size = 10
m = 10

groups = [torch.arange(i*group_size, (i+1)*group_size)
          for i in range(n_groups)]

c = torch.randn(n, dtype=dtype) * 0.5
mu = 0.2

z_l = 0.1 * torch.ones(m)
z_u = 5.0 * torch.ones(m)

b = torch.ones(m) * 5.0


def A_fn(z):
    A = torch.zeros((m, n), dtype=z.dtype, device=z.device)
    for k in range(m):
        for g in range(n_groups):
            A[k, groups[g]] = z[k]
    return A


def b_fn(z):
    return b


ineq_map = Inequality(
    A_fn=A_fn,
    b_fn=b_fn,
)

eq_map = None

p = 2.0
q = 1.0
mu = 0.3

l = -torch.ones(n, dtype=dtype)
u = torch.ones(n, dtype=dtype)

prox = GroupProx(
    groups=groups,
    tau=1e-3,
    p=p,
    q=q,
    mu=mu,
    l=l,
    u=u,
    eta=0.5,
    eps=1e-6,
    max_iter=10,
    detach_internal=False
)

solver = PDHGSolver(
    prox=prox,
    ineq_map=ineq_map,
    eq_map=eq_map,
    c=c,
    tau=1e-3,
    sigma=1e-3,
    theta=0.5,
    max_iter=200,
)

bilevel = Bilevel(
    solver=solver,
    prox=prox,
    c=c,
    z_init=torch.linspace(0.5, 2.0, m, dtype=dtype),
    z_l=z_l,
    z_u=z_u,
    z_lr=1e-3,
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

print(bilevel.z)