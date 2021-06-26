"""
    This script stores the setup parameters for jax  and common functions
    used in generating estimations for
    "The Welfare Effects of Dynamic Pricing:Evidence from Airline Markets"
--------------------------------------------------------------------------------
change log:
    v0.0.1  Mon 14 Jun 2021
-------------------------------------------------------------------------------
notes: 
--------------------------------------------------------------------------------
contributors:
    Kevin:
        name:       Kevin Williams
        email:      kevin.williams@yale.edu
--------------------------------------------------------------------------------
Copyright 2021 Yale University
"""

NUM_THREADS = 21
# adjust Jax to 64 bit floats
from jax.config import config
config.update("jax_enable_x64", True)

# define the probability that a low type wants to buy
# rewrite exp(a)/(1 + exp(a)) as 1/(1/exp(a) + 1) = 1/(1 + exp(-a))
@jit
def purchInL(beta, bL, gamma): # p is dim(P), gam is dim(T), beta is dim FE
    return (1 - gamma)[:, None, None] * 1 / \
        (1 + jnp.exp(-beta[None, None, :] - bL * prices[None, :, None]))
    # returned object is T, P dimension

# define the probability that a high type wants to buy
@jit
def purchInB(beta, bB, gamma): # p is dim(P), gam is dim(T), beta is dim FE
    return (gamma)[:, None, None] * 1 / \
        (1 + jnp.exp(-beta[None, None, :] - bB * prices[None, :, None]))
    # returned object is T,P dimension

# define, this returns a T X P matrix of probability of purchase across both consumer types
@jit
def purchIn(beta, bB, bL, gamma):
    return purchInL(beta, bL, gamma) + purchInB(beta, bB, gamma)

# define the log probability of that demand is equal to q given the parameters of the model
# this is the log of the poisson pmf
@jit
def log_demandQ(beta, bL, bB, gamma, mu, q):
    return q * jnp.log(mu[:, None, :]) + q * \
        jnp.log(purchIn(beta, bB, bL, gamma)) - \
        (mu[:, None, :] * purchIn(beta, bB, bL, gamma)) - gammaln(q + 1)

# define the probability of that demand is equal 
# to q given the parameters of the model, this is just exp(log(p)),
# which is done for numerical stablility
@jit
def demandQ(beta, bL, bB, gamma, mu, q):
    # note that this is simply
    # (mu[:, None] * purchIn(aB, bB, aL, bL, prices, gamma)) ** q * \
    #     jnp.exp(-mu[:, None] * (purchIn(aB, bB, aL, bL, prices, gamma))) / factQ[q]
    return jnp.exp(log_demandQ(beta, bL, bB, gamma, mu, q))

# fill in the matrix of all demand probabilities
def allDemand(beta, bL, bB, gamma, mu):
    vlookup = vmap((
        lambda x: demandQ(beta, bL, bB, gamma, mu, x)
    ))(jnp.arange(0, qBar))
    f = jnp.zeros((qBar, qBar, len(gamma), len(prices), len(beta)))
    for q in range(1, qBar): # seats remaining
        f = f.at[q, 0:q, :, :, :].set(vlookup[0:q, :, :, :])
        # define the probability of selling out as 1 - prob of not selling out,
        # ensuring probabilities sum to 1
        f = f.at[q, q, :, :, :].set(
            jnp.maximum(1 - jnp.sum(f[q, 0:(q), :, :, :], axis = 0), 1e-100)
        )
    # if any small floats are created, limit them to 1e-100
    # so we do not have floating point errors going forward
    f = jnp.where((f > 0) & (f < 1e-100), 1e-100, f)
    return f
    # returning dimension is seats remaining, seats sold, time, and prices

# define the CCP and EV for the dynamic firm problem
def dynEst(f, ER, gamma, sig, beta):
    np.seterr(all = "raise")
        # create storage for EV, V, CCP
    EV = jnp.zeros((T, qBar, numP, len(beta)))
    V = jnp.zeros((qBar, T, len(beta)))
    CCP = jnp.zeros((T, qBar, numP, len(beta)))
    for t in range(1, T+1):
        # work backwards in time. In the last period, we just get last period revenues
        if t == 1:
            # the softmax function can be rewritten,
            # so let"s use logsum(exp) = x* + log sum (exp (x-x*))
            grp = ER[:, -t, :, :] / (sig) * Pt[-t, 1:][None, :, None]
            grp = jnp.where(grp == 0, -jnp.inf, grp)
            V = V.at[:, -t, :].set(sig * logsumexp(grp , axis = 1) + EC * sig)
            V = V.at[0, -t, :].set(0)
            CCP = CCP.at[-t, :, :, :].set(grp - logsumexp(grp, axis = 1)[:, None, :])
            # this is rewritten from the numpy version:
            # CCP[-t, :, :] = Pt[-t, 1:][None, :] * np.exp(ER[:, -t, :] / sig) / \
            #     np.sum(np.exp(ER[:, -t, :] * Pt[-t, 1:][None, :] / sig), axis = 1)[:,None]
        else:
            grp = (ER[:, -t, :, :] / sig + EV[-t+1, :, :, :] / sig) * \
                Pt[-t, 1:][None, :, None]
            grp = jnp.where(grp == 0, -jnp.inf, grp)
            V = V.at[:, -t, :].set(sig * logsumexp(grp, axis = 1) + EC * sig)
            V = V.at[0, -t, :].set(0)
        # now we need to define EV, which is int_c" V f(c"),
        # so we"ll use tril to reset array(x) as reversearray(x)
        # this allows us to calc that the Pr(Q = 1) aligns with Pr(c" = c - 1)
        r, c = jnp.tril_indices_from(f[:, :, 0, 0, 0])
        for b in range(len(beta)):
            # update expected value function, this is for not the last period
            if t != T:
                g = jnp.array(f[:, :, -t - 1, :, b])
                g = g.at[r, c, :].set(g[r, r - c, :])
                EV = EV.at[-t, :, :, b].set(
                    jnp.sum(g * V[:, -t, b][None, :, None], axis = 1) * Pt[-t, 1:]
                )
        if t != 1:
            XX = (ER[:, -t, :, :] + EV[-t + 1, :, :, :]) / \
                sig * Pt[-t, 1:][None, :, None]
            XX = jnp.where(XX == 0, -jnp.inf, XX)
            CCP = CCP.at[-t, :, :, :].set(XX - logsumexp(XX, axis = 1)[:, None, :])
    return CCP

# calculate gradient for scaling factor using central differences
# this avoids a numerical issue with calculating the gradient entry via AD
# this only runs when sys.arg is set to fast. Else, the full gradient is computed
# which is more exact but takes considerably longer and uses more GPU mem.
def gradientSig(VAR, data):
    # The first parameters are consumer preferences:
    beta = jnp.array(VAR[0:7])
    bL = jnp.minimum(VAR[7], VAR[8])
    bB = jnp.maximum(VAR[7], VAR[8])
    gamma = 1 / (
        jnp.exp(
            -VAR[9] - jnp.arange(0, 60) * VAR[10] - \
                (jnp.arange(0, 60) ** 2) * VAR[11]
        ) + 1
    )
    # equivalent to jnp.array([1/(1 + jnp.exp(-g[0] + -t*g[1] - t**2*g[2])) for t in range(0,60)])
    # range(int(min(Tdata)), int(max(Tdata)+1))])
    muT = jnp.array(
        [VAR[12]] * (T - 20) + [VAR[13]] * 7 + \
            [VAR[14]] * 7 + [VAR[15]] * 6
    )
    muD = jnp.append(jnp.array([1]), jnp.array(VAR[16:22]))
    mu = muT[:, None] * muD[None, :]
    sig = VAR[-1]
    # first FE
    f0 = allDemand_jit(beta, bL, bB, gamma, mu)
    ER0 = jnp.sum(
        f0 * jnp.array(range(qBar))[None, :, None, None, None] * \
            prices[None, None, None, :, None],
        axis = 1
    )
    CCP0 = dynEst_jit(f0, ER0, gamma, sig + 1e-4, beta)
    loss0 = jnp.sum(CCP0[data[:, 2], data[:, 0], data[:, 3], data[:, 4]])
    CCP1 = dynEst_jit(f0, ER0, gamma, sig - 1e-4, beta)
    loss1 = jnp.sum(CCP1[data[:, 2], data[:, 0], data[:, 3], data[:, 4]])
    return (loss0 - loss1) / (2 * 1e-4)

# calculate the log-like for the data set
def logLike(VAR, data):
    # The first parameters are consumer preferences:
    beta = jnp.array(VAR[0:7])
    bL = jnp.minimum(VAR[7], VAR[8])
    bB = jnp.maximum(VAR[7], VAR[8])
    gamma = 1 / (jnp.exp(
        -VAR[9] - jnp.arange(0, 60) * VAR[10] - \
            (jnp.arange(0, 60) ** 2) * VAR[11]
        ) + 1
    )
    # equivalent to jnp.array([1/(1 + jnp.exp(-g[0] + -t*g[1] - t**2*g[2])) for t in range(0,60)])
    # range(int(min(Tdata)),int(max(Tdata)+1))])
    muT = jnp.array(
        [VAR[12]] * (T - 20) + [VAR[13]] * 7 + [VAR[14]] * 7 + [VAR[15]] * 6
    )
    muD = jnp.append(jnp.array([1]), jnp.array(VAR[16:22]))
    mu = muT[:, None] * muD[None, :]
    sig = VAR[-1]
    ######################
    # define all sales possibilities
    f0 = allDemand_jit(beta, bL, bB, gamma, mu)
    # define expected revenues
    ER0 = jnp.sum(
        f0 * jnp.array(range(qBar))[None, :, None, None, None] * \
            prices[None, None, None, :, None],
        axis = 1
    )
    # define loss associated with transition probabilities
    loss0 = jnp.sum(jnp.log(
        f0[data[:, 0], data[:, 1], data[:, 2], data[:, 3], data[:, 4]]
    ))
    # solve the DP, calculate the CCP and EV
    CCP0 = dynEst_jit(f0, ER0, gamma, sig, beta)
    # define loss for CCP
    loss1 = jnp.sum(CCP0[data[:, 2], data[:, 0], data[:, 3], data[:, 4]])
    ## SECOND FE
    return loss0 + loss1
