import math
import scipy.integrate as integrate
from scipy.stats import norm
from utils.stats import EPSILON


# LB = float("-inf")
# UB = float("inf")
LB = -3
UB = 3


def logprob_u(u, u_sd):
    return norm.logpdf(u, loc=0, scale=u_sd)


def logprob_x_u(u, x0, x1, x_sd):
    return norm.logpdf(x0, loc=u, scale=x_sd) + norm.logpdf(x1, loc=u**2, scale=x_sd)


def logprob_ux(u, x0, x1, u_sd, x_sd):
    return logprob_u(u, u_sd) + logprob_x_u(u, x0, x1, x_sd)


def logprob_x(x0, x1, u_sd, x_sd):
    def integrand(u):
        return max(math.exp(logprob_ux(u, x0, x1, u_sd, x_sd)), EPSILON)

    return math.log(integrate.quad(integrand, LB, UB)[0])


def mutual_info_u_x(u_sd, x_sd):
    def integrand(u, x0, x1):
        return math.exp(logprob_ux(u, x0, x1, u_sd, x_sd)) * (logprob_ux(u, x0, x1, u_sd, x_sd) - logprob_u(u, u_sd) -
            logprob_x(x0, x1, u_sd, x_sd))

    return integrate.tplquad(integrand, LB, UB, LB, UB, LB, UB)


print(mutual_info_u_x(1, 1)[0])