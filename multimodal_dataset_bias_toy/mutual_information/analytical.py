import math
import scipy.integrate as integrate
from scipy.stats import norm
from utils.stats import EPSILON


U_SD = 1
X_SD = 1
U_LIM = (-3, 3)
X0_LIM = (-5, 5)
X1_LIM = (-3, 10)


def logprob_u(u, u_sd):
    return norm.logpdf(u, loc=0, scale=u_sd)


def logprob_x_u(u, x0, x1, x_sd):
    return norm.logpdf(x0, loc=u, scale=x_sd) + norm.logpdf(x1, loc=u ** 2, scale=x_sd)


def logprob_ux(u, x0, x1, u_sd, x_sd):
    return logprob_u(u, u_sd) + logprob_x_u(u, x0, x1, x_sd)


def logprob_x(x0, x1, u_sd, x_sd):
    def integrand(u):
        return math.exp(logprob_ux(u, x0, x1, u_sd, x_sd))

    prob_x = integrate.quad(integrand, *U_LIM)[0]
    return math.log(max(prob_x, EPSILON))


def mutual_info_u_x(u_sd, x_sd):
    def integrand(u, x0, x1):
        return math.exp(logprob_ux(u, x0, x1, u_sd, x_sd)) * (logprob_ux(u, x0, x1, u_sd, x_sd) - logprob_u(u, u_sd) -
            logprob_x(x0, x1, u_sd, x_sd))

    return integrate.tplquad(integrand, *X1_LIM, *X0_LIM, *U_LIM)


print(mutual_info_u_x(U_SD, X_SD)[0])