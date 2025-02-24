#region imports
import math
import numpy as np
from random import random as rnd
from scipy.integrate import quad
from scipy.optimize import fsolve
from matplotlib import pyplot as plt
#endregion

# region functions
def ln_PDF(args):
    """
    Computes f(D) for the log-normal probability density function.
    :param args: (D-diameter, mu:mean of ln(D), sigma-Stdev of ln(D))
    :return: value of PDF at diameter D
    """
    D, mu, sig = args
    if D == 0.0:
        return 0.0
    p = 1/(D*sig*math.sqrt(2*math.pi))
    _exp = -((math.log(D)-mu)**2)/(2*sig**2)
    return p*math.exp(_exp)

def tln_PDF(args):
    """
    Compute the value of the truncated log-normal probability density function.
    :param args: (D, mu, sig, F_DMin-min truncation diameter, F_DMax-max truncation diameter)
    :return: value of the truncated PDF at D
    """
    D, mu, sig, F_DMin, F_DMax = args
    return ln_PDF((D, mu, sig)) / (F_DMax - F_DMin)

def F_tlnpdf(args):
    """
    Numerically integrate the truncated log-normal PDF from D_min to D
    :param args: (mu, sig, D_Min, D_Max, D, F_DMax, F_DMin)
                mu: Mean of ln(D)
                 sig: Standard deviation of ln(D)
                 D_min: Lower integration limit
                 D_max: Upper integration limit
                 D: Target diameter
                 F_DMin: Minimum truncation diameter
                 F_DMax: Maximum truncation diameter
    :return: Probability value
    """
    mu, sig, D_Min, D_Max, D, F_DMax, F_DMin = args
    if D > D_Max or D < D_Min:
        return 0
    integral, _ = quad(lambda D: tln_PDF((D, mu, sig, F_DMin, F_DMax)), D_Min, D)
    return integral

def find_D(target_prob, args):
    """
    Finds the rock diameter D that satisfies the probability condition
    :param target_prob: Probability value to match
    :param args: (ln_Mean, ln_sig, D_Min, D_Max, F_DMax, F_DMin)
    :return: Computed diameter D
    """
    func = lambda D: F_tlnpdf(args[:-2] + (D,) + args[-2:]) - target_prob
    solution = fsolve(func, x0=(args[2] + args[3]) / 2)  # Initial guess is midpoint of (D_Min, D_Max)
    return solution[0]

def makeSample(args, N=100):
    """
    Generates a sample of N rock sizes from the truncated log-normal distribution.
    :param args: (ln_Mean, ln_sig, D_Min, D_Max, F_DMax, F_DMin)
    :param N: Number of samples to generate
    :return: List of generated rock sizes
    """
    ln_Mean, ln_sig, D_Min, D_Max, F_DMax, F_DMin = args
    probs = [rnd() for _ in range(N)]
    d_s = [find_D(probs[i], (ln_Mean, ln_sig, D_Min, D_Max, F_DMax, F_DMin)) for i in range(N)]
    return d_s
# endregion

# region main
def main():
    """
    Simulates a gravel production process where rock sizes follow a truncated log-normal distribution.
    """
    mean_ln = math.log(2)  # Default mean of ln(D), in inches
    sig_ln = 1  # Default standard deviation
    D_Max = 1  # Default max diameter
    D_Min = 3.0/8.0  # Default min diameter
    N_samples = 11  # Number of samples
    N_sampleSize = 100  # Number of items per sample

    mean_ln, sig_ln = mean_ln, sig_ln  # Keep default values for now
    D_Min, D_Max = D_Min, D_Max
    N_samples, N_sampleSize = N_samples, N_sampleSize
    F_DMin, F_DMax = F_tlnpdf((mean_ln, sig_ln, D_Min, D_Max, D_Min, 1, 0)), F_tlnpdf((mean_ln, sig_ln, D_Min, D_Max, D_Max, 1, 0))

    Samples = []
    Means = []
    for n in range(N_samples):
        sample = makeSample((mean_ln, sig_ln, D_Min, D_Max, F_DMax, F_DMin), N=N_sampleSize)
        Samples.append(sample)
        mean_D = sum(sample) / N_sampleSize
        variance_D = sum((x - mean_D) ** 2 for x in sample) / (N_sampleSize - 1)
        Means.append(mean_D)
        print(f"Sample {n+1}: Mean = {mean_D:.4f}, Variance = {variance_D:.4f}")

    grand_mean = sum(Means) / N_samples
    grand_variance = sum((x - grand_mean) ** 2 for x in Means) / (N_samples - 1)

    print("\nFinal Statistics:")
    print(f"Mean of Sample Means: {grand_mean:.4f}")
    print(f"Variance of Sample Means: {grand_variance:.6f}")

if __name__ == "__main__":
    main()
# endregion