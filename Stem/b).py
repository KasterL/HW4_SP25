# region imports
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.stats import lognorm
from scipy.integrate import quad
# endregion

# region functions
def truncated_lognormal_PDF(D, mu, sigma, D_min, D_max):
    """Calculate truncated log-normal PDF.
         :param D: Diameter at which to evaluate the PDF
        :param mu: Mean of the natural logarithm of diameters
        :param sigma: Standard deviation of the natural logarithm of diameters
        :param D_min: Minimum truncation diameter
        :param D_max: Maximum truncation diameter
        :return: Truncated log-normal PDF value at D
    """
    if D < D_min or D > D_max:
        return 0
    else:
        normalization = lognorm.cdf(D_max, s=sigma, scale=np.exp(mu)) - lognorm.cdf(D_min, s=sigma, scale=np.exp(mu))
        return lognorm.pdf(D, s=sigma, scale=np.exp(mu)) / normalization

def truncated_lognormal_CDF(D, mu, sigma, D_min, D_max):
    """Calculate truncated log-normal CDF using numerical integration."""
    result, _ = quad(truncated_lognormal_PDF, D_min, D, args=(mu, sigma, D_min, D_max))
    return result

def solicit_user_input():
    """
    Ask the user to input parameters for the log-normal distribution.
    Defaults are used if the user input is invalid or empty.
    """
    try:
        mu = float(input("Enter mu (default=0.693): ") or 0.693)
    except ValueError:
        print("Invalid input! Defaulting mu to 0.693")
        mu = 0.693

    try:
        sigma = float(input("Enter sigma (default=1.0): ") or 1.0)
    except ValueError:
        print("Invalid input! Defaulting sigma to 1.0")
        sigma = 1.0

    try:
        D_min = float(input("Enter D_min (default=0.375): ") or 0.375)
    except ValueError:
        print("Invalid input! Defaulting D_min to 0.375")
        D_min = 0.375

    try:
        D_max = float(input("Enter D_max (default=1.0): ") or 1.0)
    except ValueError:
        print("Invalid input! Defaulting D_max to 1.0")
        D_max = 1.0

    if D_min >= D_max:
        print("D_min should be smaller than D_max. Using defaults D_min=0.375, D_max=1.0")
        D_min, D_max = 0.375, 1.0

    return mu, sigma, D_min, D_max
# endregion

# region main function
def main():
    # Ask user for parameters
    mu, sigma, D_min, D_max = solicit_user_input()

    # Set integration limit (75% between D_min and D_max)
    c = 0.75
    D_limit = D_min + c * (D_max - D_min)

    # Generate values for plotting
    D_vals = np.linspace(D_min, D_max, 500)
    pdf_vals = np.array([truncated_lognormal_PDF(D, mu, sigma, D_min, D_max) for D in D_vals])
    cdf_vals = np.array([truncated_lognormal_CDF(D, mu, sigma, D_min, D_max) for D in D_vals])

    # Setup plots
    fig, (ax_pdf, ax_cdf) = plt.subplots(nrows=2, ncols=1, figsize=(8, 8), sharex=True)

    # Plot PDF and fill area under curve
    ax_pdf.plot(D_vals, pdf_vals, label='PDF')
    ax_pdf.fill_between(D_vals, pdf_vals, where=(D_vals <= D_limit), color='grey', alpha=0.5)
    ax_pdf.set_ylabel('f(D)')
    ax_pdf.set_title('Truncated Log-Normal PDF')

    # Calculate probability for annotation
    probability = truncated_lognormal_CDF(D_limit, mu, sigma, D_min, D_max)

    # Annotate probability
    ax_pdf.annotate(f'P(D<{D_limit:.2f})[TLN({mu:.2f},{sigma:.2f},{D_min:.2f},{D_max:.2f})]={probability:.2f}',
                    xy=(D_limit, truncated_lognormal_PDF(D_limit, mu, sigma, D_min, D_max)),
                    xytext=(D_limit, max(pdf_vals) * 0.8),
                    arrowprops=dict(facecolor='black', arrowstyle='->'))

    # Plot CDF
    ax_cdf.plot(D_vals, cdf_vals, label='CDF', color='blue')
    ax_cdf.axhline(probability, color='black', linestyle='--')
    ax_cdf.axvline(D_limit, color='black', linestyle='--')
    ax_cdf.scatter(D_limit, probability, color='red')
    ax_cdf.set_xlabel('Diameter D')
    ax_cdf.set_ylabel('CDF')
    ax_cdf.set_title('Truncated Log-Normal CDF')

    plt.tight_layout()
    plt.show()
# endregion

# region entry point
if __name__ == '__main__':
    main()
# endregion