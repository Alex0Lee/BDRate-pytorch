import numpy as np
import scipy.interpolate


def BD_RATE(R1, PSNR1, R2, PSNR2, piecewise=0):
    """
    Calculates the Bjontegaard Delta Rate (BD-Rate) savings percentage between two encoding methods.

    Parameters:
    R1 (array-like): Bitrates for encoding method 1.
    PSNR1 (array-like): Peak Signal-to-Noise Ratios (PSNR) for encoding method 1.
    R2 (array-like): Bitrates for encoding method 2.
    PSNR2 (array-like): Peak Signal-to-Noise Ratios (PSNR) for encoding method 2.
    piecewise (int): Whether to use piecewise calculation or not, default is 0.

    Returns:
    avg_diff (float): The percentage of bitrate savings of encoding method 2 relative to encoding method 1.
    """
    # Take the logarithm of the bitrates
    lR1 = np.log(R1)
    lR2 = np.log(R2)

    # Fit polynomials to the relationship between PSNR and log(bitrate)
    p1 = np.polyfit(PSNR1, lR1, 3)
    p2 = np.polyfit(PSNR2, lR2, 3)

    # Determine the integration interval
    min_int = max(min(PSNR1), min(PSNR2))
    max_int = min(max(PSNR1), max(PSNR2))

    # Compute the integrals
    if piecewise == 0:
        # Use polynomial integration to compute the integrals
        p_int1 = np.polyint(p1)
        p_int2 = np.polyint(p2)
        int1 = np.polyval(p_int1, max_int) - np.polyval(p_int1, min_int)
        int2 = np.polyval(p_int2, max_int) - np.polyval(p_int2, min_int)
    else:
        # Use piecewise interpolation and the trapezoidal rule to compute the integrals
        lin = np.linspace(min_int, max_int, num=100, retstep=True)
        interval = lin[1]
        samples = lin[0]
        v1 = scipy.interpolate.pchip_interpolate(np.sort(PSNR1), np.sort(lR1), samples)
        v2 = scipy.interpolate.pchip_interpolate(np.sort(PSNR2), np.sort(lR2), samples)
        int1 = np.trapz(v1, dx=interval)
        int2 = np.trapz(v2, dx=interval)

    # Compute the average difference and return the result
    avg_exp_diff = (int2 - int1) / (max_int - min_int)
    avg_diff = (np.exp(avg_exp_diff) - 1) * 100
    return avg_diff
