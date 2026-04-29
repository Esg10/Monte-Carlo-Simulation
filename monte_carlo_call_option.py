import numpy as np
import matplotlib.pyplot as plt


S0 = 100.0      # Initial stock price
K = 105.0       # Strike price
T = 1.0         # Time to maturity (years)
r = 0.05        # Risk-free interest rate
sigma = 0.2     # Volatility
N = 100_000     # Number of simulations


def price_european_call(S0, K, T, r, sigma, N, seed=None):
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(N)
    ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    payoffs = np.maximum(ST - K, 0.0)
    option_price = np.exp(-r * T) * payoffs.mean()
    return option_price, ST


def plot_distribution(ST, K, option_price):
    plt.figure(figsize=(10, 6))
    plt.hist(ST, bins=100, edgecolor="black", alpha=0.75)
    plt.axvline(K, color="red", linestyle="--", linewidth=2, label=f"Strike Price (K = {K})")
    plt.title(f"Distribution of Simulated Final Stock Prices (S_T)\nEstimated Call Option Price: {option_price:.4f}")
    plt.xlabel("Final Stock Price (S_T)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    option_price, ST = price_european_call(S0, K, T, r, sigma, N, seed=42)
    print(f"Estimated European Call Option Price: {option_price:.4f}")
    plot_distribution(ST, K, option_price)
