"""
Main Program.

Decrypt RSA private key using Correlation Power Analysis.
"""

import numpy as np
from typing import Iterable, Tuple
from functools import reduce
import os

# Project Parameters
ASSETS_PATH: str = os.path.join(os.path.dirname(__file__), "assets")
MODULO_PATH: str = os.path.join(ASSETS_PATH, "N.txt")
CURVE_FILE_NAMING_PATTERN: str = "curve_{:d}.txt"
PLAINTEXT_FILE_NAMING_PATTERN: str = "msg_{:d}.txt"
PUBLIC_KEY: int = 65537

with open(MODULO_PATH, "r") as file:
    N = int(file.readline())


class NumberUtils:
    """Utility class that handles various operations on numbers."""

    @staticmethod
    def hamming_weight(number: int) -> int:
        """Calculate the Hamming distance from 0."""
        return bin(number).count("1")

    @staticmethod
    def prime_factors(n) -> Iterable[int]:
        """Compute the prime factors of the given number."""
        i = 2
        while i * i <= n:
            if n % i:
                i += 1
            else:
                n //= i
                yield i
        if n > 1:
            yield n

    @staticmethod
    def mod_inverse(a: int, b: int) -> int:
        """Apply the extended euclidean algorithm to find the gcd between a and ."""
        if b <= 1:
            raise ZeroDivisionError
        old_r, r = a, b
        old_t, t = 1, 0

        while old_r > 1:
            if r == 0:
                raise ValueError("/ by zero: not coprime")

            q = old_r // r
            old_r, r = r, old_r - q * r
            old_t, t = t, old_t - q * t

        return old_t if (old_t >= 0) else b - old_t

    @staticmethod
    def hamming_weight_for_rsa(
        data: int, exponent_bin: Iterable[int], n: int
    ) -> int:
        """Calculate key with Square-And-Multiply, and compute Hamming Weight."""
        result = 1
        for bit in reversed(exponent_bin):
            result = (result * result) % n  # Square
            if bit == 1:
                result = (result * data) % n  # Multiply

        if exponent_bin[0] == 0:  # Assume squaring if hypothesis is 0.
            result = (result * result) % n
        return NumberUtils.hamming_weight(result)

    @staticmethod
    def corr(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Compute the correlation matrix using Pearson coefficient."""
        A = (A - A.mean(axis=0)) / A.std(axis=0)
        B = (B - B.mean(axis=0)) / B.std(axis=0)
        return ((B.T).dot(A) / B.shape[0])[0]

    @staticmethod
    def bitListToInt(bits: Iterable[int]) -> int:
        """Convert a list of bit to an integer."""
        return reduce(lambda acc, value: acc * 2 + value, bits)


def load_power_consumptions(
    number_of_trace: int = 1000, number_of_points_per_trace: int = 36
) -> np.ndarray:
    """Load the traces stored in files."""
    power_consumptions = np.zeros(
        (number_of_trace, number_of_points_per_trace)
    )
    for i in range(number_of_trace):
        filename = CURVE_FILE_NAMING_PATTERN.format(i)
        pathfile = os.path.join(ASSETS_PATH, filename)

        with open(pathfile, "r") as file:
            # Parse a line of float and ignore -1000.0
            data_iter = filter(
                lambda x: x != -1000.0,
                map(float, file.readline().rstrip().split()),
            )
            power_consumptions[i][:] = np.fromiter(data_iter, float)
    return power_consumptions


def fetch_plaintexts(number_of_plaintext: int = 1000) -> Iterable[int]:
    """Load the plaintexts."""
    for i in range(number_of_plaintext):
        filename = PLAINTEXT_FILE_NAMING_PATTERN.format(i)
        pathfile = os.path.join(ASSETS_PATH, filename)

        with open(pathfile, "r") as file:
            yield int(file.readline().rstrip())


def compute_hypothesis_matrix(
    plaintexts: Tuple[int], old_key: Tuple[int]
) -> np.ndarray:
    """Return the hypothesis matrix.

    Since we are "building" the bits of the key, we need the old_key.
    Based on the old_key, the following hypothetical bit is either 0 or 1.
    Therefore, the hypothesis matrix is of shape (len(plaintexts), 2).
    """
    hypothesis_matrix = np.zeros((len(plaintexts), 2))
    for (plaintext_idx, plaintext) in enumerate(plaintexts):
        for hypothesis_bit in (0, 1):
            hypothesis_key = [hypothesis_bit] + old_key
            hypothesis_matrix[
                plaintext_idx, hypothesis_bit
            ] = NumberUtils.hamming_weight_for_rsa(
                plaintext, hypothesis_key, N
            )
    return hypothesis_matrix


def compute_key_by_factoring() -> int:
    """Calculate the private key using prime number factoring."""
    p, q = NumberUtils.prime_factors(N)
    totient = (p - 1) * (q - 1)
    return NumberUtils.mod_inverse(PUBLIC_KEY, totient)


def compute_key_by_cpa(
    number_of_plaintext=1000, number_of_points_per_trace=36
) -> int:
    """Calculate the private key using CPA."""
    plaintexts = tuple(
        fetch_plaintexts(number_of_plaintext=number_of_plaintext)
    )
    power_consumptions = load_power_consumptions(
        number_of_trace=number_of_plaintext,
        number_of_points_per_trace=number_of_points_per_trace,
    )  # shape (1000, 36)

    key_builder = [1]  # Key start with 1 for obvious reasons
    time = 1  # Skip 0, since start with existing value
    while time < number_of_points_per_trace:
        hypothesis_matrix = compute_hypothesis_matrix(
            plaintexts, key_builder
        )  # shape (1000, 2)
        power_consumption_of_bit = power_consumptions[
            :, time : time + 1
        ]  # shape (1000, 1)
        correlation_matrix = NumberUtils.corr(
            hypothesis_matrix, power_consumption_of_bit
        )  # shape (2,)

        best_hypothesis = np.argmax(correlation_matrix)

        key_builder.insert(0, best_hypothesis)  # .prepend
        if best_hypothesis == 1:
            time += 1  # Skip Multiply
        time += 1

    return NumberUtils.bitListToInt(reversed(key_builder))


if __name__ == "__main__":
    expected = compute_key_by_factoring()
    result = compute_key_by_factoring()
    print(f"Factoring : {expected:b}")
    print(f"CPA : {result:b}")
    print(f"Equality : {result == expected}")