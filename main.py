import argparse
from math import log
from random import Random
from typing import Tuple, List

import matplotlib.pyplot as plt
import numpy as np


def simulate_first_successful_transmission_time(
    lambda_: float,
    seeded_random: Random,
) -> float:
    """
    Simulate attempting to transmit packets at random intervals sampled from an exponential distribution
    with parameter lambda_ until a packet transmission is deemed to be successful by not colliding with
    any other packet within a [t-1, t+1] time interval

    :param lambda_: The parameter for the exponential distribution from which to sample the transmission intervals
    :param seeded_random: An instance of random.Random used for running simulations
    :return: The amount of simulated time in arbitrary units it took until the first successful packet transmission
    """

    def sample_interval():
        return -lambda_ * log(seeded_random.uniform(0.0, 1.0))

    t_prev = 0
    t = sample_interval()
    t_next = t + sample_interval()

    while t - t_prev < 1 or t_next - t < 1:
        t_prev = t
        t = t_next
        t_next += sample_interval()
    return t


def simulate_average_contention_interval(
    lambda_: float,
    iterations: int,
    seeded_random: Random,
) -> float:
    """
    Run a simulation to determine the average contention interval for the given lambda value

    :param lambda_: The lambda value for which to run the contention interval simulation
    :param iterations: The number of trials to use to determine the average contention interval
    :param seeded_random: An instance of random.Random used for running the simulation
    :return: the average contention interval for the given lambda value
    """
    avg_contention_interval = np.average(
        [
            simulate_first_successful_transmission_time(lambda_, seeded_random)
            for _ in range(iterations)
        ]
    )
    print(lambda_, avg_contention_interval)
    return avg_contention_interval


def find_optimal_lambda(
    iterations: int,
    seeded_random: Random,
    start_lambda=1.0,
    stop_lambda=4.001,
    step_lambda=0.01,
    plot: bool = True,
) -> Tuple[float, float]:
    """
    Run a contention interval simulation for each lambda value in the given range and return the minimum average
    contention interval and associated optimal lambda value

    :param iterations: The number of times to simulate finding the contention interval for each lambda value
    :param seeded_random: An instance of random.Random used for running simulations
    :param start_lambda: The initial lambda value to simulate for
    :param stop_lambda: The final lambda value (exclusive) to simulate for
    :param step_lambda: The step size between lambda values
    :param plot: A boolean value indicating whether or not to plot the lambda values vs contention intervals
    :return: A tuple of (optimal lambda value, minimum average contention interval)
    """
    # Run simulation for range of lambdas
    lambdas: List[float] = list(
        np.arange(start=start_lambda, stop=stop_lambda, step=step_lambda)
    )
    avg_contention_intervals: List[float] = [
        simulate_average_contention_interval(lambda_, iterations, seeded_random)
        for lambda_ in lambdas
    ]

    # identify optimal lambda and minimum interval
    i: int = np.argmin(avg_contention_intervals)
    optimal_lambda: float = lambdas[i]
    min_avg_contention_interval: float = avg_contention_intervals[i]

    if plot:
        # plot simulation and minimum interval
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.plot(lambdas, avg_contention_intervals, 'r-')
        plt.plot(optimal_lambda, min_avg_contention_interval, 'bo')
        ax.annotate(
            f'Optimal λ: {optimal_lambda:.4f}\nMinimum Contention Interval: {min_avg_contention_interval:.4f}',
            xy=(optimal_lambda, min_avg_contention_interval),
            xytext=(10, -25),
            textcoords='offset points',
        )
        ax.set_ylim(bottom=min_avg_contention_interval - 0.4)
        plt.xlabel('λ')
        plt.ylabel('Contention interval (Arbitrary Units)')
        plt.show()

    return optimal_lambda, min_avg_contention_interval


def _get_args() -> argparse.Namespace:
    """
    Parse the arguments and perform datatype validation

    :return: an argparse.Namespace containing the parsed arguments
    """
    parser: argparse.ArgumentParser = argparse.ArgumentParser()

    parser.add_argument(
        '-l',
        '--lambda',
        type=float,
        default=None,
        dest='lambda_',
        help='λ parameter for to exponential distribution of intervals',
    )
    parser.add_argument(
        '-i',
        '--iterations',
        type=int,
        default=int(1e6),
        help='number of times to run the simulation',
    )
    parser.add_argument(
        '-s',
        '--seed',
        type=int,
        default=None,
        help='seed for random intervals',
    )

    args: argparse.Namespace = parser.parse_args()

    assert args.lambda_ is None or isinstance(args.lambda_, float)
    assert isinstance(args.iterations, int)
    assert args.seed is None or isinstance(args.seed, int)

    return args


def _create_random(seed: int) -> Random:
    """
    Create an instance of random.Random seeded with seed and advanced through 1248 iterations

    :param seed: The seed to the Mersenne Twister pseudorandom number generator
    :return: An instance of random.Random seeded with seed and advanced through 1248 iterations
    """
    seeded_random: Random = Random()
    seeded_random.seed(seed)

    # throw away first 624*2 outputs of Mersenne Twister as recommended in https://stats.stackexchange.com/a/438057
    for _ in range(1248):
        seeded_random.random()

    return seeded_random


def main():
    """
    Parse the input arguments and initiate the simulation
    """
    args: argparse.Namespace = _get_args()

    seeded_random = _create_random(args.seed)

    if args.lambda_:
        # Simulate given value of lambda
        avg_contention_interval = simulate_average_contention_interval(
            lambda_=args.lambda_,
            iterations=args.iterations,
            seeded_random=seeded_random,
        )
        print(
            f'The average contention interval for λ={args.lambda_:.4f} is {avg_contention_interval:.4f}'
        )
    else:
        # Find optimal lambda in range [1,4]
        opt_lambda, avg_contention_interval = find_optimal_lambda(
            iterations=args.iterations,
            seeded_random=seeded_random,
            plot=True,
        )
        print(
            f'The optimal lambda value is λ={opt_lambda:.4f} with an average contention interval of {avg_contention_interval:.4f}'
        )


if __name__ == '__main__':
    main()
