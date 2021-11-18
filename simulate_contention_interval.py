import argparse
from math import log, ceil
from random import Random
from typing import Tuple, List

import matplotlib.pyplot as plt
import numpy as np

DEFAULT_STEP_LAMBDA = 0.01
DEFAULT_STOP_LAMBDA = 4.0
DEFAULT_START_LAMBDA = 1.0
DEFAULT_ITERATIONS: int = int(1e6)


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

    # Initial parameters discussed with Marcelo Ponce
    # t is initialized randomly via sample_interval
    # the first transmission has a greater likelihood if success since there is no prior transmission to contend with
    t_prev = float('-inf')
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
    print(f'λ={lambda_} --> {avg_contention_interval}')
    return avg_contention_interval


def find_optimal_lambda(
    iterations: int,
    seeded_random: Random,
    start_lambda: float,
    stop_lambda: float,
    step_lambda: float,
) -> Tuple[float, float]:
    """
    Run a contention interval simulation for each lambda value in the given range and return the minimum average
    contention interval and associated optimal lambda value. Plot the lambda values vs contention intervals.

    :param iterations: The number of times to simulate finding the contention interval for each lambda value
    :param seeded_random: An instance of random.Random used for running simulations
    :param start_lambda: The initial lambda value to simulate for
    :param stop_lambda: The final lambda value (inclusive) to simulate for
    :param step_lambda: The approximate step size between lambda values
    :return: A tuple of (optimal lambda value, minimum average contention interval)
    """
    # Run simulation for range of lambdas
    num_lambdas: int = ceil(((stop_lambda - start_lambda) / step_lambda))
    lambdas: List[float] = list(
        np.linspace(start=start_lambda, stop=stop_lambda, num=num_lambdas)
    )
    avg_contention_intervals: List[float] = [
        simulate_average_contention_interval(lambda_, iterations, seeded_random)
        for lambda_ in lambdas
    ]

    # identify optimal lambda and minimum interval
    i: int = np.argmin(avg_contention_intervals)
    optimal_lambda: float = lambdas[i]
    min_avg_contention_interval: float = avg_contention_intervals[i]

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
    Parse the arguments and perform basic validation

    :return: an argparse.Namespace containing the parsed arguments
    """
    parser: argparse.ArgumentParser = argparse.ArgumentParser()

    parser.add_argument(
        '-l',
        '--lambda',
        type=float,
        default=None,
        dest='lambda_',
        metavar='LAMBDA',
        help='λ parameter for to exponential distribution of intervals. Overrides all other lambda related options.',
    )
    parser.add_argument(
        '--start-lambda',
        type=float,
        default=DEFAULT_START_LAMBDA,
        help='The initial λ value to simulate for',
    )
    parser.add_argument(
        '--stop-lambda',
        type=float,
        default=DEFAULT_STOP_LAMBDA,
        help='The final λ value to simulate for',
    )
    parser.add_argument(
        '--step-lambda',
        type=float,
        default=DEFAULT_STEP_LAMBDA,
        help='The approximate step size between the λ values to simulate for',
    )
    parser.add_argument(
        '-i',
        '--iterations',
        type=int,
        default=DEFAULT_ITERATIONS,
        help='number of times to run the simulation for each λ value before averaging',
    )
    parser.add_argument(
        '-s',
        '--seed',
        type=int,
        default=None,
        help='seed for PRNG used to calculate random contention intervals',
    )

    args: argparse.Namespace = parser.parse_args()

    if args.lambda_ is not None:
        assert isinstance(args.lambda_, float)
    else:
        assert isinstance(args.start_lambda, float)
        assert isinstance(args.stop_lambda, float)
        assert isinstance(args.step_lambda, float)

    assert isinstance(args.iterations, int)
    assert args.seed is None or isinstance(args.seed, int)

    return args


def _create_random(seed: int) -> Random:
    """
    Create an instance of random.Random seeded with seed and advanced through 1248 iterations. Note that the Mersenne
    Twister pseudorandom number generator is used for all random operations

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

    if args.lambda_ is not None:
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
        # Find optimal lambda in range [start_lambda, stop_lambda] with granularity of step_lambda
        opt_lambda, avg_contention_interval = find_optimal_lambda(
            iterations=args.iterations,
            seeded_random=seeded_random,
            start_lambda=args.start_lambda,
            stop_lambda=args.stop_lambda,
            step_lambda=args.step_lambda,
        )
        print(
            f'The optimal lambda value is λ={opt_lambda:.4f} with an average contention interval of {avg_contention_interval:.4f}'
        )


if __name__ == '__main__':
    main()
