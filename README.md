# Contention Interval Simulation

A Python script to simulate Ethernet contention intervals under different conditions.

## Getting Started

### Prerequisites

* [Python 3.9+](https://www.python.org/downloads/)
* [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/download.html) (optional)

### Installation

Navigate to the project root.

- #### Pip

  ```shell
  pip install -r requirements.txt
  ```

- #### Conda (alternative)

  ```shell
  conda env create -f environment.yaml
  conda activate networks
  ```

## Usage

```
usage: simulate_contention_interval.py [-h] [-l LAMBDA] [--start-lambda START_LAMBDA]
                                       [--stop-lambda STOP_LAMBDA]
                                       [--step-lambda STEP_LAMBDA] [-i ITERATIONS]
                                       [-s SEED]

optional arguments:
  -h, --help            show this help message and exit
  -l LAMBDA, --lambda LAMBDA
                        λ parameter for to exponential distribution of intervals. Overrides
                        all other lambda related options.
  --start-lambda START_LAMBDA
                        The initial λ value to simulate for
  --stop-lambda STOP_LAMBDA
                        The final λ value to simulate for
  --step-lambda STEP_LAMBDA
                        The approximate step size between the λ values to simulate for
  -i ITERATIONS, --iterations ITERATIONS
                        number of times to run the simulation for each λ value before
                        averaging
  -s SEED, --seed SEED  seed for PRNG used to calculate random contention intervals

```

### Examples

- See the help output and description of each option
  ```shell
  python simulate_contention_interval.py --help
  ```

- Compute the average of 10000000 simulations with λ=2.0 and 123456789 as the seed to the PRNG.
  ```shell
  python simulate_contention_interval.py \
    --lambda=2.0 \
    --seed=123456789 \
    --iterations=10000000  
  ```

- Compute the average of 10000000 simulations for every λ value within the interval [1.0, 4.0] with a step size for λ of
  approximately 0.01 and 123456789 as the seed to the PRNG. Show a plot of the results with the minimum contention
  interval and corresponding optimal λ value highlighted
  ```shell
  python simulate_contention_interval.py \
    --seed=123456789 \
    --iterations=10000000 \
    --start-lambda=1.0 \
    --stop-lambda=4.0 \
    --step-lambda=0.01
  ```
  Output Plot:

  ![Output Plot](images/contention_interval.png?raw=true "Output Plot")