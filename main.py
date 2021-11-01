import argparse


def simulate(lamb: float):
    print(lamb)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("lamb", type=float)
    args = parser.parse_args()
    simulate(args.lamb)
