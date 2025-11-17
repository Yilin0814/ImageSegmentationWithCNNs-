import argparse

def get_args():
    parser = argparse.ArgumentParser(description='Model training options')

    parser.add_argument('-csv_dir', type=str, default='CSVs')

    parser.add_argument('-batch_size', type=int, default=16,
                        choices=[16, 32, 64])

    parser.add_argument('-lr', type=float, default=1e-3,)

    parser.add_argument('-wd', type=float, default=1e-5, )

    parser.add_argument('-epochs', type=float, default=10)

    parser.add_argument('-out_dir', type=str, default='session')

    args = parser.parse_args()

    return args