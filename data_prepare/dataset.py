import argparse

parser = argparse.ArgumentParser(description='test')

parser.add_argument('test', default=0, help='test parser')

args = parser.parse_args()

print(args.test)