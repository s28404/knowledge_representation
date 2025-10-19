import tensorflow as tf
import argparse

def solve_linear_system(A, b):
    x = tf.linalg.solve(A, b)
    return tf.round(x * 1000) / 1000  # Round to 3 decimal places

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--A', type=str, default='2,1,1;1,3,2;1,0,0', help='Matrix A as a comma-separated string of rows; each row is a semicolon-separated string')
    parser.add_argument('--b', type=str, default='4,5,6', help='Vector b as a comma-separated string')
    args = parser.parse_args()

    A = tf.constant([[float(num) for num in row.split(',')] for row in args.A.split(';')], dtype=tf.float32)
    b = tf.constant([[float(num)] for num in args.b.split(',')], dtype=tf.float32)

    if A.shape[0] != A.shape[1]:
        print("Error: Matrix A must be square.")
        return

    if A.shape[0] != b.shape[0]:
        print("Error: Dimensions of A and b do not match.")
        return

    try:
        x = solve_linear_system(A, b)
        print("Solution x:")
        print(x.numpy())
    except tf.errors.InvalidArgumentError:
        print("Error: The system of equations has no solution.")

if __name__ == "__main__":
    main()