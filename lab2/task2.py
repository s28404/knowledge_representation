import tensorflow as tf
import argparse

def rotate(x, y, theta):
    x_hat = x * tf.cos(theta) - y * tf.sin(theta)
    y_hat = x * tf.sin(theta) + y * tf.cos(theta)
    return x_hat, y_hat

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--x', type=float, default=3.0, help='X coordinate')
    parser.add_argument('--y', type=float, default=4.0, help='Y coordinate')
    parser.add_argument('--theta', type=float, default=0.7854, help='Rotation angle in radians (default is 45 degrees)')
    args = parser.parse_args()

    # TensorFlow needs tensors for computation
    x = tf.constant(args.x, dtype=tf.float32)
    y = tf.constant(args.y, dtype=tf.float32)
    theta = tf.constant(args.theta, dtype=tf.float32)

    x_rotated, y_rotated = rotate(x, y, theta)

    print(f'Original Point: ({args.x}, {args.y})')
    print(f'Rotated Point: ({x_rotated.numpy()}, {y_rotated.numpy()})')

if __name__ == "__main__":
    main()