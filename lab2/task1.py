import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--x', type=float, default=3.0, help='X coordinate')
    parser.add_argument('--y', type=float, default=4.0, help='Y coordinate')
    args = parser.parse_args()

    x = args.x
    y = args.y

    # Rotate by 45 degrees
    theta = np.pi / 4

    R = tf.stack([
        [tf.cos(theta), -tf.sin(theta)],
        [tf.sin(theta), tf.cos(theta)]
    ])

    point = tf.reshape(tf.constant([x, y]), (2, 1))
    rotated_point = tf.matmul(R, point)

    original_x, original_y = point[0, 0], point[1, 0]
    rotated_x, rotated_y = rotated_point[0, 0], rotated_point[1, 0]

    fig, (ax1, ax2)  = plt.subplots(1, 2, figsize=(10, 5))

    ax1.plot([0, original_x], [0, original_y], marker='^', color='b')
    ax1.set_title('Original Point')

    ax2.plot([0, rotated_x], [0, rotated_y], marker='^', color='r')
    ax2.set_title('Rotated Point')

    plt.savefig('zadanie1.png')

if __name__ == "__main__":
    main()