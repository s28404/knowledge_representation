import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import argparse
from PIL import Image  

def plot(history):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Accuracy')
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.savefig('training_plot.png')
    plt.show()

def test_plot(model, x_test, y_test):
    indices = np.random.choice(len(x_test), 10)
    x_samples = x_test[indices]
    y_true = y_test[indices]

    y_pred = model.predict(x_samples)

    plt.figure(figsize=(15, 6))

    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.imshow(x_samples[i])
        plt.title(f"True: {y_true[i]}\nPred: {np.argmax(y_pred[i])}")

    plt.tight_layout()
    plt.savefig('test_predictions.png')
    plt.show()


def predict_custom_image(model, image_path):
    
    img = Image.open(image_path).convert('L')
    img = img.resize((28, 28))

    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1, 28, 28)

    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)

    plt.imshow(img)
    plt.title(f"Predicted: {predicted_class}")
    plt.tight_layout()
    plt.show()

    return predicted_class

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", type=str, required=True)
    args = parser.parse_args()
    mnist = tf.keras.datasets.mnist
    (x_train, y_train),(x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    if args.type == "train":
        
        model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),   
            tf.keras.layers.Dense(128, activation='relu'),   
            tf.keras.layers.Dropout(0.2),   
            tf.keras.layers.Dense(10, activation='softmax') ])
        model.compile(optimizer='adam',               
                    loss='sparse_categorical_crossentropy',               
                    metrics=['accuracy'])
        history = model.fit(x_train, y_train, epochs=5)
        model.evaluate(x_test, y_test)

        plot(history)

        model.save('mnist_model.h5')

    elif args.type == "test":
        model = tf.keras.models.load_model('mnist_model.h5')
        test_plot(model, x_test, y_test)
    
    elif args.type == "own":
        model = tf.keras.models.load_model('mnist_model.h5')
        image_path = "5.webp"
        predicted_class = predict_custom_image(model, image_path)
        print(f"The predicted class is: {predicted_class}")
        
if __name__ == "__main__":
    main()