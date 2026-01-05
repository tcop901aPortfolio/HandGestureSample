import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0
train_labels = to_categorical(train_labels, 10)
test_labels = to_categorical(test_labels, 10)

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=10, batch_size=64, validation_data=(test_images, test_labels))

test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test accuracy: {test_acc * 100:.2f}%")

# Class names for CIFAR-10
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Function to test individual images
def test_random_images(num_images=5):
    import numpy as np
    indices = np.random.choice(len(test_images), num_images, replace=False)
    
    fig, axes = plt.subplots(1, num_images, figsize=(15, 3))
    if num_images == 1:
        axes = [axes]
    
    for i, idx in enumerate(indices):
        image = test_images[idx]
        true_label = np.argmax(test_labels[idx])
        
        # Make prediction
        prediction = model.predict(image.reshape(1, 32, 32, 3), verbose=0)
        predicted_label = np.argmax(prediction)
        confidence = np.max(prediction) * 100
        
        # Display image
        axes[i].imshow(image)
        axes[i].axis('off')
        color = 'green' if predicted_label == true_label else 'red'
        axes[i].set_title(f'True: {class_names[true_label]}\nPred: {class_names[predicted_label]}\n({confidence:.1f}%)', 
                         color=color, fontsize=10)
    
    plt.tight_layout()
    plt.show()

# Test with random images
print("\nTesting with random images from test set:")
test_random_images(10)

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.show()