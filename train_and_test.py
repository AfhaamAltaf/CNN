import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import os

# ========== SETUP ==========
EXPORT_DIR = "exported"
os.makedirs(EXPORT_DIR, exist_ok=True)
print("📂 Current working directory:", os.getcwd())
print(f"📦 Export folder set to: {EXPORT_DIR}")

# ========== 1. LOAD AND PREPROCESS ==========
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
x_train = x_train[..., tf.newaxis]  # shape (60000, 28, 28, 1)
x_test = x_test[..., tf.newaxis]

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# ========== 2. BUILD MODEL ==========
model = models.Sequential([
    layers.Conv2D(8, (5, 5), strides=(1, 1), padding='valid', activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D(pool_size=(2, 2), strides=2),
    layers.Flatten(),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', 'mae'])

# ========== 3. TRAIN ==========
history = model.fit(
    x_train, y_train,
    epochs=5,
    batch_size=32,
    validation_data=(x_test, y_test)
)

# ========== 4. SAVE MODEL ==========
model_path = os.path.join(EXPORT_DIR, 'trained_mnist_model.h5')
model.save(model_path)
print(f"✅ Model saved at: {model_path}")

# ========== 5. PLOT AND SAVE GRAPH ==========
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Test Acc')
plt.title("Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Train MAE')
plt.plot(history.history['val_mae'], label='Test MAE')
plt.title("Mean Absolute Error")
plt.xlabel("Epoch")
plt.ylabel("MAE")
plt.legend()

plot_path = os.path.join(EXPORT_DIR, "training_metrics.png")
plt.savefig(plot_path)
plt.close()
print(f"📊 Metrics graph saved to: {plot_path}")

# ========== 6. EXTRACT WEIGHTS ==========
model = tf.keras.models.load_model(model_path)

# Conv2D Layer
W_conv, b_conv = model.layers[0].get_weights()  # (5, 5, 1, 8), (8,)
W_conv = W_conv.squeeze(axis=2)                 # (5, 5, 8)
W_conv = np.transpose(W_conv, (2, 0, 1))        # (8, 5, 5)

# Dense Layer
W_fc, b_fc = model.layers[-1].get_weights()     # (1152, 10), (10,)
W_fc = W_fc.T                                    # (10, 1152)

# ========== 7. SAVE WEIGHTS TO .H FILES ==========
def save_array_as_c(name, arr):
    filename = os.path.join(EXPORT_DIR, f"{name}.h")
    with open(filename, 'w') as f:
        flat_arr = arr.flatten()
        f.write(f'#define {name.upper()}_SIZE {flat_arr.size}\n')
        f.write(f'float {name}[{flat_arr.size}] = {{\n')
        for i, val in enumerate(flat_arr):
            f.write(f'{val:.8f}f, ')
            if (i + 1) % 8 == 0:
                f.write('\n')
        f.write('\n};\n')
    print(f"✅ Saved: {filename}")

save_array_as_c('W_conv', W_conv)
save_array_as_c('b_conv', b_conv)
save_array_as_c('W_fc', W_fc)
save_array_as_c('b_fc', b_fc)

# Save one test image too
test_img = x_test[0]
save_array_as_c('input_img', test_img)

print("\n🎉 All outputs saved in 'exported/' folder")