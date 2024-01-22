import data
import model
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

def visualize_predictions(images, masks, model, num_images=5):
    plt.figure(figsize=(10, num_images * 5))
    for i in range(num_images):
        img = images[i]
        true_mask = masks[i]
        predicted_mask = model.predict(np.expand_dims(img, axis=0))
        predicted_mask = np.argmax(predicted_mask, axis=-1)[0, :, :]

        plt.subplot(num_images, 3, i * 3 + 1)
        plt.imshow(img)
        plt.title("Original Image")

        plt.subplot(num_images, 3, i * 3 + 2)
        plt.imshow(np.argmax(true_mask, axis=-1))
        plt.title("True Mask")

        plt.subplot(num_images, 3, i * 3 + 3)
        plt.imshow(predicted_mask)
        plt.title("Predicted Mask")

    plt.tight_layout()
    plt.show()

annotations = data.load_annotations('resources/annotations.csv')
X_train, X_val, y_train, y_val = data.preprocess_data(annotations)

unet_model = model.build_unet_model((256, 256, 3))
unet_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks
checkpoint = ModelCheckpoint('best_model.h5', verbose=1, monitor='val_loss', save_best_only=True, mode='auto')
early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto')

# Train, evaluate, visualize the model
history = unet_model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    batch_size=32,
    epochs=50,
    callbacks=[checkpoint, early_stopping]
)
val_loss, val_accuracy = unet_model.evaluate(X_val, y_val)
print(f"Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}")

visualize_predictions(X_val, y_val, unet_model, num_images=5)
