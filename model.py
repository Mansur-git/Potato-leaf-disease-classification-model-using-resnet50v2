import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications import ResNet50V2
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import os

#  Enable memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

#  Define dataset paths``   
train_dir = r"PLD_3_Classes_256/Training"
val_dir = r"PLD_3_Classes_256/Validation"

#  Image Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.3,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1.0 / 255)

#  Load images
train_generator = train_datagen.flow_from_directory(
    train_dir, target_size=(256, 256), batch_size=16, class_mode="categorical", shuffle=True
)

val_generator = val_datagen.flow_from_directory(
    val_dir, target_size=(256, 256), batch_size=16, class_mode="categorical", shuffle=False
)

#  Compute class weights
labels = train_generator.classes
class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(labels), y=labels)
class_weights = dict(enumerate(class_weights))
print("Class Weights:", class_weights)

#  Load ResNet50V2
base_model = ResNet50V2(weights="imagenet", include_top=False, input_shape=(256, 256, 3))
base_model.trainable = False  # Freeze layers initially

#  Define Model
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(512, activation='relu'),
    Dropout(0.4),
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(3, activation='softmax')  # 3 classes
])

#  Compile the model
model.compile(optimizer=Adam(learning_rate=1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

#  Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3, min_lr=1e-6)

#  Train the model
history = model.fit(train_generator, validation_data=val_generator, epochs=30, 
                    class_weight=class_weights, callbacks=[early_stopping, reduce_lr], verbose=1)

# Unfreeze layers for fine-tuning
base_model.trainable = True
for layer in base_model.layers[:100]:  # Freeze first 100 layers
    layer.trainable = False

#  Recompile with a lower learning rate
model.compile(optimizer=Adam(learning_rate=1e-5),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

#  Fine-tune
history_fine = model.fit(train_generator, validation_data=val_generator, epochs=20,
                         class_weight=class_weights, callbacks=[early_stopping, reduce_lr], verbose=1)

#  Save model
model.save("potato_leaf_model_resnet50v2.h5")

print("Fine-tuned model saved successfully.")

#  Evaluate model
y_pred = model.predict(val_generator)
y_pred = np.argmax(y_pred, axis=1)
y_true = val_generator.classes

print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=val_generator.class_indices.keys()))

print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))
