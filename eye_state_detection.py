import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
import time
import winsound

# Chemins de données
BASE_PATH = r"D:\bureau\BD&AI 1\ci1\s4\algo\dataset_new"
TRAIN_PATH = os.path.join(BASE_PATH, "train")
TEST_PATH = os.path.join(BASE_PATH, "test")

# Chemins spécifiques pour les yeux
OPEN_EYES_PATH = os.path.join(TRAIN_PATH, "Open")
CLOSED_EYES_PATH = os.path.join(TRAIN_PATH, "Closed")

# Créer un dossier pour sauvegarder le modèle et les résultats
SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_model")
os.makedirs(SAVE_DIR, exist_ok=True)

def load_and_preprocess_images(directory, label):
    """
    Charge et prétraite les images du répertoire spécifié
    """
    images = []
    labels = []

    files = [f for f in os.listdir(directory) if f.endswith(('.jpg', '.png', '.jpeg'))]
    print(f"Chargement des images depuis {directory}: {len(files)} fichiers trouvés")

    for i, filename in enumerate(files):
        img_path = os.path.join(directory, filename)
        try:
            # Chargement et prétraitement de l'image
            img = cv2.imread(img_path)
            if img is None:
                print(f"Erreur de chargement de l'image: {img_path}")
                continue

            # Conversion en niveaux de gris
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Redimensionnement à 24x24 pixels
            resized = cv2.resize(gray, (24, 24))

            # Normalisation
            normalized = resized / 255.0

            images.append(normalized)
            labels.append(label)

            # Afficher la progression
            if i % 500 == 0 and i > 0:
                print(f"Chargées {i} images...")
        except Exception as e:
            print(f"Erreur lors du traitement de {filename}: {e}")

    return np.array(images), np.array(labels)

def create_model():
    """
    Crée et retourne le modèle CNN
    """
    model = Sequential()

    # Première couche de convolution
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(24, 24, 1)))
    model.add(MaxPooling2D((2, 2)))

    # Deuxième couche de convolution
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))

    # Troisième couche de convolution
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))

    # Aplatissement
    model.add(Flatten())

    # Couches entièrement connectées
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))  # 2 classes: yeux ouverts/fermés

    # Compilation du modèle
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def plot_training_history(history):
    """
    Affiche et sauvegarde les graphiques d'entraînement
    """
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Précision du modèle')
    plt.xlabel('Époque')
    plt.ylabel('Précision')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Perte du modèle')
    plt.xlabel('Époque')
    plt.ylabel('Perte')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'training_results.png'))
    plt.show()


def main():
    # Chargement des images avec les yeux ouverts
    print("Chargement des images avec les yeux ouverts...")
    open_images, open_labels = load_and_preprocess_images(OPEN_EYES_PATH, 1)  # 1 pour yeux ouverts

    # Chargement des images avec les yeux fermés
    print("Chargement des images avec les yeux fermés...")
    closed_images, closed_labels = load_and_preprocess_images(CLOSED_EYES_PATH, 0)  # 0 pour yeux fermés

    print(f"Images yeux ouverts: {open_images.shape}")
    print(f"Images yeux fermés: {closed_images.shape}")

    # Concaténation des données
    X = np.concatenate((open_images, closed_images))
    y = np.concatenate((open_labels, closed_labels))
    
    # Reshape pour le CNN
    X = X.reshape(-1, 24, 24, 1)
    y = to_categorical(y, 2)  # One-hot encoding (0: fermé, 1: ouvert)
    
    # Split des données
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Formes finales:")
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")

    # Création du modèle
    model = create_model()
    model.summary()

    # Callbacks
    checkpoint = ModelCheckpoint(
        os.path.join(SAVE_DIR, 'eye_state_model_checkpoint.h5'),
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )

    # Entraînement du modèle
    print("Début de l'entraînement...")
    history = model.fit(
        X_train, y_train,
        epochs=15,
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=[checkpoint, early_stopping]
    )

    # Évaluation du modèle
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Précision sur l'ensemble de test: {accuracy*100:.2f}%")

    # Visualisation des résultats
    plot_training_history(history)

    # Sauvegarde du modèle final
    model.save(os.path.join(SAVE_DIR, 'eye_state_model_final.h5'))
    print(f"Modèle sauvegardé à {os.path.join(SAVE_DIR, 'eye_state_model_final.h5')}")
if __name__ == "__main__":
    main()