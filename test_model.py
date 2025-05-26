import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
import time
import argparse
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

# Chemin du modèle
MODEL_PATH = os.path.join(SAVE_DIR, "eye_state_model_final.h5")

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

def preprocess_image(image_path):
    """
    Prétraite une image pour la prédiction
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Impossible de charger l'image: {image_path}")
        
    # Conversion en niveaux de gris
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Redimensionnement à 24x24 pixels
    resized = cv2.resize(gray, (24, 24))
    
    # Normalisation
    normalized = resized / 255.0
    
    # Reshape pour le modèle
    normalized = normalized.reshape(1, 24, 24, 1)
    
    return img, normalized

def train_model():
    """
    Entraîne le modèle de détection d'état des yeux
    """
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
    model.save(MODEL_PATH)
    print(f"Modèle sauvegardé à {MODEL_PATH}")
    
    # Jouer un son pour indiquer que l'entraînement est terminé
    try:
        winsound.Beep(1000, 500)  # 1000 Hz for 500 ms
    except:
        print("\a")  # Fallback beep si winsound n'est pas disponible
    
    # Tester le modèle directement après l'entraînement
    test_model(model)

def test_model(model=None):
    """
    Teste le modèle sur les images de test
    """
    # Charger le modèle si non fourni
    if model is None:
        print(f"Chargement du modèle depuis {MODEL_PATH}...")
        try:
            # Essayer plusieurs méthodes pour charger le modèle
            try:
                model = load_model(MODEL_PATH, compile=False)
                print("Modèle chargé avec succès!")
            except Exception as e1:
                print(f"Erreur méthode 1: {e1}")
                try:
                    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
                    print("Modèle chargé avec succès (méthode alternative)!")
                except Exception as e2:
                    print(f"Erreur méthode 2: {e2}")
                    # Dernière tentative
                    keras_path = os.path.join(SAVE_DIR, 'eye_state_model_checkpoint.h5')
                    if os.path.exists(keras_path):
                        model = load_model(keras_path, compile=False)
                        print(f"Modèle chargé depuis le point de contrôle: {keras_path}")
        except Exception as e:
            print(f"Erreur lors du chargement du modèle: {e}")
    
    if model is None:
        print("Impossible de charger le modèle. Assurez-vous qu'il existe ou entraînez-le d'abord.")
        return
    
    # Tester sur quelques images du répertoire de test
    test_files = []
    if os.path.exists(TEST_PATH):
        # Récupérer quelques images de test (max 10)
        for root, dirs, files in os.walk(TEST_PATH):
            for file in files:
                if file.endswith(('.jpg', '.png', '.jpeg')):
                    test_files.append(os.path.join(root, file))
                    if len(test_files) >= 10:
                        break
            if len(test_files) >= 10:
                break
    
    if not test_files:
        print("Aucune image de test trouvée.")
        return
    
    # Prédire sur les images de test
    plt.figure(figsize=(15, 10))
    for i, image_path in enumerate(test_files):
        try:
            original_img, processed_img = preprocess_image(image_path)
            
            # Faire la prédiction
            prediction = model.predict(processed_img, verbose=0)[0]
            state = "Ouvert" if np.argmax(prediction) == 1 else "Fermé"
            confidence = prediction[np.argmax(prediction)] * 100
            
            # Afficher l'image et la prédiction
            plt.subplot(2, 5, i+1)
            plt.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
            plt.title(f"État: {state}\nConfiance: {confidence:.1f}%")
            plt.axis('off')
            
            print(f"Image {i+1}: {os.path.basename(image_path)} - État prédit: {state} (Confiance: {confidence:.1f}%)")
        except Exception as e:
            print(f"Erreur avec l'image {image_path}: {e}")
    
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "test_results.png"))
    plt.show()

def main():
    # Configuration des arguments
    parser = argparse.ArgumentParser(description='Détection de l\'état des yeux: entraînement et test')
    parser.add_argument('--mode', type=str, default='train',
                      help='Mode: "train" pour entraîner, "test" pour tester, "both" pour les deux')
    args = parser.parse_args()
    
    # Exécuter selon le mode choisi
    if args.mode.lower() == 'train':
        print("Mode: Entraînement du modèle")
        train_model()
    elif args.mode.lower() == 'test':
        print("Mode: Test du modèle")
        test_model()
    elif args.mode.lower() == 'both':
        print("Mode: Entraînement puis test du modèle")
        train_model()
    else:
        print(f"Mode inconnu: {args.mode}")
        print("Utilisez --mode train, --mode test ou --mode both")
        
if __name__ == "__main__":
    main()