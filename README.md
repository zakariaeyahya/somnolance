
# Projet de Détection de l'État des Yeux et de Somnolence

Ce projet vise à détecter si les yeux d'une personne sont ouverts ou fermés en utilisant un modèle de réseau de neurones convolutifs (CNN). Il est conçu pour être utilisé dans des applications de détection de somnolence, notamment pour les conducteurs.

## Structure du Projet

| Fichier | Description |
|---------|-------------|
| `eye_state_detection.py` | Script principal pour l'entraînement du modèle de détection de l'état des yeux. |
| `main.py` | Script pour exécuter la détection de somnolence en temps réel en utilisant le modèle entraîné. |
| `test_model.py` | Script pour entraîner et tester un modèle CNN pour détecter l'état des yeux. |
| `streamlit_app_updated.py` | Application Streamlit pour détecter la somnolence des conducteurs. |
| `script.py` | Script pour détecter la somnolence en utilisant un modèle pré-entraîné. |
| `services/` | Dossier contenant les différents services modulaires utilisés par l'application Streamlit. |

## Prérequis

Avant de commencer, assurez-vous d'avoir installé les bibliothèques suivantes :

- OpenCV
- NumPy
- Matplotlib
- TensorFlow
- scikit-learn
- Streamlit
- Pygame

Vous pouvez installer ces bibliothèques en utilisant pip :

```bash
pip install opencv-python numpy matplotlib tensorflow scikit-learn streamlit pygame
 

## Utilisation

### Entraînement du Modèle

Pour entraîner le modèle de détection de l'état des yeux, exécutez le script `eye_state_detection.py` :

```bash
python eye_state_detection.py
```

Ce script va charger les images des yeux ouverts et fermés, les prétraiter, entraîner le modèle CNN, et sauvegarder le modèle entraîné dans le dossier `saved_model`.

### Détection de Somnolence en Temps Réel

Pour exécuter la détection de somnolence en temps réel, utilisez le script `main.py` :

```bash
python main.py --model saved_model/eye_state_model_final.h5 --threshold 2.5 --camera 0 --alarm alarm_sound.wav
```

#### Arguments

| Argument | Description |
|----------|-------------|
| `--model` | Chemin vers le modèle de classification des yeux. Par défaut : `saved_model/eye_state_model_final.h5`. |
| `--threshold` | Seuil en secondes pour détecter la somnolence. Par défaut : `2.5`. |
| `--camera` | ID de la caméra à utiliser. Par défaut : `0`. |
| `--alarm` | Chemin vers le fichier son d'alarme. Par défaut : `None`. |
| `--debug` | Activer le mode debug. Par défaut : `False`. |

### Exécution de l'Application Streamlit

Pour exécuter l'application Streamlit, utilisez la commande suivante :

```bash
streamlit run streamlit_app_updated.py
```

### Exécution du Script de Test du Modèle

Pour exécuter le script de test du modèle, utilisez la commande suivante :

```bash
python test_model.py --mode train
```

#### Arguments

| Argument | Description |
|----------|-------------|
| `--mode` | Mode d'exécution : "train" pour entraîner le modèle, "test" pour tester le modèle, "both" pour entraîner puis tester le modèle. Par défaut : `train`. |

### Exécution du Script de Détection de Somnolence

Pour exécuter le script de détection de somnolence, utilisez la commande suivante :

```bash
python script.py --model saved_model/eye_state_model_final.h5 --alarm alarm_sound.wav --camera 0 --threshold 2.5
```

#### Arguments

| Argument | Description |
|----------|-------------|
| `--model` | Chemin vers le modèle entraîné (.h5). Par défaut : `saved_model/eye_state_model_final.h5`. |
| `--alarm` | Chemin vers le fichier audio d'alarme (.wav ou .mp3). Par défaut : `D:\\bureau\\BD&AI 1\\ci2\\S2\\droite\\one.wav`. |
| `--camera` | ID de la caméra à utiliser. Par défaut : `0`. |
| `--threshold` | Seuil en secondes pour détecter la somnolence. Par défaut : `2.5`. |
| `--score_threshold` | Seuil de score pour détecter la somnolence. Par défaut : `15`. |
| `--use_score` | Utiliser la méthode de score au lieu du temps pour la détection de somnolence. |

## Description des Fichiers

### `eye_state_detection.py`

Ce script contient les fonctions suivantes :

- `load_and_preprocess_images` : Charge et prétraite les images des yeux.
- `create_model` : Crée et retourne le modèle CNN.
- `plot_training_history` : Affiche et sauvegarde les graphiques d'entraînement.
- `main` : Fonction principale pour charger les images, entraîner le modèle, et sauvegarder les résultats.

### `main.py`

Ce script contient la classe `DrowsinessDetector` qui gère la détection de somnolence en temps réel. Il utilise le modèle entraîné pour prédire si les yeux sont ouverts ou fermés et déclenche une alarme si les yeux sont fermés pendant une durée supérieure au seuil spécifié.

### `test_model.py`

Ce script est conçu pour entraîner et tester un modèle de réseau de neurones convolutifs (CNN) pour détecter si les yeux sont ouverts ou fermés. Il utilise TensorFlow et Keras pour la création et l'entraînement du modèle, et OpenCV pour le traitement des images.

### `streamlit_app_updated.py`

Cette application Streamlit, nommée "Guardian Eye", est conçue pour détecter la somnolence des conducteurs en utilisant une architecture modulaire. Elle utilise des services modulaires pour une meilleure organisation et maintenabilité du code.

### `script.py`

Ce script est conçu pour détecter la somnolence d'un conducteur en utilisant un modèle pré-entraîné pour évaluer si les yeux sont ouverts ou fermés. Il utilise la bibliothèque OpenCV pour capturer des images en temps réel à partir d'une caméra et applique le modèle pour prédire l'état des yeux.

## Auteurs

- YAHYA ZAKARIAE

## Licence

Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.
```

 
