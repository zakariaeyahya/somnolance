"""
Script de détection de somnolence du conducteur
Ce script utilise le modèle entraîné pour détecter si une personne est somnolente
en fonction de l'état des yeux (ouverts ou fermés).
"""

import os
import cv2
import numpy as np
import time
import argparse
import tensorflow as tf
from tensorflow.keras.models import load_model
import pygame

def load_model_safely(model_path):
    """
    Charge le modèle avec plusieurs méthodes pour éviter les erreurs
    """
    print(f"Chargement du modèle depuis {model_path}...")
    
    # Liste des différentes méthodes à essayer
    try:
        # Méthode 1: Standard
        model = load_model(model_path, compile=False)
        print("Modèle chargé avec succès!")
        return model
    except Exception as e1:
        print(f"Méthode 1 échouée: {e1}")
        
        try:
            # Méthode 2: Avec tf.keras
            model = tf.keras.models.load_model(model_path, compile=False)
            print("Modèle chargé avec succès (méthode alternative)!")
            return model
        except Exception as e2:
            print(f"Méthode 2 échouée: {e2}")
            
            # Vérifier si un checkpoint existe
            checkpoint_path = os.path.join(os.path.dirname(model_path), "eye_state_model_checkpoint.h5")
            if os.path.exists(checkpoint_path):
                try:
                    model = load_model(checkpoint_path, compile=False)
                    print(f"Modèle chargé depuis le point de contrôle!")
                    return model
                except Exception as e3:
                    print(f"Chargement du point de contrôle échoué: {e3}")
    
    print("Toutes les méthodes de chargement ont échoué.")
    return None

def preprocess_eye(eye_frame):
    """
    Prétraite une image d'œil pour la prédiction
    """
    # Redimensionner l'image à 24x24 pixels
    resized = cv2.resize(eye_frame, (24, 24))
    
    # Normaliser
    normalized = resized / 255.0
    
    # Reshape pour le modèle
    normalized = normalized.reshape(1, 24, 24, 1)
    
    return normalized

def detect_eyes(frame, face_cascade, eye_cascade):
    """
    Détecte les yeux dans l'image
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Détection des visages
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    eye_frames = []
    eye_positions = []
    face_count = len(faces)
    
    for (x, y, w, h) in faces:
        # Dessiner un rectangle autour du visage
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Région d'intérêt pour le visage
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        
        # Détecter les yeux dans le visage
        eyes = eye_cascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(20, 20)
        )
        
        for (ex, ey, ew, eh) in eyes:
            # Ne plus dessiner de rectangle autour des yeux
            # La ligne suivante a été supprimée:
            # cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
            
            # Extraire l'image de l'œil
            eye_frame = roi_gray[ey:ey+eh, ex:ex+ew]
            if eye_frame.size > 0:  # Vérifier que l'image n'est pas vide
                eye_frames.append(eye_frame)
                eye_positions.append((x+ex, y+ey, ew, eh))  # Position globale de l'œil
    
    return frame, eye_frames, eye_positions, face_count
def main():
    # Analyse des arguments
    parser = argparse.ArgumentParser(description='Détection de somnolence du conducteur')
    parser.add_argument('--model', type=str, default='saved_model/eye_state_model_final.h5',
                      help='Chemin vers le modèle entraîné (.h5)')
    parser.add_argument('--alarm', type=str, default='D:\\bureau\\BD&AI 1\\ci2\\S2\\droite\\one.wav',
                      help='Chemin vers le fichier audio d\'alarme (.wav ou .mp3)')
    parser.add_argument('--camera', type=int, default=0,
                      help='ID de la caméra à utiliser (par défaut: 0)')
    parser.add_argument('--threshold', type=float, default=2.5,
                      help='Seuil en secondes pour détecter la somnolence (par défaut: 2.5)')
    parser.add_argument('--score_threshold', type=int, default=15,
                      help='Seuil de score pour détecter la somnolence (par défaut: 15)')
    parser.add_argument('--use_score', action='store_true',
                      help='Utiliser la méthode de score au lieu du temps')
    
    args = parser.parse_args()
    
    # Résoudre le chemin du modèle
    model_path = args.model
    if not os.path.isabs(model_path):
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), model_path)
    
    # Initialisation de pygame pour l'audio
    pygame.mixer.init()
    sound = None
    
    # Chemin fixe vers votre fichier audio spécifique
    audio_file = "D:\\bureau\\BD&AI 1\\ci2\\S2\\droite\\one.wav"
    
    if os.path.exists(audio_file):
        try:
            sound = pygame.mixer.Sound(audio_file)
            print(f"Fichier audio d'alarme chargé: {audio_file}")
        except Exception as e:
            print(f"Erreur lors du chargement du fichier audio: {e}")
            print("Utilisation du beep système à la place.")
    else:
        print(f"Fichier audio non trouvé: {audio_file}")
        print("Utilisation du beep système à la place.")
    
    # Fonction pour jouer l'alarme
    def play_alarm():
        if sound is not None:
            if not pygame.mixer.get_busy():
                sound.play()
        else:
            # Utiliser le beep système comme fallback
            try:
                import winsound
                winsound.Beep(2500, 1000)  # Fréquence, durée en ms
            except:
                print("\a")  # Beep ASCII standard (peut ne pas fonctionner sur tous les systèmes)
    
    # Charger les classificateurs Haar Cascade
    face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    eye_cascade_path = cv2.data.haarcascades + 'haarcascade_eye.xml'
    
    if not os.path.exists(face_cascade_path):
        print(f"Erreur: Fichier cascade pour le visage non trouvé: {face_cascade_path}")
        return
    
    if not os.path.exists(eye_cascade_path):
        print(f"Erreur: Fichier cascade pour les yeux non trouvé: {eye_cascade_path}")
        return
    
    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    eye_cascade = cv2.CascadeClassifier(eye_cascade_path)
    
    # Charger le modèle
    model = load_model_safely(model_path)
    
    if model is None:
        print("Erreur critique: Impossible de charger le modèle.")
        return
    
    # Initialiser la webcam
    print(f"Ouverture de la caméra {args.camera}...")
    cap = cv2.VideoCapture(args.camera)
    
    if not cap.isOpened():
        print(f"Erreur: Impossible d'ouvrir la caméra {args.camera}")
        return
    
    # Variables pour la détection de somnolence
    eyes_closed_start = None
    is_alarm_active = False
    score = 0
    frame_counter = 0
    fps_start_time = time.time()
    fps = 0
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    print("Détection de somnolence démarrée. Appuyez sur 'q' pour quitter.")
    print(f"Utilisation du seuil de {args.threshold} secondes pour déclencher l'alarme audio.")
    print(f"Fichier audio configuré: {audio_file}")
    
    # Boucle principale
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Erreur: Impossible de capturer une image depuis la webcam.")
            break
        
        # Calculer le FPS
        frame_counter += 1
        if (time.time() - fps_start_time) > 1:
            fps = frame_counter / (time.time() - fps_start_time)
            frame_counter = 0
            fps_start_time = time.time()
        
        # Obtenir les dimensions de l'image
        height, width = frame.shape[:2]
        
        # Détecter les yeux
        processed_frame, eye_frames, eye_positions, face_count = detect_eyes(frame, face_cascade, eye_cascade)
        
        # Statut global des yeux (considérer fermé si tous les yeux détectés sont fermés)
        all_eyes_closed = True
        
        # Texte à afficher
        status_text = "Aucun œil détecté"
        
        if eye_frames:
            all_predictions = []
            
            for i, eye_frame in enumerate(eye_frames):
                # Prétraiter l'image de l'œil
                processed_eye = preprocess_eye(eye_frame)
                
                # Prédiction
                prediction = model.predict(processed_eye, verbose=0)[0]
                eye_state = np.argmax(prediction)  # 0: fermé, 1: ouvert
                confidence = prediction[eye_state] * 100
                
                all_predictions.append((eye_state, confidence))
                
                # Afficher l'état de chaque œil individuellement
                eye_x, eye_y, eye_w, eye_h = eye_positions[i]
                eye_status = "Ouvert" if eye_state == 1 else "Fermé"
                cv2.putText(processed_frame, f"{eye_status} {confidence:.0f}%", 
                            (eye_x, eye_y-5), font, 0.6, (255, 255, 0), 1, cv2.LINE_AA)
                
                # Si au moins un œil est ouvert, tous ne sont pas fermés
                if eye_state == 1:  # œil ouvert
                    all_eyes_closed = False
            
            # Méthode basée sur le temps
            if not args.use_score:
                # Afficher le statut en fonction de l'état des yeux
                if all_eyes_closed:
                    status_text = "Yeux FERMÉS"
                    status_color = (0, 0, 255)  # Rouge
                    
                    # Commencer à chronométrer si les yeux viennent de se fermer
                    if eyes_closed_start is None:
                        eyes_closed_start = time.time()
                        
                    # Vérifier si les yeux sont fermés depuis trop longtemps
                    elapsed_time = time.time() - eyes_closed_start if eyes_closed_start else 0
                    
                    # Afficher le temps écoulé
                    cv2.putText(processed_frame, f"Fermés depuis: {elapsed_time:.1f}s", 
                                (50, 80), font, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
                    
                    if elapsed_time > args.threshold:
                        if not is_alarm_active:
                            play_alarm()
                            is_alarm_active = True
                        
                        status_text = "ALERTE SOMNOLENCE!"
                        # Dessiner un cadre rouge autour de l'écran
                        cv2.rectangle(processed_frame, (0, 0), (width, height), (0, 0, 255), 3)
                else:
                    status_text = "Yeux OUVERTS"
                    status_color = (0, 255, 0)  # Vert
                    eyes_closed_start = None
                    is_alarm_active = False
            
            # Méthode basée sur le score
            else:
                if all_eyes_closed:
                    score += 1
                    status_text = "Yeux FERMÉS"
                    status_color = (0, 0, 255)  # Rouge
                else:
                    score = max(0, score - 1)  # Éviter les scores négatifs
                    status_text = "Yeux OUVERTS"
                    status_color = (0, 255, 0)  # Vert
                
                # Vérifier si le score dépasse le seuil
                if score > args.score_threshold:
                    if not is_alarm_active:
                        play_alarm()
                        is_alarm_active = True
                    
                    status_text = "ALERTE SOMNOLENCE!"
                    # Dessiner un cadre rouge autour de l'écran
                    cv2.rectangle(processed_frame, (0, 0), (width, height), (0, 0, 255), 3)
                else:
                    is_alarm_active = False
                
                # Afficher le score
                cv2.putText(processed_frame, f"Score: {score}", 
                            (50, 80), font, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
        else:
            status_color = (255, 255, 255)  # Blanc
            
            if not args.use_score:
                eyes_closed_start = None
            else:
                # Réduire progressivement le score si aucun œil n'est détecté
                score = max(0, score - 1)
            
            is_alarm_active = False
        
        # Zone d'information en bas de l'écran
        cv2.rectangle(processed_frame, (0, height-50), (width, height), (0, 0, 0), thickness=cv2.FILLED)
        
        # Informations sur le nombre de visages détectés
        if face_count == 0:
            cv2.putText(processed_frame, "Aucun visage détecté", (width//2 - 100, height-20), 
                       font, 0.7, (0, 255, 255), 1, cv2.LINE_AA)
        
        # Afficher le statut principal sur l'image
        cv2.putText(processed_frame, status_text, (50, 50), 
                    font, 1, status_color, 2, cv2.LINE_AA)
        
        # Afficher le FPS
        cv2.putText(processed_frame, f"FPS: {fps:.1f}", 
                    (width - 120, 30), font, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        
        # Afficher la méthode utilisée
        method_text = "Méthode: Score" if args.use_score else "Méthode: Temps"
        cv2.putText(processed_frame, method_text, 
                    (width - 200, height - 20), font, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Afficher le seuil utilisé
        if args.use_score:
            threshold_text = f"Seuil: {args.score_threshold} points"
        else:
            threshold_text = f"Seuil: {args.threshold}s"
        cv2.putText(processed_frame, threshold_text, 
                    (10, height - 20), font, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Afficher l'image traitée
        cv2.imshow('Détection de somnolence du conducteur', processed_frame)
        
        # Sortir si 'q' est pressé
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Libérer les ressources
    cap.release()
    cv2.destroyAllWindows()
    print("Programme terminé.")

if __name__ == "__main__":
    main()