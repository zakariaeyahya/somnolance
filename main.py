import cv2
import numpy as np
import time
import os
import logging
import argparse
from threading import Thread
import tensorflow as tf
from tensorflow.keras.models import load_model

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('drowsiness_detection')

# Constants
DEFAULT_CLOSED_EYES_THRESHOLD = 2.5  # secondes
FACE_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
EYE_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_eye.xml'


class DrowsinessDetector:
    """
    Classe pour la détection de somnolence basée sur la surveillance des yeux
    """
    def __init__(self, model_path, closed_eyes_threshold=DEFAULT_CLOSED_EYES_THRESHOLD, 
                 camera_id=0, alarm_sound=None):
        """
        Initialise le détecteur de somnolence
        
        Args:
            model_path (str): Chemin vers le modèle de classification des yeux
            closed_eyes_threshold (float): Seuil en secondes pour la détection de somnolence
            camera_id (int): ID de la caméra à utiliser
            alarm_sound (str, optional): Chemin vers le fichier son d'alarme
        """
        self.model_path = model_path
        self.closed_eyes_threshold = closed_eyes_threshold
        self.camera_id = camera_id
        self.alarm_sound = alarm_sound
        
        # État interne
        self.eyes_closed_start = None
        self.is_alarm_active = False
        self.model = None
        self.face_cascade = None
        self.eye_cascade = None
        self.cap = None
        
        # Mesure de performance
        self.fps = 0
        self.frame_counter = 0
        self.fps_start_time = time.time()
        
        # Initialisation des composants
        self._load_cascades()
        self._load_model()
        self._setup_camera()
    
    def _load_cascades(self):
        """Charge les classificateurs en cascade Haar pour la détection de visages et d'yeux"""
        logger.info("Chargement des cascades Haar...")
        self.face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
        self.eye_cascade = cv2.CascadeClassifier(EYE_CASCADE_PATH)
        
        if self.face_cascade.empty() or self.eye_cascade.empty():
            raise ValueError("Impossible de charger les fichiers de cascade Haar.")
    
    def _load_model(self):
        """Charge le modèle de classification des yeux"""
        logger.info(f"Chargement du modèle depuis {self.model_path}...")
        try:
            # Essayer de charger le modèle avec différentes méthodes
            try:
                self.model = load_model(self.model_path, compile=False)
                logger.info("Modèle chargé avec succès!")
            except Exception as e:
                logger.warning(f"Première tentative échouée: {e}")
                self.model = tf.keras.models.load_model(self.model_path, compile=False)
                logger.info("Modèle chargé avec la méthode alternative!")
        except Exception as e:
            logger.error(f"Impossible de charger le modèle: {e}")
            raise
    
    def _setup_camera(self):
        """Initialise la capture vidéo"""
        logger.info(f"Initialisation de la caméra (ID: {self.camera_id})...")
        self.cap = cv2.VideoCapture(self.camera_id)
        
        if not self.cap.isOpened():
            raise IOError(f"Impossible d'ouvrir la caméra avec l'ID {self.camera_id}")
    
    def detect_eyes(self, frame):
        """
        Détecte les yeux dans l'image
        
        Args:
            frame: Image capturée par la caméra
            
        Returns:
            tuple: (image traitée, cadres des yeux, positions des yeux)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        eye_frames = []
        eye_positions = []
        
        for (x, y, w, h) in faces:
            # Dessiner un rectangle autour du visage
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Région d'intérêt pour le visage
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            
            # Détecter les yeux dans le visage
            eyes = self.eye_cascade.detectMultiScale(roi_gray)
            
            for (ex, ey, ew, eh) in eyes:
                # Dessiner un rectangle autour des yeux
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
                
                # Extraire l'image de l'œil
                eye_frame = roi_gray[ey:ey+eh, ex:ex+ew]
                if eye_frame.size > 0:  # Vérifier que l'image n'est pas vide
                    eye_frames.append(eye_frame)
                    eye_positions.append((x+ex, y+ey, ew, eh))
        
        return frame, eye_frames, eye_positions
    
    def predict_eye_state(self, eye_frame):
        """
        Prédit si l'œil est ouvert ou fermé
        
        Args:
            eye_frame: Image de l'œil en niveaux de gris
            
        Returns:
            tuple: (état de l'œil, confiance)
        """
        try:
            # Redimensionner l'image à 24x24 pixels
            resized = cv2.resize(eye_frame, (24, 24))
            
            # Normaliser
            normalized = resized / 255.0
            
            # Reshape pour le modèle
            normalized = normalized.reshape(1, 24, 24, 1)
            
            # Prédiction
            prediction = self.model.predict(normalized, verbose=0)[0]
            
            # 0: fermé, 1: ouvert
            eye_state = np.argmax(prediction)
            confidence = prediction[eye_state] * 100
            
            return eye_state, confidence
        except Exception as e:
            logger.error(f"Erreur lors de la prédiction: {e}")
            return 1, 0  # Par défaut, considérer l'œil comme ouvert
    
    def play_alarm(self):
        """
        Joue un son d'alarme pour alerter l'utilisateur
        """
        # Version multi-plateforme pour l'alarme
        try:
            # Si un fichier son est spécifié et existe
            if self.alarm_sound and os.path.exists(self.alarm_sound):
                # Utiliser playsound si disponible (multi-plateforme)
                try:
                    from playsound import playsound
                    alarm_thread = Thread(target=playsound, args=(self.alarm_sound,))
                    alarm_thread.daemon = True
                    alarm_thread.start()
                    return
                except ImportError:
                    pass
            
            # Si playsound n'est pas disponible ou pas de fichier son spécifié
            # Utiliser winsound sur Windows
            if os.name == 'nt':
                try:
                    import winsound
                    winsound.Beep(2500, 1000)
                    return
                except ImportError:
                    pass
            
            # Méthode de repli pour Unix/Linux/Mac
            # Afficher un message d'alerte visuelle plus fort
            logger.warning("Système d'alarme audio non disponible - utilisation d'une alerte visuelle uniquement")
            
        except Exception as e:
            logger.error(f"Erreur lors de la lecture de l'alarme: {e}")
    
    def update_fps(self):
        """Met à jour le calcul des FPS"""
        self.frame_counter += 1
        if (time.time() - self.fps_start_time) > 1:
            self.fps = self.frame_counter / (time.time() - self.fps_start_time)
            self.frame_counter = 0
            self.fps_start_time = time.time()
    
    def process_frame(self, frame):
        """
        Traite une image pour la détection de somnolence
        
        Args:
            frame: Image capturée par la caméra
            
        Returns:
            Image traitée avec annotations
        """
        # Mise à jour des FPS
        self.update_fps()
        
        # Détecter les yeux
        processed_frame, eye_frames, eye_positions = self.detect_eyes(frame)
        
        # Statut global des yeux (considérer fermé si tous les yeux détectés sont fermés)
        all_eyes_closed = True
        
        # Texte à afficher
        status_text = "Aucun œil détecté"
        
        if eye_frames:
            for i, eye_frame in enumerate(eye_frames):
                eye_state, confidence = self.predict_eye_state(eye_frame)
                
                # Afficher l'état de chaque œil individuellement
                eye_x, eye_y, eye_w, eye_h = eye_positions[i]
                eye_status = "Ouvert" if eye_state == 1 else "Fermé"
                cv2.putText(processed_frame, f"{eye_status} {confidence:.0f}%", 
                            (eye_x, eye_y-5), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.6, (255, 255, 0), 1, cv2.LINE_AA)
                
                # Si au moins un œil est ouvert, tous ne sont pas fermés
                if eye_state == 1:  # œil ouvert
                    all_eyes_closed = False
            
            # Afficher le statut en fonction de l'état des yeux
            if all_eyes_closed:
                status_text = "Yeux FERMÉS"
                status_color = (0, 0, 255)  # Rouge
                
                # Commencer à chronométrer si les yeux viennent de se fermer
                if self.eyes_closed_start is None:
                    self.eyes_closed_start = time.time()
                    
                # Vérifier si les yeux sont fermés depuis trop longtemps
                elapsed_time = time.time() - self.eyes_closed_start if self.eyes_closed_start else 0
                if elapsed_time > self.closed_eyes_threshold:
                    if not self.is_alarm_active:
                        self.play_alarm()
                        self.is_alarm_active = True
                    
                    status_text = "ALERTE SOMNOLENCE!"
                    
                # Afficher le temps écoulé
                cv2.putText(processed_frame, f"Fermés depuis: {elapsed_time:.1f}s", 
                            (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.7, (0, 0, 255), 2, cv2.LINE_AA)
            else:
                status_text = "Yeux OUVERTS"
                status_color = (0, 255, 0)  # Vert
                self.eyes_closed_start = None
                self.is_alarm_active = False
        else:
            status_color = (255, 255, 255)  # Blanc
            self.eyes_closed_start = None
            self.is_alarm_active = False
        
        # Afficher le statut principal sur l'image
        cv2.putText(processed_frame, status_text, (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2, cv2.LINE_AA)
        
        # Afficher le FPS
        cv2.putText(processed_frame, f"FPS: {self.fps:.1f}", 
                    (processed_frame.shape[1] - 120, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, (0, 255, 255), 2, cv2.LINE_AA)
        
        return processed_frame
    
    def run(self):
        """
        Lance la détection de somnolence en temps réel
        """
        logger.info("Démarrage de la détection de somnolence en temps réel")
        logger.info("Appuyez sur 'q' pour quitter.")
        
        try:
            while True:
                ret, frame = self.cap.read()
                
                if not ret:
                    logger.error("Impossible de capturer une image depuis la webcam.")
                    break
                
                # Traiter l'image
                processed_frame = self.process_frame(frame)
                
                # Afficher l'image traitée
                cv2.imshow('Détection de somnolence', processed_frame)
                
                # Sortir si 'q' est pressé
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        except KeyboardInterrupt:
            logger.info("Interruption par l'utilisateur")
        except Exception as e:
            logger.error(f"Erreur inattendue: {e}")
        finally:
            # Libérer les ressources
            if self.cap:
                self.cap.release()
            cv2.destroyAllWindows()
            logger.info("Ressources libérées")


def parse_arguments():
    """
    Parse les arguments de ligne de commande
    
    Returns:
        Arguments parsés
    """
    parser = argparse.ArgumentParser(description='Système de détection de somnolence')
    parser.add_argument('--model', type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                                               "saved_model", "eye_state_model_final.h5"),
                        help='Chemin vers le modèle de classification des yeux')
    parser.add_argument('--threshold', type=float, default=DEFAULT_CLOSED_EYES_THRESHOLD,
                        help='Seuil en secondes pour détecter la somnolence')
    parser.add_argument('--camera', type=int, default=0,
                        help='ID de la caméra à utiliser')
    parser.add_argument('--alarm', type=str, default=None,
                        help='Chemin vers le fichier son d\'alarme')
    parser.add_argument('--debug', action='store_true',
                        help='Activer le mode debug')
    
    return parser.parse_args()


def main():
    """
    Fonction principale
    """
    args = parse_arguments()
    
    # Configurer le niveau de log
    if args.debug:
        logger.setLevel(logging.DEBUG)
    
    try:
        # Créer et exécuter le détecteur de somnolence
        detector = DrowsinessDetector(
            model_path=args.model,
            closed_eyes_threshold=args.threshold,
            camera_id=args.camera,
            alarm_sound=args.alarm
        )
        detector.run()
    
    except Exception as e:
        logger.critical(f"Erreur lors de l'exécution du programme: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())