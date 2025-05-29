"""
Service de détection des yeux et analyse de somnolence
Traitement d'images et détection en temps réel
"""

import cv2
import numpy as np
import time
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class DetectionService:
    """Service pour la détection des yeux et analyse de somnolence"""
    
    def __init__(self, model_service):
        self.model_service = model_service
        
        # Configuration de détection
        self.face_detection_params = {
            'scaleFactor': 1.3,
            'minNeighbors': 5,
            'minSize': (50, 50)
        }
        
        self.eye_detection_params = {
            'scaleFactor': 1.1,
            'minNeighbors': 10,
            'minSize': (20, 20)
        }
    
    def detect_faces_and_eyes(self, frame):
        """
        Détecte les visages et les yeux dans une image
        
        Args:
            frame: Image d'entrée
            
        Returns:
            tuple: (faces, eyes_data) où eyes_data contient position et état
        """
        if not self.model_service.cascades_loaded:
            logger.warning("Cascades non chargées")
            return [], []
        
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Détection des visages
            faces = self.model_service.face_cascade.detectMultiScale(
                gray, 
                **self.face_detection_params
            )
            
            eyes_data = []
            
            for (x, y, w, h) in faces:
                # Région d'intérêt pour le visage
                roi_gray = gray[y:y+h, x:x+w]
                
                # Détection des yeux dans le visage
                eyes = self.model_service.eye_cascade.detectMultiScale(
                    roi_gray,
                    **self.eye_detection_params
                )
                
                face_eyes = []
                for (ex, ey, ew, eh) in eyes:
                    # Extraire l'image de l'œil
                    eye_frame = roi_gray[ey:ey+eh, ex:ex+ew]
                    
                    if eye_frame.size > 0:
                        # Prédire l'état de l'œil
                        eye_state, confidence = self.model_service.predict_eye_state(eye_frame)
                        
                        eye_info = {
                            'position': (x + ex, y + ey, ew, eh),  # Position globale
                            'relative_position': (ex, ey, ew, eh),  # Position relative au visage
                            'state': eye_state,  # 0: fermé, 1: ouvert
                            'confidence': confidence,
                            'face_region': (x, y, w, h)
                        }
                        face_eyes.append(eye_info)
                
                eyes_data.extend(face_eyes)
            
            return faces, eyes_data
            
        except Exception as e:
            logger.error(f"Erreur détection visages/yeux: {e}")
            return [], []
    
    def analyze_drowsiness_state(self, eyes_data):
        """
        Analyse l'état de somnolence basé sur les données des yeux
        
        Args:
            eyes_data: Liste des données des yeux détectés
            
        Returns:
            dict: Analyse de l'état de somnolence
        """
        if not eyes_data:
            return {
                'eyes_detected': False,
                'total_eyes': 0,
                'closed_eyes': 0,
                'open_eyes': 0,
                'all_eyes_closed': False,
                'any_eye_closed': False,
                'average_confidence': 0,
                'drowsiness_level': 0
            }
        
        total_eyes = len(eyes_data)
        closed_eyes = sum(1 for eye in eyes_data if eye['state'] == 0)
        open_eyes = total_eyes - closed_eyes
        
        # Calcul des statistiques
        all_eyes_closed = closed_eyes == total_eyes and total_eyes > 0
        any_eye_closed = closed_eyes > 0
        
        # Confiance moyenne
        total_confidence = sum(eye['confidence'] for eye in eyes_data)
        average_confidence = total_confidence / total_eyes if total_eyes > 0 else 0
        
        # Niveau de somnolence (0-100%)
        if total_eyes == 0:
            drowsiness_level = 0
        else:
            # Pourcentage d'yeux fermés pondéré par la confiance
            weighted_closed = sum(
                eye['confidence'] / 100 for eye in eyes_data if eye['state'] == 0
            )
            drowsiness_level = min(100, (weighted_closed / total_eyes) * 100)
        
        return {
            'eyes_detected': True,
            'total_eyes': total_eyes,
            'closed_eyes': closed_eyes,
            'open_eyes': open_eyes,
            'all_eyes_closed': all_eyes_closed,
            'any_eye_closed': any_eye_closed,
            'average_confidence': average_confidence,
            'drowsiness_level': drowsiness_level,
            'eyes_data': eyes_data
        }
    
    def create_annotated_frame(self, frame, faces, eyes_data, drowsiness_analysis):
        """
        Crée une image annotée avec les détections et informations
        
        Args:
            frame: Image originale
            faces: Liste des visages détectés
            eyes_data: Données des yeux
            drowsiness_analysis: Analyse de somnolence
            
        Returns:
            np.array: Image annotée
        """
        annotated_frame = frame.copy()
        overlay = annotated_frame.copy()
        
        try:
            # Annotations des visages
            for (x, y, w, h) in faces:
                # Cadre principal du visage
                cv2.rectangle(overlay, (x-10, y-10), (x+w+10, y+h+10), (0, 255, 255), 2)
                cv2.rectangle(overlay, (x, y), (x+w, y+h), (0, 200, 0), 1)
                
                # Coins technologiques
                self._draw_tech_corners(overlay, x, y, w, h)
            
            # Annotations des yeux
            for eye in eyes_data:
                eye_x, eye_y, eye_w, eye_h = eye['position']
                
                if eye['state'] == 1:  # Ouvert
                    color = (0, 255, 0)
                    status = f"OUVERT ({eye['confidence']:.1f}%)"
                    # Effet de lueur verte
                    cv2.circle(overlay, (eye_x + eye_w//2, eye_y + eye_h//2),
                             max(eye_w, eye_h)//2 + 5, (0, 255, 0), 2)
                else:  # Fermé
                    color = (0, 0, 255)
                    status = f"FERMÉ ({eye['confidence']:.1f}%)"
                    # Effet d'alerte rouge
                    cv2.circle(overlay, (eye_x + eye_w//2, eye_y + eye_h//2),
                             max(eye_w, eye_h)//2 + 10, (0, 0, 255), 3)
                
                cv2.rectangle(overlay, (eye_x, eye_y), (eye_x+eye_w, eye_y+eye_h), color, 2)
                cv2.putText(overlay, status, (eye_x, eye_y-10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Statut global
            self._add_hud_overlay(overlay, drowsiness_analysis)
            
            # Mélanger les overlays
            annotated_frame = cv2.addWeighted(annotated_frame, 0.7, overlay, 0.3, 0)
            
            return annotated_frame
            
        except Exception as e:
            logger.error(f"Erreur création image annotée: {e}")
            return frame
    
    def _draw_tech_corners(self, overlay, x, y, w, h, corner_size=20):
        """Dessine des coins technologiques autour du visage"""
        # Coins supérieurs
        cv2.line(overlay, (x, y), (x+corner_size, y), (0, 255, 255), 3)
        cv2.line(overlay, (x, y), (x, y+corner_size), (0, 255, 255), 3)
        cv2.line(overlay, (x+w, y), (x+w-corner_size, y), (0, 255, 255), 3)
        cv2.line(overlay, (x+w, y), (x+w, y+corner_size), (0, 255, 255), 3)
        
        # Coins inférieurs
        cv2.line(overlay, (x, y+h), (x+corner_size, y+h), (0, 255, 255), 3)
        cv2.line(overlay, (x, y+h), (x, y+h-corner_size), (0, 255, 255), 3)
        cv2.line(overlay, (x+w, y+h), (x+w-corner_size, y+h), (0, 255, 255), 3)
        cv2.line(overlay, (x+w, y+h), (x+w, y+h-corner_size), (0, 255, 255), 3)
    
    def _add_hud_overlay(self, overlay, analysis):
        """Ajoute un HUD (Head-Up Display) à l'image"""
        height, width = overlay.shape[:2]
        
        # Déterminer le statut et la couleur
        if not analysis['eyes_detected']:
            status = "AUCUN VISAGE DÉTECTÉ"
            color = (255, 255, 255)
            hud_height = 80
        elif analysis['all_eyes_closed']:
            status = f"ALERTE: {analysis['closed_eyes']}/{analysis['total_eyes']} YEUX FERMÉS"
            color = (0, 0, 255)
            hud_height = 120
            # Effet d'alerte clignotant
            if int(time.time() * 3) % 2:
                cv2.rectangle(overlay, (0, 0), (width, height), (0, 0, 255), 5)
        else:
            status = "SURVEILLANCE ACTIVE"
            color = (0, 255, 0)
            hud_height = 120
        
        # Créer le HUD
        hud_overlay = np.zeros((hud_height, width, 3), dtype=np.uint8)
        cv2.rectangle(hud_overlay, (0, 0), (width, hud_height), (20, 20, 20), -1)
        
        # Texte principal du HUD
        cv2.putText(hud_overlay, status, (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        
        if analysis['eyes_detected']:
            cv2.putText(hud_overlay, f"YEUX DETECTES: {analysis['total_eyes']}", (20, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            cv2.putText(hud_overlay, f"NIVEAU SOMNOLENCE: {analysis['drowsiness_level']:.1f}%", 
                       (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        
        cv2.putText(hud_overlay, f"TIMESTAMP: {datetime.now().strftime('%H:%M:%S')}",
                   (width-200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        
        # Ajouter le HUD à l'image
        overlay[:hud_height, :] = hud_overlay
    
    def calculate_eye_aspect_ratio(self, eye_landmarks):
        """
        Calcule le ratio d'aspect de l'œil (EAR)
        
        Args:
            eye_landmarks: Points de l'œil (6 points)
            
        Returns:
            float: Ratio d'aspect de l'œil
        """
        try:
            # Distances verticales
            A = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
            B = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
            
            # Distance horizontale
            C = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
            
            # Calcul EAR
            ear = (A + B) / (2.0 * C)
            return ear
            
        except Exception as e:
            logger.error(f"Erreur calcul EAR: {e}")
            return 0.3  # Valeur par défaut pour œil ouvert
    
    def detect_blink(self, ear_history, threshold=0.25):
        """
        Détecte un clignement basé sur l'historique EAR
        
        Args:
            ear_history: Historique des valeurs EAR
            threshold: Seuil de détection
            
        Returns:
            bool: True si clignement détecté
        """
        if len(ear_history) < 3:
            return False
        
        # Détecter une transition fermé -> ouvert
        return (ear_history[-3] < threshold and 
                ear_history[-2] < threshold and 
                ear_history[-1] > threshold)