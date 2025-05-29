"""
Service de gestion des modèles IA
Chargement et prédictions du modèle de détection des yeux
"""

import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import streamlit as st
import logging

logger = logging.getLogger(__name__)

class ModelService:
    """Service pour la gestion du modèle de détection des yeux"""
    
    def __init__(self):
        self.model = None
        self.face_cascade = None
        self.eye_cascade = None
        self.model_loaded = False
        self.cascades_loaded = False
        
    @st.cache_resource
    def load_drowsiness_model(_self, model_path="saved_model/eye_state_model_final.h5"):
        """
        Charge le modèle de détection de somnolence avec gestion d'erreurs avancée
        
        Args:
            model_path (str): Chemin vers le modèle
            
        Returns:
            tuple: (model, error_message)
        """
        try:
            if os.path.exists(model_path):
                # Essayer différentes méthodes de chargement
                try:
                    model = load_model(model_path, compile=False)
                    _self.model = model
                    _self.model_loaded = True
                    logger.info(f"Modèle chargé avec succès: {model_path}")
                    return model, None
                except Exception as e1:
                    logger.warning(f"Première méthode échouée: {e1}")
                    try:
                        model = tf.keras.models.load_model(model_path, compile=False)
                        _self.model = model
                        _self.model_loaded = True
                        logger.info("Modèle chargé avec méthode alternative")
                        return model, None
                    except Exception as e2:
                        logger.error(f"Deuxième méthode échouée: {e2}")
                        return None, f"Erreur lors du chargement: {str(e2)}"
            else:
                error_msg = f"Modèle non trouvé: {model_path}"
                logger.error(error_msg)
                return None, error_msg
        except Exception as e:
            error_msg = f"Erreur critique lors du chargement: {str(e)}"
            logger.error(error_msg)
            return None, error_msg
    
    @st.cache_resource
    def load_cascades(_self):
        """
        Charge les classificateurs Haar avec optimisations
        
        Returns:
            tuple: (face_cascade, eye_cascade, error_message)
        """
        try:
            face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            eye_cascade_path = cv2.data.haarcascades + 'haarcascade_eye.xml'
            
            face_cascade = cv2.CascadeClassifier(face_cascade_path)
            eye_cascade = cv2.CascadeClassifier(eye_cascade_path)
            
            if face_cascade.empty() or eye_cascade.empty():
                error_msg = "Fichiers cascade Haar non trouvés ou corrompus"
                logger.error(error_msg)
                return None, None, error_msg
            
            _self.face_cascade = face_cascade
            _self.eye_cascade = eye_cascade
            _self.cascades_loaded = True
            
            logger.info("Cascades Haar chargées avec succès")
            return face_cascade, eye_cascade, None
            
        except Exception as e:
            error_msg = f"Erreur cascade: {str(e)}"
            logger.error(error_msg)
            return None, None, error_msg
    
    def preprocess_eye(self, eye_frame):
        """
        Prétraite l'image de l'œil avec améliorations
        
        Args:
            eye_frame: Image de l'œil en niveaux de gris
            
        Returns:
            np.array: Image prétraitée pour le modèle ou None si erreur
        """
        try:
            if eye_frame is None or eye_frame.size == 0:
                return None
                
            # Égalisation d'histogramme pour améliorer le contraste
            eye_frame = cv2.equalizeHist(eye_frame)
            
            # Redimensionnement à 24x24 pixels
            resized = cv2.resize(eye_frame, (24, 24))
            
            # Normalisation
            normalized = resized / 255.0
            
            # Reshape pour le modèle
            normalized = normalized.reshape(1, 24, 24, 1)
            
            return normalized
            
        except Exception as e:
            logger.error(f"Erreur prétraitement œil: {e}")
            return None
    
    def predict_eye_state(self, eye_frame):
        """
        Prédit si l'œil est ouvert ou fermé
        
        Args:
            eye_frame: Image de l'œil en niveaux de gris
            
        Returns:
            tuple: (état de l'œil, confiance) ou (1, 0) si erreur
        """
        if not self.model_loaded or self.model is None:
            logger.warning("Modèle non chargé pour la prédiction")
            return 1, 0  # Par défaut, considérer l'œil comme ouvert
        
        try:
            # Prétraitement
            processed_eye = self.preprocess_eye(eye_frame)
            
            if processed_eye is None:
                return 1, 0
            
            # Prédiction
            prediction = self.model.predict(processed_eye, verbose=0)[0]
            
            # 0: fermé, 1: ouvert
            eye_state = np.argmax(prediction)
            confidence = prediction[eye_state] * 100
            
            return eye_state, confidence
            
        except Exception as e:
            logger.error(f"Erreur lors de la prédiction: {e}")
            return 1, 0  # Par défaut, considérer l'œil comme ouvert
    
    def get_model_info(self):
        """
        Retourne les informations sur le modèle chargé
        
        Returns:
            dict: Informations sur le modèle
        """
        if not self.model_loaded or self.model is None:
            return {
                'loaded': False,
                'architecture': None,
                'parameters': 0,
                'input_shape': None,
                'output_shape': None
            }
        
        try:
            return {
                'loaded': True,
                'architecture': str(type(self.model)),
                'parameters': self.model.count_params(),
                'input_shape': self.model.input_shape,
                'output_shape': self.model.output_shape,
                'layers': len(self.model.layers)
            }
        except Exception as e:
            logger.error(f"Erreur récupération infos modèle: {e}")
            return {'loaded': True, 'error': str(e)}
    
    def is_ready(self):
        """
        Vérifie si le service est prêt à être utilisé
        
        Returns:
            bool: True si le modèle et les cascades sont chargés
        """
        return self.model_loaded and self.cascades_loaded
    
    def get_status(self):
        """
        Retourne le statut détaillé du service
        
        Returns:
            dict: Statut du service
        """
        return {
            'model_loaded': self.model_loaded,
            'cascades_loaded': self.cascades_loaded,
            'ready': self.is_ready(),
            'model_info': self.get_model_info()
        }