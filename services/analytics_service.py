"""
Service d'analyse et statistiques avancées
Calculs mathématiques pour métriques de somnolence
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import deque
import logging
import math

logger = logging.getLogger(__name__)

class AnalyticsService:
    """Service pour l'analyse et le calcul des métriques de somnolence"""
    
    def __init__(self, history_size=3600):  # 1 heure d'historique par défaut
        self.history_size = history_size
        self.session_start = datetime.now()
        
        # Historiques de données
        self.detection_history = deque(maxlen=history_size)
        self.blink_history = deque(maxlen=1000)
        self.alert_history = []
        self.drowsiness_events = []
        
        # Compteurs
        self.total_frames = 0
        self.drowsy_frames = 0
        self.total_blinks = 0
        self.total_alerts = 0
        
        # États
        self.current_drowsiness_level = 0
        self.max_drowsiness_duration = 0
        self.last_blink_time = datetime.now()
        
        # Paramètres de calcul
        self.normal_blink_rate = 16  # clignements/minute normaux
        self.ear_threshold = 0.25  # Eye Aspect Ratio threshold
    
    def add_detection_frame(self, eyes_closed, confidence=0.0, ear_value=None):
        """
        Ajoute une nouvelle frame de détection à l'historique
        
        Args:
            eyes_closed (bool): État des yeux (fermés ou non)
            confidence (float): Confiance de la détection
            ear_value (float): Valeur Eye Aspect Ratio optionnelle
        """
        timestamp = datetime.now()
        
        detection_data = {
            'timestamp': timestamp,
            'eyes_closed': eyes_closed,
            'confidence': confidence,
            'ear_value': ear_value,
            'frame_number': self.total_frames
        }
        
        self.detection_history.append(detection_data)
        self.total_frames += 1
        
        if eyes_closed:
            self.drowsy_frames += 1
    
    def add_blink(self, blink_duration=None):
        """
        Enregistre un clignement détecté
        
        Args:
            blink_duration (float): Durée du clignement en secondes
        """
        timestamp = datetime.now()
        
        blink_data = {
            'timestamp': timestamp,
            'duration': blink_duration or 0.15,  # Durée moyenne
            'interval': (timestamp - self.last_blink_time).total_seconds()
        }
        
        self.blink_history.append(blink_data)
        self.total_blinks += 1
        self.last_blink_time = timestamp
    
    def add_alert_event(self, alert_type, duration, drowsiness_level):
        """
        Enregistre un événement d'alerte
        
        Args:
            alert_type (str): Type d'alerte
            duration (float): Durée de l'épisode de somnolence
            drowsiness_level (float): Niveau de somnolence (0-100)
        """
        alert_data = {
            'timestamp': datetime.now(),
            'type': alert_type,
            'duration': duration,
            'drowsiness_level': drowsiness_level
        }
        
        self.alert_history.append(alert_data)
        self.drowsiness_events.append(alert_data)
        self.total_alerts += 1
        
        # Mettre à jour la durée maximale
        if duration > self.max_drowsiness_duration:
            self.max_drowsiness_duration = duration
    
    def calculate_blink_frequency(self, window_minutes=1):
        """
        Calcule la fréquence de clignement
        
        Args:
            window_minutes (float): Fenêtre temporelle en minutes
            
        Returns:
            float: Fréquence de clignement (clignements/minute)
        """
        if not self.blink_history:
            return 0.0
        
        current_time = datetime.now()
        window_start = current_time - timedelta(minutes=window_minutes)
        
        # Compter les clignements dans la fenêtre
        recent_blinks = [
            blink for blink in self.blink_history 
            if blink['timestamp'] >= window_start
        ]
        
        if not recent_blinks:
            return 0.0
        
        # Formule: BR = (N_blinks × 60) / T_session
        actual_duration = (current_time - recent_blinks[0]['timestamp']).total_seconds()
        if actual_duration > 0:
            return (len(recent_blinks) * 60) / actual_duration
        
        return 0.0
    
    def calculate_drowsiness_percentage(self, window_minutes=5):
        """
        Calcule le pourcentage de somnolence
        
        Args:
            window_minutes (float): Fenêtre temporelle en minutes
            
        Returns:
            float: Pourcentage de somnolence (0-100)
        """
        if not self.detection_history:
            return 0.0
        
        current_time = datetime.now()
        window_start = current_time - timedelta(minutes=window_minutes)
        
        # Filtrer les détections dans la fenêtre
        recent_detections = [
            d for d in self.detection_history 
            if d['timestamp'] >= window_start
        ]
        
        if not recent_detections:
            return 0.0
        
        # Formule: SP = (F_drowsy / F_total) × 100
        drowsy_count = sum(1 for d in recent_detections if d['eyes_closed'])
        total_count = len(recent_detections)
        
        return (drowsy_count / total_count) * 100 if total_count > 0 else 0.0
    
    def calculate_alerts_per_hour(self):
        """
        Calcule le nombre d'alertes par heure
        
        Returns:
            float: Alertes par heure
        """
        session_duration = (datetime.now() - self.session_start).total_seconds()
        
        if session_duration == 0:
            return 0.0
        
        # Formule: APH = (N_alerts × 3600) / T_session
        return (self.total_alerts * 3600) / session_duration
    
    def calculate_vigilance_score(self):
        """
        Calcule le score de vigilance composite
        
        Returns:
            float: Score de vigilance (0-100)
        """
        # Récupérer les métriques de base
        sp = self.calculate_drowsiness_percentage()
        aph = self.calculate_alerts_per_hour()
        br = self.calculate_blink_frequency()
        ecd = self._calculate_average_closure_duration()
        
        # Normaliser la fréquence de clignement
        nbr = (br - self.normal_blink_rate) / 4
        
        # Coefficients de pondération
        alpha = 2.5  # Impact somnolence
        beta = 1.2   # Impact alertes
        gamma = 3.0  # Impact durée fermeture
        delta = 5.0  # Bonus clignement normal
        
        # Formule: VS = 100 - α×SP - β×(APH/10) - γ×ECD + δ×NBR
        vigilance_score = (100 
                          - alpha * sp 
                          - beta * (aph / 10) 
                          - gamma * ecd 
                          + delta * max(-1, min(1, nbr)))
        
        return max(0, min(100, vigilance_score))
    
    def calculate_risk_index(self):
        """
        Calcule l'indice de risque composite (IRC)
        
        Returns:
            float: Indice de risque (0-100)
        """
        sp = self.calculate_drowsiness_percentage()
        br = self.calculate_blink_frequency()
        aph = self.calculate_alerts_per_hour()
        ecd = self._calculate_average_closure_duration()
        
        # Pondérations
        w1, w2, w3, w4 = 0.4, 0.2, 0.2, 0.2
        
        # Formule: IRC = w₁×SP + w₂×(1/BR) + w₃×APH + w₄×ECD
        inverse_br = (1 / max(br, 1)) * 100  # Normaliser
        
        irc = (w1 * sp + 
               w2 * min(inverse_br, 100) + 
               w3 * min(aph, 100) + 
               w4 * min(ecd * 10, 100))
        
        return min(100, irc)
    
    def calculate_incident_probability(self):
        """
        Calcule la probabilité d'incident basée sur l'IRC
        
        Returns:
            float: Probabilité d'incident (0-1)
        """
        irc = self.calculate_risk_index()
        lambda_param = 0.15
        
        # Formule: PI = 1 - e^(-λ×IRC)
        return 1 - math.exp(-lambda_param * irc / 100)
    
    def _calculate_average_closure_duration(self):
        """
        Calcule la durée moyenne de fermeture des yeux
        
        Returns:
            float: Durée moyenne en secondes
        """
        if not self.drowsiness_events:
            return 0.0
        
        durations = [event['duration'] for event in self.drowsiness_events]
        return sum(durations) / len(durations)
    
    def calculate_exponential_moving_average(self, values, alpha=0.3):
        """
        Calcule la moyenne mobile exponentielle pour lissage
        
        Args:
            values (list): Liste de valeurs
            alpha (float): Facteur de lissage
            
        Returns:
            list: Valeurs lissées
        """
        if not values:
            return []
        
        ema = [values[0]]
        
        for i in range(1, len(values)):
            # EMA_t = α × X_t + (1-α) × EMA_{t-1}
            ema_value = alpha * values[i] + (1 - alpha) * ema[-1]
            ema.append(ema_value)
        
        return ema
    
    def calculate_adaptive_threshold(self):
        """
        Calcule un seuil adaptatif basé sur le temps de session
        
        Returns:
            float: Seuil adaptatif en secondes
        """
        session_duration_hours = (datetime.now() - self.session_start).total_seconds() / 3600
        
        base_threshold = 2.5
        kappa = 0.2
        
        # Formule: Threshold_t = Threshold_base × (1 + κ × ln(1 + t))
        adaptive_threshold = base_threshold * (1 + kappa * math.log(1 + session_duration_hours))
        
        return adaptive_threshold
    
    def calculate_cumulative_fatigue(self, decay_factor=0.95):
        """
        Calcule la fatigue cumulée avec persistance
        
        Args:
            decay_factor (float): Facteur de persistance
            
        Returns:
            float: Score de fatigue cumulée
        """
        if not hasattr(self, '_cumulative_fatigue'):
            self._cumulative_fatigue = 0.0
        
        current_drowsiness = self.calculate_drowsiness_percentage(window_minutes=1)
        
        # Formule: CF_t = CF_{t-1} × δ + SP_t × (1-δ)
        self._cumulative_fatigue = (self._cumulative_fatigue * decay_factor + 
                                   current_drowsiness * (1 - decay_factor))
        
        return self._cumulative_fatigue
    
    def get_trend_analysis(self, window_minutes=10):
        """
        Analyse les tendances sur une fenêtre temporelle
        
        Args:
            window_minutes (float): Fenêtre d'analyse en minutes
            
        Returns:
            dict: Analyse des tendances
        """
        current_time = datetime.now()
        window_start = current_time - timedelta(minutes=window_minutes)
        
        # Filtrer les données récentes
        recent_data = [
            d for d in self.detection_history 
            if d['timestamp'] >= window_start
        ]
        
        if len(recent_data) < 5:
            return {'trend': 'insufficient_data', 'slope': 0, 'correlation': 0}
        
        # Calculer la tendance de somnolence
        timestamps = [(d['timestamp'] - window_start).total_seconds() for d in recent_data]
        drowsiness_values = [1 if d['eyes_closed'] else 0 for d in recent_data]
        
        # Régression linéaire simple
        n = len(timestamps)
        sum_x = sum(timestamps)
        sum_y = sum(drowsiness_values)
        sum_xy = sum(x * y for x, y in zip(timestamps, drowsiness_values))
        sum_x2 = sum(x * x for x in timestamps)
        
        if n * sum_x2 - sum_x * sum_x == 0:
            slope = 0
        else:
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        
        # Déterminer la tendance
        if slope > 0.001:
            trend = 'increasing'
        elif slope < -0.001:
            trend = 'decreasing'
        else:
            trend = 'stable'
        
        # Calculer la corrélation
        if len(set(drowsiness_values)) > 1:
            correlation = np.corrcoef(timestamps, drowsiness_values)[0, 1]
        else:
            correlation = 0
        
        return {
            'trend': trend,
            'slope': slope,
            'correlation': correlation,
            'data_points': len(recent_data)
        }
    
    def get_session_summary(self):
        """
        Génère un résumé complet de la session
        
        Returns:
            dict: Résumé détaillé de la session
        """
        session_duration = datetime.now() - self.session_start
        
        return {
            # Métriques de base
            'session_duration': session_duration.total_seconds(),
            'total_frames': self.total_frames,
            'drowsy_frames': self.drowsy_frames,
            'total_blinks': self.total_blinks,
            'total_alerts': self.total_alerts,
            
            # Métriques calculées
            'blink_frequency': self.calculate_blink_frequency(),
            'drowsiness_percentage': self.calculate_drowsiness_percentage(),
            'alerts_per_hour': self.calculate_alerts_per_hour(),
            'vigilance_score': self.calculate_vigilance_score(),
            'risk_index': self.calculate_risk_index(),
            'incident_probability': self.calculate_incident_probability(),
            
            # Statistiques avancées
            'max_drowsiness_duration': self.max_drowsiness_duration,
            'average_closure_duration': self._calculate_average_closure_duration(),
            'cumulative_fatigue': self.calculate_cumulative_fatigue(),
            'adaptive_threshold': self.calculate_adaptive_threshold(),
            
            # Analyse des tendances
            'trend_analysis': self.get_trend_analysis(),
            
            # Timestamps
            'session_start': self.session_start,
            'last_update': datetime.now()
        }
    
    def export_data_for_analysis(self):
        """
        Exporte les données pour analyse externe
        
        Returns:
            dict: Données structurées pour export
        """
        return {
            'detection_history': list(self.detection_history),
            'blink_history': list(self.blink_history),
            'alert_history': self.alert_history,
            'drowsiness_events': self.drowsiness_events,
            'session_summary': self.get_session_summary()
        }
    
    def reset_session(self):
        """Remet à zéro toutes les statistiques de session"""
        self.detection_history.clear()
        self.blink_history.clear()
        self.alert_history.clear()
        self.drowsiness_events.clear()
        
        self.total_frames = 0
        self.drowsy_frames = 0
        self.total_blinks = 0
        self.total_alerts = 0
        
        self.current_drowsiness_level = 0
        self.max_drowsiness_duration = 0
        self.session_start = datetime.now()
        self.last_blink_time = datetime.now()
        
        if hasattr(self, '_cumulative_fatigue'):
            self._cumulative_fatigue = 0.0
        
        logger.info("Session analytics réinitialisées")