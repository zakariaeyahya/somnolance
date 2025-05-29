"""
Service d'export CSV pour les performances Guardian Eye
Enregistrement des données de performance et statistiques
"""

import csv
import os
import pandas as pd
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class CSVExportService:
    """Service pour l'export des données de performance en CSV"""
    
    def __init__(self, export_dir="exports"):
        self.export_dir = export_dir
        os.makedirs(export_dir, exist_ok=True)
        
        # Fichiers CSV
        self.performance_file = os.path.join(export_dir, "guardian_eye_performance.csv")
        self.sessions_file = os.path.join(export_dir, "guardian_eye_sessions.csv")
        self.alerts_file = os.path.join(export_dir, "guardian_eye_alerts.csv")
        
        # Initialiser les fichiers CSV s'ils n'existent pas
        self._initialize_csv_files()
    
    def _initialize_csv_files(self):
        """Initialise les fichiers CSV avec les en-têtes"""
        
        # Fichier de performance détaillée
        if not os.path.exists(self.performance_file):
            with open(self.performance_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'session_id', 'frame_number', 'fps', 'eyes_detected',
                    'eyes_closed', 'drowsiness_level', 'confidence', 'blink_frequency',
                    'processing_time_ms', 'alert_active', 'ear_value'
                ])
        
        # Fichier de sessions
        if not os.path.exists(self.sessions_file):
            with open(self.sessions_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'session_id', 'start_time', 'end_time', 'duration_minutes',
                    'total_frames', 'total_alerts', 'max_drowsiness_duration',
                    'avg_drowsiness_level', 'total_blinks', 'avg_blink_frequency',
                    'vigilance_score', 'risk_index', 'incident_probability',
                    'profile_used', 'threshold_seconds', 'sensitivity'
                ])
        
        # Fichier d'alertes
        if not os.path.exists(self.alerts_file):
            with open(self.alerts_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'session_id', 'alert_type', 'duration_seconds',
                    'drowsiness_level', 'trigger_reason', 'alarm_type', 'resolved_time'
                ])
    
    def log_frame_performance(self, session_id, frame_data):
        """
        Enregistre les performances d'une frame
        
        Args:
            session_id (str): ID de la session
            frame_data (dict): Données de la frame
        """
        try:
            with open(self.performance_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.now().isoformat(),
                    session_id,
                    frame_data.get('frame_number', 0),
                    frame_data.get('fps', 0),
                    frame_data.get('eyes_detected', 0),
                    frame_data.get('eyes_closed', False),
                    frame_data.get('drowsiness_level', 0),
                    frame_data.get('confidence', 0),
                    frame_data.get('blink_frequency', 0),
                    frame_data.get('processing_time_ms', 0),
                    frame_data.get('alert_active', False),
                    frame_data.get('ear_value', 0)
                ])
        except Exception as e:
            logger.error(f"Erreur enregistrement frame performance: {e}")
    
    def log_session_summary(self, session_id, session_data, settings):
        """
        Enregistre le résumé d'une session
        
        Args:
            session_id (str): ID de la session
            session_data (dict): Données de la session
            settings (dict): Paramètres utilisés
        """
        try:
            with open(self.sessions_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    session_id,
                    session_data.get('session_start', '').isoformat() if session_data.get('session_start') else '',
                    datetime.now().isoformat(),
                    session_data.get('session_duration', 0) / 60,  # En minutes
                    session_data.get('total_frames', 0),
                    session_data.get('total_alerts', 0),
                    session_data.get('max_drowsiness_duration', 0),
                    session_data.get('drowsiness_percentage', 0),
                    session_data.get('total_blinks', 0),
                    session_data.get('blink_frequency', 0),
                    session_data.get('vigilance_score', 0),
                    session_data.get('risk_index', 0),
                    session_data.get('incident_probability', 0),
                    settings.get('profile', 'Standard'),
                    settings.get('threshold', 2.5),
                    settings.get('sensitivity', 1.0)
                ])
        except Exception as e:
            logger.error(f"Erreur enregistrement session: {e}")
    
    def log_alert_event(self, session_id, alert_data):
        """
        Enregistre un événement d'alerte
        
        Args:
            session_id (str): ID de la session
            alert_data (dict): Données de l'alerte
        """
        try:
            with open(self.alerts_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.now().isoformat(),
                    session_id,
                    alert_data.get('type', 'drowsiness'),
                    alert_data.get('duration', 0),
                    alert_data.get('drowsiness_level', 0),
                    alert_data.get('trigger_reason', 'eyes_closed'),
                    alert_data.get('alarm_type', 'standard'),
                    alert_data.get('resolved_time', '').isoformat() if alert_data.get('resolved_time') else ''
                ])
        except Exception as e:
            logger.error(f"Erreur enregistrement alerte: {e}")
    
    def export_session_to_csv(self, session_id, analytics_service):
        """
        Exporte une session complète vers un fichier CSV dédié
        
        Args:
            session_id (str): ID de la session
            analytics_service: Service d'analytics
            
        Returns:
            str: Chemin du fichier exporté
        """
        try:
            export_data = analytics_service.export_data_for_analysis()
            filename = f"session_{session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            filepath = os.path.join(self.export_dir, filename)
            
            # Créer un DataFrame avec toutes les données
            rows = []
            
            # Données de détection
            for detection in export_data.get('detection_history', []):
                rows.append({
                    'type': 'detection',
                    'timestamp': detection['timestamp'].isoformat(),
                    'eyes_closed': detection['eyes_closed'],
                    'confidence': detection.get('confidence', 0),
                    'ear_value': detection.get('ear_value', 0),
                    'frame_number': detection.get('frame_number', 0)
                })
            
            # Données de clignements
            for blink in export_data.get('blink_history', []):
                rows.append({
                    'type': 'blink',
                    'timestamp': blink['timestamp'].isoformat(),
                    'duration': blink['duration'],
                    'interval': blink['interval']
                })
            
            # Données d'alertes
            for alert in export_data.get('alert_history', []):
                rows.append({
                    'type': 'alert',
                    'timestamp': alert['timestamp'].isoformat(),
                    'alert_type': alert['type'],
                    'duration': alert['duration'],
                    'drowsiness_level': alert['drowsiness_level']
                })
            
            # Créer le DataFrame et sauvegarder
            df = pd.DataFrame(rows)
            df.to_csv(filepath, index=False, encoding='utf-8')
            
            logger.info(f"Session exportée vers: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Erreur export session CSV: {e}")
            return None
    
    def get_performance_stats(self, session_id=None):
        """
        Analyse les statistiques de performance depuis les CSV
        
        Args:
            session_id (str, optional): ID de session spécifique
            
        Returns:
            dict: Statistiques de performance
        """
        try:
            if not os.path.exists(self.performance_file):
                return {}
            
            df = pd.read_csv(self.performance_file)
            
            if session_id:
                df = df[df['session_id'] == session_id]
            
            if df.empty:
                return {}
            
            stats = {
                'avg_fps': df['fps'].mean(),
                'min_fps': df['fps'].min(),
                'max_fps': df['fps'].max(),
                'avg_processing_time': df['processing_time_ms'].mean(),
                'total_frames': len(df),
                'frames_with_eyes': df['eyes_detected'].sum(),
                'detection_rate': (df['eyes_detected'].sum() / len(df)) * 100,
                'avg_drowsiness': df['drowsiness_level'].mean(),
                'alerts_triggered': df['alert_active'].sum(),
                'avg_confidence': df['confidence'].mean()
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Erreur analyse performance: {e}")
            return {}
    
    def cleanup_old_files(self, days_to_keep=30):
        """
        Nettoie les anciens fichiers CSV
        
        Args:
            days_to_keep (int): Nombre de jours à conserver
        """
        try:
            cutoff_time = datetime.now().timestamp() - (days_to_keep * 24 * 3600)
            
            for filename in os.listdir(self.export_dir):
                filepath = os.path.join(self.export_dir, filename)
                if os.path.isfile(filepath) and filename.endswith('.csv'):
                    if os.path.getmtime(filepath) < cutoff_time:
                        os.remove(filepath)
                        logger.info(f"Fichier ancien supprimé: {filename}")
                        
        except Exception as e:
            logger.error(f"Erreur nettoyage fichiers: {e}")