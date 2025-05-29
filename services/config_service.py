"""
Service de configuration et paramétrage
Gestion des profils, paramètres et configuration système
"""

import json
import os
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

@dataclass
class DetectionConfig:
    """Configuration des paramètres de détection"""
    # Paramètres de base
    threshold_seconds: float = 2.5
    sensitivity: float = 1.0
    ear_threshold: float = 0.25
    blink_threshold: float = 0.3
    
    # Paramètres de cascade Haar
    face_scale_factor: float = 1.3
    face_min_neighbors: int = 5
    face_min_size: tuple = (50, 50)
    
    eye_scale_factor: float = 1.1
    eye_min_neighbors: int = 10
    eye_min_size: tuple = (20, 20)
    
    # Paramètres d'alerte
    progressive_alarm: bool = True
    alarm_cooldown: float = 1.0
    max_alarm_intensity: float = 1.0

@dataclass
class CameraConfig:
    """Configuration de la caméra"""
    camera_id: int = 0
    frame_width: int = 1280
    frame_height: int = 720
    fps: int = 30
    auto_exposure: bool = True
    brightness: float = 0.5
    contrast: float = 0.5

@dataclass
class AudioConfig:
    """Configuration audio"""
    enabled: bool = True
    sample_rate: int = 22050
    buffer_size: int = 512
    volume: float = 0.8
    voice_alerts: bool = False
    custom_sound_path: Optional[str] = None

@dataclass
class AnalyticsConfig:
    """Configuration des analytics"""
    history_size: int = 3600
    enable_trend_analysis: bool = True
    export_data: bool = False
    real_time_charts: bool = True
    update_interval: float = 0.1

@dataclass
class UIConfig:
    """Configuration de l'interface utilisateur"""
    theme: str = "dark"
    language: str = "fr"
    show_debug_info: bool = False
    auto_fullscreen: bool = False
    hud_overlay: bool = True

class ConfigService:
    """Service de gestion de configuration"""
    
    def __init__(self, config_file="config/guardian_eye_config.json"):
        self.config_file = config_file
        self.config_dir = os.path.dirname(config_file)
        
        # Configurations par défaut
        self.detection = DetectionConfig()
        self.camera = CameraConfig()
        self.audio = AudioConfig()
        self.analytics = AnalyticsConfig()
        self.ui = UIConfig()
        
        # Profils prédéfinis
        self.predefined_profiles = self._create_predefined_profiles()
        self.current_profile = "Standard"
        
        # Créer le dossier de config si nécessaire
        os.makedirs(self.config_dir, exist_ok=True)
        
        # Charger la configuration
        self.load_config()
    
    def _create_predefined_profiles(self) -> Dict[str, DetectionConfig]:
        """Crée les profils prédéfinis"""
        profiles = {
            "Standard": DetectionConfig(
                threshold_seconds=2.5,
                sensitivity=1.0,
                progressive_alarm=True
            ),
            "Conduite Urbaine": DetectionConfig(
                threshold_seconds=1.5,
                sensitivity=0.8,
                alarm_cooldown=2.0,
                ear_threshold=0.22
            ),
            "Autoroute": DetectionConfig(
                threshold_seconds=1.0,
                sensitivity=1.2,
                alarm_cooldown=0.5,
                ear_threshold=0.28,
                max_alarm_intensity=1.0
            ),
            "Nuit": DetectionConfig(
                threshold_seconds=2.0,
                sensitivity=0.6,
                alarm_cooldown=1.5,
                ear_threshold=0.3,
                progressive_alarm=True
            ),
            "Sensible": DetectionConfig(
                threshold_seconds=1.5,
                sensitivity=1.5,
                alarm_cooldown=0.3,
                ear_threshold=0.3,
                max_alarm_intensity=0.8
            ),
            "Détendu": DetectionConfig(
                threshold_seconds=4.0,
                sensitivity=0.7,
                alarm_cooldown=3.0,
                ear_threshold=0.2,
                progressive_alarm=False
            )
        }
        return profiles
    
    def load_config(self) -> bool:
        """
        Charge la configuration depuis le fichier
        
        Returns:
            bool: True si chargé avec succès
        """
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
                
                # Charger chaque section
                if 'detection' in config_data:
                    self.detection = DetectionConfig(**config_data['detection'])
                
                if 'camera' in config_data:
                    self.camera = CameraConfig(**config_data['camera'])
                
                if 'audio' in config_data:
                    self.audio = AudioConfig(**config_data['audio'])
                
                if 'analytics' in config_data:
                    self.analytics = AnalyticsConfig(**config_data['analytics'])
                
                if 'ui' in config_data:
                    self.ui = UIConfig(**config_data['ui'])
                
                if 'current_profile' in config_data:
                    self.current_profile = config_data['current_profile']
                
                logger.info(f"Configuration chargée depuis {self.config_file}")
                return True
            else:
                logger.info("Fichier de configuration non trouvé, utilisation des valeurs par défaut")
                self.save_config()  # Créer le fichier avec les valeurs par défaut
                return False
                
        except Exception as e:
            logger.error(f"Erreur lors du chargement de la configuration: {e}")
            return False
    
    def save_config(self) -> bool:
        """
        Sauvegarde la configuration dans le fichier
        
        Returns:
            bool: True si sauvegardé avec succès
        """
        try:
            config_data = {
                'detection': asdict(self.detection),
                'camera': asdict(self.camera),
                'audio': asdict(self.audio),
                'analytics': asdict(self.analytics),
                'ui': asdict(self.ui),
                'current_profile': self.current_profile,
                'version': "1.0.0",
                'last_updated': str(datetime.now())
            }
            
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=4, ensure_ascii=False, default=str)
            
            logger.info(f"Configuration sauvegardée dans {self.config_file}")
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde: {e}")
            return False
    
    def apply_profile(self, profile_name: str) -> bool:
        """
        Applique un profil prédéfini
        
        Args:
            profile_name (str): Nom du profil à appliquer
            
        Returns:
            bool: True si profil appliqué avec succès
        """
        if profile_name in self.predefined_profiles:
            self.detection = self.predefined_profiles[profile_name]
            self.current_profile = profile_name
            logger.info(f"Profil '{profile_name}' appliqué")
            return True
        else:
            logger.warning(f"Profil '{profile_name}' non trouvé")
            return False
    
    def create_custom_profile(self, name: str, config: DetectionConfig) -> bool:
        """
        Crée un profil personnalisé
        
        Args:
            name (str): Nom du profil
            config (DetectionConfig): Configuration du profil
            
        Returns:
            bool: True si créé avec succès
        """
        try:
            self.predefined_profiles[name] = config
            logger.info(f"Profil personnalisé '{name}' créé")
            return True
        except Exception as e:
            logger.error(f"Erreur création profil '{name}': {e}")
            return False
    
    def get_profile_list(self) -> list:
        """Retourne la liste des profils disponibles"""
        return list(self.predefined_profiles.keys())
    
    def get_current_config(self) -> Dict[str, Any]:
        """
        Retourne la configuration actuelle complète
        
        Returns:
            dict: Configuration complète
        """
        return {
            'detection': asdict(self.detection),
            'camera': asdict(self.camera),
            'audio': asdict(self.audio),
            'analytics': asdict(self.analytics),
            'ui': asdict(self.ui),
            'current_profile': self.current_profile
        }
    
    def update_detection_config(self, **kwargs) -> bool:
        """
        Met à jour la configuration de détection
        
        Args:
            **kwargs: Paramètres à mettre à jour
            
        Returns:
            bool: True si mise à jour réussie
        """
        try:
            for key, value in kwargs.items():
                if hasattr(self.detection, key):
                    setattr(self.detection, key, value)
                    logger.debug(f"Paramètre détection mis à jour: {key} = {value}")
                else:
                    logger.warning(f"Paramètre détection inconnu: {key}")
            return True
        except Exception as e:
            logger.error(f"Erreur mise à jour config détection: {e}")
            return False
    
    def update_camera_config(self, **kwargs) -> bool:
        """Met à jour la configuration caméra"""
        try:
            for key, value in kwargs.items():
                if hasattr(self.camera, key):
                    setattr(self.camera, key, value)
                    logger.debug(f"Paramètre caméra mis à jour: {key} = {value}")
            return True
        except Exception as e:
            logger.error(f"Erreur mise à jour config caméra: {e}")
            return False
    
    def update_audio_config(self, **kwargs) -> bool:
        """Met à jour la configuration audio"""
        try:
            for key, value in kwargs.items():
                if hasattr(self.audio, key):
                    setattr(self.audio, key, value)
                    logger.debug(f"Paramètre audio mis à jour: {key} = {value}")
            return True
        except Exception as e:
            logger.error(f"Erreur mise à jour config audio: {e}")
            return False
    
    def validate_config(self) -> Dict[str, list]:
        """
        Valide la configuration actuelle
        
        Returns:
            dict: Dictionnaire avec les erreurs par section
        """
        errors = {
            'detection': [],
            'camera': [],
            'audio': [],
            'analytics': [],
            'ui': []
        }
        
        # Validation détection
        if self.detection.threshold_seconds < 0.5 or self.detection.threshold_seconds > 10.0:
            errors['detection'].append("Seuil doit être entre 0.5 et 10.0 secondes")
        
        if self.detection.sensitivity < 0.1 or self.detection.sensitivity > 3.0:
            errors['detection'].append("Sensibilité doit être entre 0.1 et 3.0")
        
        if self.detection.ear_threshold < 0.1 or self.detection.ear_threshold > 0.5:
            errors['detection'].append("Seuil EAR doit être entre 0.1 et 0.5")
        
        # Validation caméra
        if self.camera.camera_id < 0:
            errors['camera'].append("ID caméra doit être >= 0")
        
        if self.camera.frame_width < 320 or self.camera.frame_width > 1920:
            errors['camera'].append("Largeur frame doit être entre 320 et 1920")
        
        if self.camera.frame_height < 240 or self.camera.frame_height > 1080:
            errors['camera'].append("Hauteur frame doit être entre 240 et 1080")
        
        # Validation audio
        if self.audio.volume < 0.0 or self.audio.volume > 1.0:
            errors['audio'].append("Volume doit être entre 0.0 et 1.0")
        
        if self.audio.sample_rate not in [22050, 44100, 48000]:
            errors['audio'].append("Taux d'échantillonnage doit être 22050, 44100 ou 48000")
        
        # Validation analytics
        if self.analytics.history_size < 100 or self.analytics.history_size > 10000:
            errors['analytics'].append("Taille historique doit être entre 100 et 10000")
        
        if self.analytics.update_interval < 0.01 or self.analytics.update_interval > 1.0:
            errors['analytics'].append("Intervalle mise à jour doit être entre 0.01 et 1.0")
        
        return errors
    
    def reset_to_defaults(self, section: Optional[str] = None) -> bool:
        """
        Remet la configuration aux valeurs par défaut
        
        Args:
            section (str, optional): Section spécifique à réinitialiser
            
        Returns:
            bool: True si réinitialisé avec succès
        """
        try:
            if section is None or section == 'detection':
                self.detection = DetectionConfig()
            
            if section is None or section == 'camera':
                self.camera = CameraConfig()
            
            if section is None or section == 'audio':
                self.audio = AudioConfig()
            
            if section is None or section == 'analytics':
                self.analytics = AnalyticsConfig()
            
            if section is None or section == 'ui':
                self.ui = UIConfig()
            
            if section is None:
                self.current_profile = "Standard"
            
            logger.info(f"Configuration réinitialisée: {section or 'toutes sections'}")
            return True
            
        except Exception as e:
            logger.error(f"Erreur réinitialisation config: {e}")
            return False
    
    def export_config(self, export_path: str) -> bool:
        """
        Exporte la configuration vers un fichier
        
        Args:
            export_path (str): Chemin d'export
            
        Returns:
            bool: True si export réussi
        """
        try:
            config_data = self.get_current_config()
            config_data['export_timestamp'] = str(datetime.now())
            config_data['export_version'] = "1.0.0"
            
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=4, ensure_ascii=False, default=str)
            
            logger.info(f"Configuration exportée vers {export_path}")
            return True
            
        except Exception as e:
            logger.error(f"Erreur export config: {e}")
            return False
    
    def import_config(self, import_path: str) -> bool:
        """
        Importe la configuration depuis un fichier
        
        Args:
            import_path (str): Chemin du fichier à importer
            
        Returns:
            bool: True si import réussi
        """
        try:
            if not os.path.exists(import_path):
                logger.error(f"Fichier d'import non trouvé: {import_path}")
                return False
            
            with open(import_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            # Sauvegarder config actuelle comme backup
            backup_path = self.config_file + ".backup"
            self.export_config(backup_path)
            
            # Importer nouvelle config
            if 'detection' in config_data:
                self.detection = DetectionConfig(**config_data['detection'])
            
            if 'camera' in config_data:
                self.camera = CameraConfig(**config_data['camera'])
            
            if 'audio' in config_data:
                self.audio = AudioConfig(**config_data['audio'])
            
            if 'analytics' in config_data:
                self.analytics = AnalyticsConfig(**config_data['analytics'])
            
            if 'ui' in config_data:
                self.ui = UIConfig(**config_data['ui'])
            
            # Valider la configuration importée
            errors = self.validate_config()
            if any(errors.values()):
                logger.warning("Configuration importée contient des erreurs")
                for section, error_list in errors.items():
                    for error in error_list:
                        logger.warning(f"{section}: {error}")
            
            logger.info(f"Configuration importée depuis {import_path}")
            return True
            
        except Exception as e:
            logger.error(f"Erreur import config: {e}")
            return False
    
    def get_profile_comparison(self, profile1: str, profile2: str) -> Dict[str, Any]:
        """
        Compare deux profils
        
        Args:
            profile1 (str): Premier profil
            profile2 (str): Deuxième profil
            
        Returns:
            dict: Comparaison des profils
        """
        if profile1 not in self.predefined_profiles or profile2 not in self.predefined_profiles:
            return {'error': 'Un ou plusieurs profils non trouvés'}
        
        config1 = asdict(self.predefined_profiles[profile1])
        config2 = asdict(self.predefined_profiles[profile2])
        
        differences = {}
        for key in config1:
            if config1[key] != config2[key]:
                differences[key] = {
                    profile1: config1[key],
                    profile2: config2[key]
                }
        
        return {
            'profile1': profile1,
            'profile2': profile2,
            'differences': differences,
            'identical': len(differences) == 0
        }
    
    def get_recommended_profile(self, driving_conditions: Dict[str, Any]) -> str:
        """
        Recommande un profil basé sur les conditions de conduite
        
        Args:
            driving_conditions (dict): Conditions de conduite
            
        Returns:
            str: Nom du profil recommandé
        """
        # Analyser les conditions
        time_of_day = driving_conditions.get('time_of_day', 'day')
        road_type = driving_conditions.get('road_type', 'mixed')
        driver_experience = driving_conditions.get('experience', 'intermediate')
        fatigue_level = driving_conditions.get('fatigue', 'normal')
        
        # Logique de recommandation
        if time_of_day == 'night' or fatigue_level == 'high':
            return 'Nuit'
        elif road_type == 'highway':
            return 'Autoroute'
        elif road_type == 'city':
            return 'Conduite Urbaine'
        elif driver_experience == 'beginner' or fatigue_level == 'low':
            return 'Sensible'
        elif driver_experience == 'expert':
            return 'Détendu'
        else:
            return 'Standard'
    
    def auto_tune_sensitivity(self, session_data: Dict[str, Any]) -> float:
        """
        Ajuste automatiquement la sensibilité basée sur les données de session
        
        Args:
            session_data (dict): Données de la session précédente
            
        Returns:
            float: Nouvelle valeur de sensibilité
        """
        false_positives = session_data.get('false_positive_rate', 0)
        missed_detections = session_data.get('missed_detection_rate', 0)
        current_sensitivity = self.detection.sensitivity
        
        # Ajustement basé sur les taux d'erreur
        if false_positives > 0.2:  # Trop de faux positifs
            new_sensitivity = max(0.3, current_sensitivity - 0.1)
        elif missed_detections > 0.1:  # Trop de détections manquées
            new_sensitivity = min(2.0, current_sensitivity + 0.1)
        else:
            new_sensitivity = current_sensitivity
        
        return new_sensitivity
    
    def get_config_summary(self) -> str:
        """
        Génère un résumé textuel de la configuration
        
        Returns:
            str: Résumé de la configuration
        """
        summary = f"""
Configuration Guardian Eye - Profil: {self.current_profile}

DÉTECTION:
- Seuil d'alerte: {self.detection.threshold_seconds}s
- Sensibilité: {self.detection.sensitivity}
- Seuil EAR: {self.detection.ear_threshold}
- Alarme progressive: {'Oui' if self.detection.progressive_alarm else 'Non'}

CAMÉRA:
- ID: {self.camera.camera_id}
- Résolution: {self.camera.frame_width}x{self.camera.frame_height}
- FPS: {self.camera.fps}

AUDIO:
- Activé: {'Oui' if self.audio.enabled else 'Non'}
- Volume: {self.audio.volume*100:.0f}%
- Alertes vocales: {'Oui' if self.audio.voice_alerts else 'Non'}

ANALYTICS:
- Historique: {self.analytics.history_size} points
- Graphiques temps réel: {'Oui' if self.analytics.real_time_charts else 'Non'}
- Export données: {'Oui' if self.analytics.export_data else 'Non'}
        """
        return summary.strip()
    
    def get_status(self) -> Dict[str, Any]:
        """
        Retourne le statut du service de configuration
        
        Returns:
            dict: Statut du service
        """
        errors = self.validate_config()
        total_errors = sum(len(error_list) for error_list in errors.values())
        
        return {
            'config_loaded': os.path.exists(self.config_file),
            'current_profile': self.current_profile,
            'available_profiles': len(self.predefined_profiles),
            'validation_errors': total_errors,
            'last_modified': os.path.getmtime(self.config_file) if os.path.exists(self.config_file) else None,
            'config_file_path': self.config_file
        }