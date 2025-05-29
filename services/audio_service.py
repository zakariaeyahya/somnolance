"""
Service de gestion audio et alarmes
Génération d'alarmes dynamiques et notifications sonores
"""

import pygame
import numpy as np
import threading
import time
import os
import logging

logger = logging.getLogger(__name__)

class AudioService:
    """Service pour la gestion des alarmes et notifications audio"""
    
    def __init__(self):
        self.audio_initialized = False
        self.last_alarm_time = 0
        self.alarm_cooldown = 1.0  # Cooldown entre alarmes en secondes
        self.current_alarm_thread = None
        
        # Configuration audio
        self.sample_rate = 22050
        self.buffer_size = 512
        
        # Initialiser pygame mixer
        self._initialize_audio()
    
    def _initialize_audio(self):
        """Initialise le système audio avec sons multiples"""
        try:
            pygame.mixer.init(
                frequency=self.sample_rate, 
                size=-16, 
                channels=2, 
                buffer=self.buffer_size
            )
            self.audio_initialized = True
            logger.info("Système audio initialisé avec succès")
            return True
            
        except Exception as e:
            logger.error(f"Erreur initialisation audio: {e}")
            self.audio_initialized = False
            return False
    
    def generate_tone(self, frequency, duration_ms, intensity=1.0, wave_type='sine'):
        """
        Génère un ton audio dynamique
        
        Args:
            frequency (float): Fréquence en Hz
            duration_ms (int): Durée en millisecondes
            intensity (float): Intensité (0.0 à 1.0)
            wave_type (str): Type d'onde ('sine', 'square', 'sawtooth')
            
        Returns:
            pygame.mixer.Sound: Objet son pygame
        """
        try:
            frames = int(duration_ms * self.sample_rate / 1000)
            arr = np.zeros(frames, dtype=np.float32)
            
            for i in range(frames):
                t = i / self.sample_rate
                
                if wave_type == 'sine':
                    arr[i] = np.sin(2 * np.pi * frequency * t)
                elif wave_type == 'square':
                    arr[i] = np.sign(np.sin(2 * np.pi * frequency * t))
                elif wave_type == 'sawtooth':
                    arr[i] = 2 * (t * frequency - np.floor(t * frequency + 0.5))
                else:
                    arr[i] = np.sin(2 * np.pi * frequency * t)  # Défaut: sinus
            
            # Appliquer l'intensité et un fade in/out
            fade_frames = min(frames // 10, 1000)  # 10% ou max 1000 échantillons
            
            # Fade in
            for i in range(fade_frames):
                arr[i] *= (i / fade_frames)
            
            # Fade out
            for i in range(fade_frames):
                arr[frames - 1 - i] *= (i / fade_frames)
            
            # Appliquer l'intensité
            arr *= intensity
            
            # Convertir en format audio
            arr = (arr * 32767).astype(np.int16)
            
            # Créer un son stéréo
            stereo_arr = np.column_stack((arr, arr))
            
            return pygame.sndarray.make_sound(stereo_arr)
            
        except Exception as e:
            logger.error(f"Erreur génération ton: {e}")
            return None
    
    def play_alarm_sequence(self, alarm_type="standard", intensity=1.0):
        """
        Joue une séquence d'alarme selon le type et l'intensité
        
        Args:
            alarm_type (str): Type d'alarme
            intensity (float): Intensité de l'alarme
        """
        if not self.audio_initialized:
            self._fallback_alarm(intensity)
            return
        
        # Définir les paramètres selon le type d'alarme
        alarm_configs = {
            'gentle': {
                'frequencies': [440, 523, 659],  # A4, C5, E5
                'durations': [500, 400, 300],
                'wave_type': 'sine',
                'repeat': 1
            },
            'standard': {
                'frequencies': [659, 784, 988],  # E5, G5, B5
                'durations': [300, 250, 200],
                'wave_type': 'sine',
                'repeat': 2
            },
            'urgent': {
                'frequencies': [880, 1047, 1319],  # A5, C6, E6
                'durations': [200, 150, 100],
                'wave_type': 'square',
                'repeat': 3
            },
            'critical': {
                'frequencies': [1760, 2093, 2637],  # A6, C7, E7
                'durations': [100, 80, 60],
                'wave_type': 'square',
                'repeat': 4
            }
        }
        
        config = alarm_configs.get(alarm_type, alarm_configs['standard'])
        
        def play_sequence():
            try:
                for _ in range(config['repeat']):
                    for freq, duration in zip(config['frequencies'], config['durations']):
                        if pygame.mixer.get_busy():
                            pygame.mixer.stop()
                        
                        sound = self.generate_tone(
                            freq, 
                            duration, 
                            intensity, 
                            config['wave_type']
                        )
                        
                        if sound:
                            sound.play()
                            time.sleep(duration / 1000.0)
                    
                    # Pause entre répétitions
                    if config['repeat'] > 1:
                        time.sleep(0.1)
                        
            except Exception as e:
                logger.error(f"Erreur lecture séquence alarme: {e}")
                self._fallback_alarm(intensity)
        
        # Jouer dans un thread séparé
        if self.current_alarm_thread and self.current_alarm_thread.is_alive():
            return  # Alarme déjà en cours
        
        self.current_alarm_thread = threading.Thread(target=play_sequence)
        self.current_alarm_thread.daemon = True
        self.current_alarm_thread.start()
    
    def play_progressive_alarm(self, elapsed_time, threshold=2.5):
        """
        Joue une alarme progressive basée sur le temps écoulé
        
        Args:
            elapsed_time (float): Temps écoulé en secondes
            threshold (float): Seuil de déclenchement
        """
        current_time = time.time()
        
        # Vérifier le cooldown
        if current_time - self.last_alarm_time < self.alarm_cooldown:
            return
        
        # Déterminer le type d'alarme selon le temps écoulé
        if elapsed_time < threshold:
            return  # Pas d'alarme encore
        elif elapsed_time < threshold * 1.5:
            alarm_type = "gentle"
            intensity = 0.5
            self.alarm_cooldown = 2.0
        elif elapsed_time < threshold * 2.5:
            alarm_type = "standard" 
            intensity = 0.7
            self.alarm_cooldown = 1.5
        elif elapsed_time < threshold * 4:
            alarm_type = "urgent"
            intensity = 0.9
            self.alarm_cooldown = 1.0
        else:
            alarm_type = "critical"
            intensity = 1.0
            self.alarm_cooldown = 0.5
        
        self.play_alarm_sequence(alarm_type, intensity)
        self.last_alarm_time = current_time
        
        logger.info(f"Alarme {alarm_type} déclenchée (temps: {elapsed_time:.1f}s)")
    
    def play_notification(self, notification_type="info"):
        """
        Joue une notification audio légère
        
        Args:
            notification_type (str): Type de notification
        """
        if not self.audio_initialized:
            return
        
        notifications = {
            'info': {'freq': 800, 'duration': 200, 'intensity': 0.3},
            'success': {'freq': 1000, 'duration': 300, 'intensity': 0.4},
            'warning': {'freq': 600, 'duration': 400, 'intensity': 0.5},
            'error': {'freq': 400, 'duration': 500, 'intensity': 0.6}
        }
        
        config = notifications.get(notification_type, notifications['info'])
        
        def play_notification_sound():
            try:
                sound = self.generate_tone(
                    config['freq'],
                    config['duration'], 
                    config['intensity'],
                    'sine'
                )
                if sound:
                    sound.play()
            except Exception as e:
                logger.error(f"Erreur notification audio: {e}")
        
        thread = threading.Thread(target=play_notification_sound)
        thread.daemon = True
        thread.start()
    
    def _fallback_alarm(self, intensity=1.0):
        """
        Alarme de secours si pygame ne fonctionne pas
        
        Args:
            intensity (float): Intensité de l'alarme
        """
        try:
            # Essayer winsound sur Windows
            if os.name == 'nt':
                import winsound
                frequency = int(2500 * intensity)
                duration = int(1000 * intensity)
                winsound.Beep(frequency, duration)
                return
        except ImportError:
            pass
        
        # Beep ASCII comme dernier recours
        beep_count = max(1, int(3 * intensity))
        for _ in range(beep_count):
            print("\a", end="", flush=True)
            time.sleep(0.2)
    
    def load_custom_sound(self, file_path):
        """
        Charge un fichier son personnalisé
        
        Args:
            file_path (str): Chemin vers le fichier audio
            
        Returns:
            pygame.mixer.Sound: Objet son ou None si erreur
        """
        if not self.audio_initialized:
            return None
        
        try:
            if os.path.exists(file_path):
                sound = pygame.mixer.Sound(file_path)
                logger.info(f"Son personnalisé chargé: {file_path}")
                return sound
            else:
                logger.warning(f"Fichier son non trouvé: {file_path}")
                return None
        except Exception as e:
            logger.error(f"Erreur chargement son: {e}")
            return None
    
    def play_custom_alarm(self, sound_file=None, repeat=1):
        """
        Joue une alarme personnalisée
        
        Args:
            sound_file (str): Chemin vers le fichier son
            repeat (int): Nombre de répétitions
        """
        if sound_file:
            custom_sound = self.load_custom_sound(sound_file)
            if custom_sound:
                def play_custom():
                    try:
                        for _ in range(repeat):
                            if pygame.mixer.get_busy():
                                pygame.mixer.stop()
                            custom_sound.play()
                            # Attendre que le son se termine
                            while pygame.mixer.get_busy():
                                time.sleep(0.1)
                            time.sleep(0.2)  # Pause entre répétitions
                    except Exception as e:
                        logger.error(f"Erreur alarme personnalisée: {e}")
                
                thread = threading.Thread(target=play_custom)
                thread.daemon = True
                thread.start()
                return
        
        # Fallback vers alarme standard
        self.play_alarm_sequence("standard", 1.0)
    
    def stop_all_alarms(self):
        """Arrête toutes les alarmes en cours"""
        try:
            if self.audio_initialized:
                pygame.mixer.stop()
            
            if self.current_alarm_thread and self.current_alarm_thread.is_alive():
                # Le thread se terminera naturellement
                pass
                
            logger.info("Toutes les alarmes arrêtées")
            
        except Exception as e:
            logger.error(f"Erreur arrêt alarmes: {e}")
    
    def set_volume(self, volume):
        """
        Ajuste le volume global
        
        Args:
            volume (float): Volume de 0.0 à 1.0
        """
        try:
            if self.audio_initialized:
                pygame.mixer.set_num_channels(8)  # Permet plusieurs sons simultanés
                # Note: pygame.mixer n'a pas de contrôle de volume global
                # Le volume est contrôlé par l'intensité dans generate_tone
                logger.info(f"Volume configuré: {volume}")
        except Exception as e:
            logger.error(f"Erreur configuration volume: {e}")
    
    def test_audio_system(self):
        """
        Teste le système audio avec différents types de sons
        
        Returns:
            dict: Résultats des tests
        """
        results = {
            'audio_initialized': self.audio_initialized,
            'tone_generation': False,
            'alarm_playback': False,
            'custom_sound': False
        }
        
        if not self.audio_initialized:
            return results
        
        try:
            # Test génération de ton
            test_tone = self.generate_tone(800, 200, 0.3)
            if test_tone:
                results['tone_generation'] = True
                test_tone.play()
                time.sleep(0.3)
            
            # Test alarme
            self.play_notification('info')
            results['alarm_playback'] = True
            
            # Test son personnalisé (si fichier de test existe)
            test_file = "test_sound.wav"  # Fichier de test optionnel
            if os.path.exists(test_file):
                test_custom = self.load_custom_sound(test_file)
                if test_custom:
                    results['custom_sound'] = True
            
        except Exception as e:
            logger.error(f"Erreur test système audio: {e}")
        
        return results
    
    def get_status(self):
        """
        Retourne le statut du service audio
        
        Returns:
            dict: Statut du service
        """
        return {
            'initialized': self.audio_initialized,
            'sample_rate': self.sample_rate,
            'buffer_size': self.buffer_size,
            'last_alarm': self.last_alarm_time,
            'cooldown': self.alarm_cooldown,
            'alarm_active': (self.current_alarm_thread and 
                           self.current_alarm_thread.is_alive()),
            'mixer_busy': pygame.mixer.get_busy() if self.audio_initialized else False
        }