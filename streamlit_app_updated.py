"""
Application Streamlit Guardian Eye - Version Modulaire
Utilise les services modulaires pour une architecture propre
"""

import streamlit as st
import cv2
import numpy as np
import time
import threading
from datetime import datetime
import logging

# Import des services modulaires
from services import (
    ModelService,
    DetectionService, 
    AudioService,
    AnalyticsService,
    DashboardService
)
from services.config_service import ConfigService

# Configuration logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('guardian_eye_app')

# Configuration de la page
st.set_page_config(
    page_title="🚗 Guardian Eye - Détecteur de Somnolence Intelligent",
    page_icon="👁️‍🗨️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS créatif (même style que l'original)
creative_css = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Roboto:wght@300;400;700&display=swap');

:root {
    --primary-color: #00D4FF;
    --secondary-color: #FF6B6B;
    --accent-color: #4ECDC4;
    --bg-dark: #0E1117;
    --bg-card: #1E2329;
    --text-light: #FAFAFA;
    --success-color: #00FF88;
    --warning-color: #FFB800;
    --danger-color: #FF4757;
}

@keyframes pulse {
    0% { transform: scale(1); opacity: 1; }
    50% { transform: scale(1.05); opacity: 0.8; }
    100% { transform: scale(1); opacity: 1; }
}

@keyframes glow {
    0% { box-shadow: 0 0 5px var(--primary-color); }
    50% { box-shadow: 0 0 20px var(--primary-color), 0 0 30px var(--primary-color); }
    100% { box-shadow: 0 0 5px var(--primary-color); }
}

@keyframes shake {
    0%, 100% { transform: translateX(0); }
    25% { transform: translateX(-5px); }
    75% { transform: translateX(5px); }
}

.stApp {
    background: linear-gradient(135deg, #0E1117 0%, #1E2329 100%);
    font-family: 'Roboto', sans-serif;
}

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
.stDeployButton {visibility: hidden;}
</style>
"""

st.markdown(creative_css, unsafe_allow_html=True)

class GuardianEyeApp:
    """Application principale Guardian Eye"""
    
    def __init__(self):
        self.initialize_services()
        self.initialize_session_state()
    
    @st.cache_resource
    def initialize_services(_self):
        """Initialise tous les services de manière cached"""
        try:
            # Service de configuration
            config_service = ConfigService()
            
            # Service de modèle IA
            model_service = ModelService()
            model, model_error = model_service.load_drowsiness_model()
            face_cascade, eye_cascade, cascade_error = model_service.load_cascades()
            
            # Service de détection
            detection_service = DetectionService(model_service)
            
            # Service audio
            audio_service = AudioService()
            
            # Service d'analytics
            analytics_service = AnalyticsService()
            
            # Service de dashboard
            dashboard_service = DashboardService(analytics_service)
            
            services = {
                'config': config_service,
                'model': model_service,
                'detection': detection_service,
                'audio': audio_service,
                'analytics': analytics_service,
                'dashboard': dashboard_service
            }
            
            # Vérifier les erreurs
            errors = []
            if model_error:
                errors.append(f"Modèle: {model_error}")
            if cascade_error:
                errors.append(f"Cascades: {cascade_error}")
            
            return services, errors
            
        except Exception as e:
            logger.error(f"Erreur initialisation services: {e}")
            return None, [f"Erreur critique: {e}"]
    
    def initialize_session_state(self):
        """Initialise l'état de session"""
        if 'camera_active' not in st.session_state:
            st.session_state.camera_active = False
        
        if 'emergency_stop' not in st.session_state:
            st.session_state.emergency_stop = False
        
        if 'eyes_closed_start' not in st.session_state:
            st.session_state.eyes_closed_start = None
        
        if 'alert_active' not in st.session_state:
            st.session_state.alert_active = False
        
        if 'last_alarm_time' not in st.session_state:
            st.session_state.last_alarm_time = 0
        
        if 'frame_count' not in st.session_state:
            st.session_state.frame_count = 0
    
    def run(self):
        """Lance l'application principale"""
        services, init_errors = self.initialize_services()
        
        if not services:
            st.error("❌ Erreur critique lors de l'initialisation des services")
            for error in init_errors:
                st.error(error)
            return
        
        # Extraire les services
        config_service = services['config']
        model_service = services['model']
        detection_service = services['detection']
        audio_service = services['audio']
        analytics_service = services['analytics']
        dashboard_service = services['dashboard']
        
        # Créer l'interface
        dashboard_service.create_main_header()
        
        # Afficher les erreurs d'initialisation si présentes
        if init_errors:
            st.warning("⚠️ Certains services ont des problèmes:")
            for error in init_errors:
                st.warning(error)
        
        # Panneau de contrôle
        settings = dashboard_service.create_control_panel()
        
        # Appliquer les paramètres au service de configuration
        config_service.update_detection_config(
            threshold_seconds=settings['threshold'],
            sensitivity=settings['sensitivity'],
            progressive_alarm=settings['alarm_progression']
        )
        
        # Interface de contrôle principal
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            camera_toggle = st.checkbox("🎥 Activer Guardian Eye", value=st.session_state.camera_active)
            if camera_toggle != st.session_state.camera_active:
                st.session_state.camera_active = camera_toggle
                st.rerun()
        
        with col2:
            if st.button("🔄 Reset Stats", help="Remet à zéro les statistiques"):
                analytics_service.reset_session()
                st.success("✅ Statistiques réinitialisées")
                st.rerun()
        
        with col3:
            if st.button("🚨 ARRÊT D'URGENCE", help="Arrête immédiatement le système"):
                st.session_state.emergency_stop = True
                st.session_state.camera_active = False
                audio_service.stop_all_alarms()
                st.error("🛑 Arrêt d'urgence activé")
        
        # Créer les métriques du dashboard
        dashboard_service.create_metrics_cards()
        
        # Zone de statut principal
        status_container = st.empty()
        
        # Zone pour les graphiques en temps réel
        if settings['analytics_display']:
            with st.expander("📊 Analytics Temps Réel", expanded=True):
                charts_container = st.empty()
        
        # Zone vidéo principale
        video_container = st.empty()
        
        # Messages intelligents et conseils
        advice_container = st.empty()
        
        # Logique principale de surveillance
        if st.session_state.camera_active and not st.session_state.emergency_stop:
            self._run_surveillance_loop(
                services, settings, status_container, 
                charts_container if settings['analytics_display'] else None,
                video_container, advice_container
            )
        
        elif st.session_state.emergency_stop:
            video_container.error("🚨 ARRÊT D'URGENCE ACTIVÉ")
            status_container.info("🛑 Système arrêté par l'utilisateur")
        
        else:
            self._show_standby_interface(video_container, status_container, model_service)
        
        # Section d'aide
        self._create_help_section()
    
    def _run_surveillance_loop(self, services, settings, status_container, 
                              charts_container, video_container, advice_container):
        """Lance la boucle de surveillance principale"""
        
        # Extraire les services
        model_service = services['model']
        detection_service = services['detection']
        audio_service = services['audio']
        analytics_service = services['analytics']
        dashboard_service = services['dashboard']
        config_service = services['config']
        
        # Vérifier que les services sont prêts
        if not model_service.is_ready():
            st.error("❌ Services IA non prêts - Vérifiez le modèle et les cascades")
            return
        
        # Initialiser la caméra
        cap = cv2.VideoCapture(config_service.camera.camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config_service.camera.frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config_service.camera.frame_height)
        cap.set(cv2.CAP_PROP_FPS, config_service.camera.fps)
        
        if not cap.isOpened():
            st.error("❌ Impossible d'ouvrir la caméra")
            return
        
        st.success("✅ Guardian Eye activé - Surveillance en cours")
        
        # Variables de surveillance
        last_analytics_update = time.time()
        frame_skip_counter = 0
        
        # Conteneurs pour mise à jour temps réel
        frame_placeholder = video_container.empty()
        
        try:
            while st.session_state.camera_active and not st.session_state.emergency_stop:
                ret, frame = cap.read()
                
                if not ret:
                    st.error("❌ Erreur de capture vidéo")
                    break
                
                current_time = time.time()
                st.session_state.frame_count += 1
                
                # Optimisation: traiter 1 frame sur 3 pour les performances
                if frame_skip_counter % 3 == 0:
                    # Détection des visages et yeux
                    faces, eyes_data = detection_service.detect_faces_and_eyes(frame)
                    
                    # Analyse de somnolence
                    drowsiness_analysis = detection_service.analyze_drowsiness_state(eyes_data)
                    
                    # Créer l'image annotée
                    annotated_frame = detection_service.create_annotated_frame(
                        frame, faces, eyes_data, drowsiness_analysis
                    )
                    
                    # Enregistrer les données analytics
                    analytics_service.add_detection_frame(
                        drowsiness_analysis['any_eye_closed'],
                        drowsiness_analysis['average_confidence']
                    )
                    
                    # Gestion des alertes
                    self._handle_drowsiness_alerts(
                        drowsiness_analysis, current_time, settings,
                        audio_service, analytics_service
                    )
                    
                    # Mise à jour du statut
                    dashboard_service.create_status_display(
                        drowsiness_analysis, st.session_state.alert_active
                    )
                    
                    # Afficher l'image traitée
                    annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                    frame_placeholder.image(annotated_frame_rgb, channels="RGB", use_container_width=True)
                    
                    # Mise à jour des graphiques (moins fréquente)
                    if charts_container and current_time - last_analytics_update > 2:
                        with charts_container.container():
                            dashboard_service.create_realtime_charts()
                        last_analytics_update = current_time
                    
                    # Conseils intelligents
                    self._provide_smart_advice(
                        analytics_service, advice_container, current_time, settings
                    )
                
                frame_skip_counter += 1
                
                # Contrôle de vitesse
                time.sleep(0.05 if frame_skip_counter % 3 == 0 else 0.02)
                
                # Vérifier l'état de l'interface
                if st.button("⏹️ Arrêter la surveillance", key=f"stop_{st.session_state.frame_count}"):
                    break
        
        except Exception as e:
            logger.error(f"Erreur dans la boucle de surveillance: {e}")
            st.error(f"❌ Erreur de surveillance: {e}")
        
        finally:
            # Libérer les ressources
            cap.release()
            audio_service.stop_all_alarms()
            
            # Afficher le rapport de session
            self._show_session_report(analytics_service, dashboard_service)
    
    def _handle_drowsiness_alerts(self, drowsiness_analysis, current_time, settings, 
                                 audio_service, analytics_service):
        """Gère la logique d'alerte de somnolence"""
        
        if drowsiness_analysis['any_eye_closed']:
            if st.session_state.eyes_closed_start is None:
                st.session_state.eyes_closed_start = current_time
            
            elapsed_time = current_time - st.session_state.eyes_closed_start
            
            # Vérifier si on dépasse le seuil
            if elapsed_time > settings['threshold']:
                if not st.session_state.alert_active:
                    st.session_state.alert_active = True
                    
                    # Enregistrer l'événement d'alerte
                    analytics_service.add_alert_event(
                        "drowsiness_detected",
                        elapsed_time,
                        drowsiness_analysis['drowsiness_level']
                    )
                    
                    # Déclencher l'alarme
                    if settings['alarm_progression']:
                        audio_service.play_progressive_alarm(elapsed_time, settings['threshold'])
                    else:
                        audio_service.play_alarm_sequence("standard", 1.0)
                    
                    st.session_state.last_alarm_time = current_time
                
                # Alarmes répétées pour somnolence prolongée
                elif current_time - st.session_state.last_alarm_time > 3.0:
                    alarm_type = "critical" if elapsed_time > settings['threshold'] * 2 else "urgent"
                    audio_service.play_alarm_sequence(alarm_type, settings['sensitivity'])
                    st.session_state.last_alarm_time = current_time
        
        else:
            # Yeux ouverts - réinitialiser les alertes
            if st.session_state.eyes_closed_start is not None:
                # Détecter un clignement potentiel
                blink_duration = current_time - st.session_state.eyes_closed_start
                if 0.1 < blink_duration < 0.5:  # Clignement normal
                    analytics_service.add_blink(blink_duration)
            
            st.session_state.eyes_closed_start = None
            st.session_state.alert_active = False
    
    def _provide_smart_advice(self, analytics_service, advice_container, current_time, settings):
        """Fournit des conseils intelligents basés sur l'analyse"""
        
        # Conseils de pause intelligente
        if settings['smart_pause']:
            session_summary = analytics_service.get_session_summary()
            session_minutes = session_summary['session_duration'] / 60
            
            # Suggérer une pause après 30 minutes avec alertes multiples
            if (session_minutes > 30 and 
                session_summary['total_alerts'] > 2 and
                session_minutes % 30 < 1):  # Toutes les 30 minutes
                
                advice_container.warning(
                    "💡 **Conseil Smart Pause**: Vous avez conduit 30+ minutes avec plusieurs alertes. "
                    "Prenez une pause de 15-20 minutes pour votre sécurité."
                )
            
            # Alerte fatigue cumulative élevée
            elif session_summary['cumulative_fatigue'] > 60:
                advice_container.error(
                    "🚨 **Fatigue Critique Détectée**: Votre niveau de fatigue cumulée est élevé. "
                    "ARRÊTEZ-VOUS dès que possible."
                )
            
            # Conseils sur la fréquence de clignement
            blink_freq = session_summary['blink_frequency']
            if blink_freq < 8:
                advice_container.info(
                    "👁️ **Conseil Clignement**: Votre fréquence de clignement est faible. "
                    "Clignez volontairement plus souvent pour hydrater vos yeux."
                )
    
    def _show_session_report(self, analytics_service, dashboard_service):
        """Affiche le rapport de session détaillé"""
        
        st.success("📊 Session de surveillance terminée")
        
        # Rapport détaillé
        with st.expander("📈 Rapport Détaillé de Session", expanded=True):
            dashboard_service.create_session_report()
            
            # Analytics avancées
            st.markdown("---")
            dashboard_service.create_advanced_analytics()
            
            # Graphique des tendances
            st.markdown("---")
            dashboard_service.create_trend_analysis_chart()
            
            # Export des données
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("💾 Exporter les Données"):
                    export_data = analytics_service.export_data_for_analysis()
                    st.download_button(
                        label="📄 Télécharger Rapport JSON",
                        data=str(export_data),
                        file_name=f"guardian_eye_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
                    st.success("✅ Données préparées pour export")
            
            with col2:
                if st.button("📧 Partager Rapport"):
                    summary = analytics_service.get_session_summary()
                    report_text = f"""
Rapport Guardian Eye - {datetime.now().strftime('%d/%m/%Y %H:%M')}

RÉSUMÉ:
- Durée: {summary['session_duration']/60:.0f} minutes
- Alertes: {summary['total_alerts']}
- Score Vigilance: {summary['vigilance_score']:.0f}/100
- Niveau Risque: {summary['risk_index']:.1f}%

RECOMMANDATION: {self._get_recommendation_text(summary)}
                    """
                    st.text_area("📝 Rapport à Partager", report_text, height=200)
    
    def _get_recommendation_text(self, summary):
        """Génère un texte de recommandation basé sur le résumé"""
        if summary['vigilance_score'] >= 85:
            return "Excellente vigilance maintenue. Continuez ainsi !"
        elif summary['total_alerts'] > 5:
            return "Nombreuses alertes détectées. Repos recommandé avant de reprendre la route."
        elif summary['risk_index'] > 70:
            return "Niveau de risque élevé. Évitez de conduire sans repos suffisant."
        else:
            return "Session correcte. Restez vigilant et prenez des pauses régulières."
    
    def _show_standby_interface(self, video_container, status_container, model_service):
        """Affiche l'interface en mode veille"""
        
        model_info = model_service.get_model_info()
        model_status = "✅ Chargé" if model_info['loaded'] else "❌ Erreur"
        cascade_status = "✅ Prêtes" if model_service.cascades_loaded else "❌ Erreur"
        
        video_container.markdown(f'''
        <div style="
            background: rgba(30, 35, 41, 0.8);
            border-radius: 15px;
            padding: 40px;
            text-align: center;
            margin: 20px 0;
        ">
            <h2 style="color: #00D4FF;">🚗 Guardian Eye Prêt</h2>
            <p style="color: #FAFAFA; font-size: 1.1rem;">
                Activez la surveillance pour commencer la détection de somnolence
            </p>
            <div style="margin-top: 30px;">
                <div style="color: #4ECDC4; margin: 10px 0;">
                    🤖 Modèle IA: {model_status}
                </div>
                <div style="color: #4ECDC4; margin: 10px 0;">
                    👁️ Détection faciale: {cascade_status}
                </div>
                <div style="color: #4ECDC4; margin: 10px 0;">
                    🔊 Système audio: ✅ Prêt
                </div>
                <div style="color: #4ECDC4; margin: 10px 0;">
                    📊 Analytics: ✅ Actifs
                </div>
            </div>
        </div>
        ''', unsafe_allow_html=True)
        
        status_container.empty()
    
    def _create_help_section(self):
        """Crée la section d'aide dans la barre latérale"""
        
        with st.sidebar:
            st.markdown("---")
            st.markdown("## ℹ️ Guide d'Utilisation")
            
            with st.expander("🎯 Profils de Conduite"):
                st.markdown("""
                **Standard**: Équilibre sensibilité/fiabilité  
                **Urbaine**: Moins sensible (embouteillages)  
                **Autoroute**: Plus sensible (vitesse élevée)  
                **Nuit**: Adapté à la fatigue nocturne  
                **Sensible**: Détection très réactive  
                **Détendu**: Pour conducteurs expérimentés
                """)
            
            with st.expander("📊 Niveaux de Somnolence"):
                st.markdown("""
                **0-30%**: État normal ✅  
                **30-50%**: Légère fatigue ⚠️  
                **50-70%**: Attention requise 🔶  
                **70-100%**: Danger immédiat 🚨
                """)
            
            with st.expander("🔧 Optimisation Performances"):
                st.markdown("""
                • **Éclairage**: Lumière uniforme sur le visage  
                • **Position**: 50-80cm de la caméra  
                • **Stabilité**: Caméra fixe, pas de vibrations  
                • **Résolution**: 720p recommandé pour fluidité  
                • **Processeur**: Fermez autres applications lourdes
                """)
            
            with st.expander("🧮 Formules Mathématiques"):
                st.markdown("""
                **Fréquence Clignement**:  
                `BR = (N_clignements × 60) / T_session`
                
                **% Somnolence**:  
                `SP = (Frames_somnolentes / Frames_totales) × 100`
                
                **Score Vigilance**:  
                `VS = 100 - 2.5×SP - 1.2×(APH/10) - 3×ECD + 5×NBR`
                
                **Indice Risque**:  
                `IRC = 0.4×SP + 0.2×(1/BR) + 0.2×APH + 0.2×ECD`
                """)
            
            with st.expander("🆘 Dépannage"):
                st.markdown("""
                **Caméra non détectée**:  
                - Vérifiez les permissions caméra  
                - Changez l'ID caméra (0, 1, 2...)  
                - Redémarrez l'application
                
                **Pas de détection visage**:  
                - Améliorez l'éclairage  
                - Ajustez votre position  
                - Nettoyez l'objectif caméra
                
                **Faux positifs**:  
                - Réduisez la sensibilité  
                - Changez de profil (Détendu)  
                - Vérifiez les reflets/ombres
                """)
            
            st.markdown("---")
            st.markdown("### 🆘 En cas d'urgence")
            st.error("**Utilisez le bouton ARRÊT D'URGENCE**")
            st.info("⚠️ Guardian Eye ne remplace pas votre vigilance personnelle")
            
            # Informations système
            st.markdown("---")
            st.markdown("### 🔧 Informations Système")
            
            if st.button("📊 État des Services"):
                services, _ = self.initialize_services()
                if services:
                    model_status = services['model'].get_status()
                    st.json({
                        "Modèle IA": "✅ Actif" if model_status['ready'] else "❌ Erreur",
                        "Détection": "✅ Prête" if model_status['cascades_loaded'] else "❌ Erreur",
                        "Audio": "✅ Initialisé",
                        "Analytics": "✅ Fonctionnels",
                        "Paramètres": f"✅ Chargés",
                    })

def main():
    """Fonction principale de l'application"""
    try:
        app = GuardianEyeApp()
        app.run()
    except Exception as e:
        logger.critical(f"Erreur critique application: {e}")
        st.error(f"❌ Erreur critique: {e}")
        st.info("🔄 Rechargez la page pour redémarrer l'application")

if __name__ == "__main__":
    main()