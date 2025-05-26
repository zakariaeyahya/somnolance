import streamlit as st
import cv2
import numpy as np
import time
import tensorflow as tf
from tensorflow.keras.models import load_model
import tempfile
import os
from PIL import Image, ImageDraw
import pygame
import threading
import json
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import pandas as pd
import random

# Configuration de la page avec thème sombre
st.set_page_config(
    page_title="🚗 Guardian Eye - Détecteur de Somnolence Intelligent",
    page_icon="👁️‍🗨️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS créatif avec animations et thème moderne
creative_css = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Roboto:wght@300;400;700&display=swap');

/* Variables CSS pour le thème */
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

/* Animation keyframes */
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

/* Style général */
.stApp {
    background: linear-gradient(135deg, #0E1117 0%, #1E2329 100%);
    font-family: 'Roboto', sans-serif;
}

/* Titre principal avec effet néon */
.main-title {
    font-family: 'Orbitron', monospace;
    font-size: 3rem;
    font-weight: 900;
    text-align: center;
    background: linear-gradient(45deg, var(--primary-color), var(--accent-color));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 2rem;
    text-shadow: 0 0 30px rgba(0, 212, 255, 0.5);
    animation: glow 3s ease-in-out infinite;
}

/* Cards avec effet glassmorphism */
.status-card {
    background: rgba(30, 35, 41, 0.8);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(0, 212, 255, 0.3);
    border-radius: 15px;
    padding: 20px;
    margin: 10px 0;
    box-shadow: 0 8px 32px rgba(0, 212, 255, 0.1);
    transition: all 0.3s ease;
}

.status-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 15px 35px rgba(0, 212, 255, 0.2);
}

/* Alerte danger avec animation */
.danger-alert {
    background: linear-gradient(45deg, #FF4757, #FF6B6B);
    color: white;
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    font-weight: bold;
    font-size: 1.5rem;
    animation: shake 0.5s ease-in-out infinite, pulse 1s ease-in-out infinite;
    box-shadow: 0 0 50px rgba(255, 71, 87, 0.5);
}

/* Statut de surveillance */
.monitoring-active {
    background: linear-gradient(45deg, var(--success-color), var(--accent-color));
    color: white;
    padding: 15px;
    border-radius: 10px;
    text-align: center;
    font-weight: bold;
    animation: pulse 2s ease-in-out infinite;
}

/* Compteurs avec effet néon */
.neon-counter {
    font-family: 'Orbitron', monospace;
    font-size: 2rem;
    font-weight: 700;
    color: var(--primary-color);
    text-shadow: 0 0 10px var(--primary-color);
    text-align: center;
}

/* Boutons futuristes */
.futuristic-button {
    background: linear-gradient(45deg, var(--primary-color), var(--accent-color));
    border: none;
    color: white;
    padding: 12px 25px;
    border-radius: 25px;
    font-weight: bold;
    cursor: pointer;
    transition: all 0.3s ease;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.futuristic-button:hover {
    transform: scale(1.05);
    box-shadow: 0 5px 20px rgba(0, 212, 255, 0.4);
}

/* Masquer les éléments Streamlit par défaut */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
.stDeployButton {visibility: hidden;}

/* Progress bar personnalisée */
.custom-progress {
    background: linear-gradient(90deg, var(--success-color), var(--warning-color), var(--danger-color));
    height: 10px;
    border-radius: 5px;
    overflow: hidden;
    position: relative;
}

.progress-glow {
    position: absolute;
    top: 0;
    left: 0;
    height: 100%;
    background: rgba(255, 255, 255, 0.3);
    border-radius: 5px;
    animation: glow 2s ease-in-out infinite;
}
</style>
"""

st.markdown(creative_css, unsafe_allow_html=True)

# Classe pour gérer les statistiques avancées
class AdvancedStats:
    def __init__(self):
        self.reset_daily_stats()

    def reset_daily_stats(self):
        self.detection_history = []
        self.alertes_count = 0
        self.session_start = datetime.now()
        self.total_blinks = 0
        self.drowsiness_events = []
        self.max_drowsiness_duration = 0

    def add_detection(self, eyes_closed, drowsiness_level):
        timestamp = datetime.now()
        self.detection_history.append({
            'timestamp': timestamp,
            'eyes_closed': eyes_closed,
            'drowsiness_level': drowsiness_level
        })

        # Garder seulement les 1000 dernières détections
        if len(self.detection_history) > 1000:
            self.detection_history = self.detection_history[-1000:]

    def add_drowsiness_event(self, duration):
        self.drowsiness_events.append({
            'timestamp': datetime.now(),
            'duration': duration
        })
        self.alertes_count += 1
        if duration > self.max_drowsiness_duration:
            self.max_drowsiness_duration = duration

    def get_drowsiness_trend(self):
        if len(self.detection_history) < 60:
            return []

        # Calculer la tendance sur les 60 dernières secondes
        recent_data = self.detection_history[-60:]
        timestamps = [d['timestamp'] for d in recent_data]
        drowsiness_levels = [d['drowsiness_level'] for d in recent_data]

        return list(zip(timestamps, drowsiness_levels))

@st.cache_resource
def load_drowsiness_model():
    """Charge le modèle de détection de somnolence avec gestion d'erreurs avancée"""
    model_path = "saved_model/eye_state_model_final.h5"

    try:
        if os.path.exists(model_path):
            model = load_model(model_path, compile=False)
            return model, None
        else:
            return None, f"Modèle non trouvé: {model_path}"
    except Exception as e:
        return None, f"Erreur lors du chargement: {str(e)}"

@st.cache_resource
def load_cascades():
    """Charge les classificateurs Haar avec optimisations"""
    try:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        return face_cascade, eye_cascade, None
    except Exception as e:
        return None, None, f"Erreur cascade: {str(e)}"

@st.cache_resource
def initialize_audio():
    """Initialise le système audio avec sons multiples"""
    try:
        pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
        return True, None
    except Exception as e:
        return False, f"Erreur audio: {str(e)}"

def play_dynamic_alarm(alarm_type="standard", intensity=1.0):
    """Joue différents types d'alarmes selon la situation"""
    def play_sound():
        try:
            # Sons dynamiques basés sur l'intensité
            if alarm_type == "gentle":
                # Son doux pour première alerte
                frequencies = [440, 523, 659]  # A4, C5, E5
                duration = 500
            elif alarm_type == "urgent":
                # Son urgent pour somnolence prolongée
                frequencies = [880, 1047, 1319]  # A5, C6, E6
                duration = 200
            elif alarm_type == "critical":
                # Son critique pour danger imminent
                frequencies = [1760, 2093, 2637]  # A6, C7, E7
                duration = 100
            else:
                frequencies = [659, 784, 988]  # Standard
                duration = 300

            # Générer et jouer les sons
            for freq in frequencies:
                # Créer une onde sinusoïdale
                sample_rate = 22050
                frames = int(duration * sample_rate / 1000)
                arr = np.zeros(frames)

                for i in range(frames):
                    arr[i] = np.sin(2 * np.pi * freq * i / sample_rate) * intensity

                # Convertir en format audio
                arr = (arr * 32767).astype(np.int16)
                sound = pygame.sndarray.make_sound(np.column_stack((arr, arr)))
                sound.play()
                time.sleep(duration / 1000)

        except Exception as e:
            # Fallback vers beep système
            try:
                import winsound
                winsound.Beep(int(2500 * intensity), int(1000 * intensity))
            except:
                print("\a" * int(3 * intensity))

    thread = threading.Thread(target=play_sound)
    thread.daemon = True
    thread.start()

def calculate_drowsiness_level(eyes_closed_duration, blink_frequency, head_position=None):
    """Calcule un niveau de somnolence sophistiqué (0-100)"""
    level = 0

    # Facteur principal: durée des yeux fermés
    if eyes_closed_duration > 0:
        level += min(eyes_closed_duration * 20, 60)  # Max 60 points

    # Facteur secondaire: fréquence de clignement
    if blink_frequency < 10:  # Moins de 10 clignements/minute = somnolence
        level += (10 - blink_frequency) * 2
    elif blink_frequency > 30:  # Plus de 30 = fatigue
        level += (blink_frequency - 30) * 1.5

    # Facteur tertiaire: position de la tête (si disponible)
    if head_position and head_position < -15:  # Tête qui tombe
        level += 20

    return min(int(level), 100)

def enhanced_eye_detection(frame, model, face_cascade, eye_cascade):
    """Détection avancée avec analyse de patterns"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(50, 50))

    eye_states = []
    annotated_frame = frame.copy()
    face_data = []

    # Appliquer un effet de surveillance futuriste
    overlay = annotated_frame.copy()

    for (x, y, w, h) in faces:
        # Effet de scan futuriste
        cv2.rectangle(overlay, (x-10, y-10), (x+w+10, y+h+10), (0, 255, 255), 2)
        cv2.rectangle(overlay, (x, y), (x+w, y+h), (0, 200, 0), 1)

        # Points de coins pour effet tech
        corner_size = 20
        cv2.line(overlay, (x, y), (x+corner_size, y), (0, 255, 255), 3)
        cv2.line(overlay, (x, y), (x, y+corner_size), (0, 255, 255), 3)
        cv2.line(overlay, (x+w, y), (x+w-corner_size, y), (0, 255, 255), 3)
        cv2.line(overlay, (x+w, y), (x+w, y+corner_size), (0, 255, 255), 3)
        cv2.line(overlay, (x, y+h), (x+corner_size, y+h), (0, 255, 255), 3)
        cv2.line(overlay, (x, y+h), (x, y+h-corner_size), (0, 255, 255), 3)
        cv2.line(overlay, (x+w, y+h), (x+w-corner_size, y+h), (0, 255, 255), 3)
        cv2.line(overlay, (x+w, y+h), (x+w, y+h-corner_size), (0, 255, 255), 3)

        roi_gray = gray[y:y+h, x:x+w]
        roi_color = annotated_frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 10, minSize=(20, 20))

        face_eyes = []
        for (ex, ey, ew, eh) in eyes:
            eye_frame = roi_gray[ey:ey+eh, ex:ex+ew]

            if eye_frame.size > 0:
                processed_eye = preprocess_eye(eye_frame)

                if processed_eye is not None:
                    prediction = model.predict(processed_eye, verbose=0)[0]
                    eye_state = np.argmax(prediction)
                    confidence = prediction[eye_state] * 100

                    eye_states.append(eye_state)
                    face_eyes.append({
                        'state': eye_state,
                        'confidence': confidence,
                        'position': (ex, ey, ew, eh)
                    })

                    # Effet visuel avancé pour les yeux
                    eye_x, eye_y = x + ex, y + ey

                    if eye_state == 1:  # Ouvert
                        color = (0, 255, 0)
                        status = f"OUVERT ({confidence:.1f}%)"
                        # Effet de lueur verte
                        cv2.circle(overlay, (eye_x + ew//2, eye_y + eh//2),
                                 max(ew, eh)//2 + 5, (0, 255, 0), 2)
                    else:  # Fermé
                        color = (0, 0, 255)
                        status = f"FERMÉ ({confidence:.1f}%)"
                        # Effet d'alerte rouge
                        cv2.circle(overlay, (eye_x + ew//2, eye_y + eh//2),
                                 max(ew, eh)//2 + 10, (0, 0, 255), 3)

                    cv2.rectangle(overlay, (eye_x, eye_y), (eye_x+ew, eye_y+eh), color, 2)
                    cv2.putText(overlay, status, (eye_x, eye_y-10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        face_data.append({
            'position': (x, y, w, h),
            'eyes': face_eyes
        })

    # Mélanger l'overlay avec l'image originale
    annotated_frame = cv2.addWeighted(annotated_frame, 0.7, overlay, 0.3, 0)

    # Analyse globale
    if eye_states:
        closed_eyes_count = sum(1 for state in eye_states if state == 0)
        total_eyes = len(eye_states)
        any_closed = closed_eyes_count >= 1

        # Interface HUD futuriste
        if any_closed:
            status = f"ALERTE: {closed_eyes_count}/{total_eyes} YEUX FERMÉS"
            color = (0, 0, 255)
            # Effet d'alerte clignotant
            if int(time.time() * 3) % 2:
                cv2.rectangle(annotated_frame, (0, 0),
                            (annotated_frame.shape[1], annotated_frame.shape[0]),
                            (0, 0, 255), 5)
        else:
            status = "SURVEILLANCE ACTIVE"
            color = (0, 255, 0)

        # HUD principal
        hud_height = 120
        hud_overlay = np.zeros((hud_height, annotated_frame.shape[1], 3), dtype=np.uint8)
        cv2.rectangle(hud_overlay, (0, 0), (annotated_frame.shape[1], hud_height),
                     (20, 20, 20), -1)

        # Texte principal du HUD
        cv2.putText(hud_overlay, status, (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        cv2.putText(hud_overlay, f"YEUX DETECTES: {total_eyes}", (20, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.putText(hud_overlay, f"TIMESTAMP: {datetime.now().strftime('%H:%M:%S')}",
                   (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

        # Ajouter le HUD à l'image
        annotated_frame = np.vstack([hud_overlay, annotated_frame])

    else:
        status = "AUCUN VISAGE DÉTECTÉ"
        any_closed = False

        # HUD pour aucune détection
        hud_height = 80
        hud_overlay = np.zeros((hud_height, annotated_frame.shape[1], 3), dtype=np.uint8)
        cv2.rectangle(hud_overlay, (0, 0), (annotated_frame.shape[1], hud_height),
                     (50, 50, 50), -1)
        cv2.putText(hud_overlay, status, (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        annotated_frame = np.vstack([hud_overlay, annotated_frame])

    return annotated_frame, any_closed, face_data

def preprocess_eye(eye_frame):
    """Prétraite l'image de l'œil avec améliorations"""
    try:
        # Égalisation d'histogramme pour améliorer le contraste
        eye_frame = cv2.equalizeHist(eye_frame)
        resized = cv2.resize(eye_frame, (24, 24))
        normalized = resized / 255.0
        normalized = normalized.reshape(1, 24, 24, 1)
        return normalized
    except:
        return None

def create_dashboard():
    """Crée un tableau de bord interactif avec statistiques"""
    st.markdown('<div class="main-title">🚗 GUARDIAN EYE</div>', unsafe_allow_html=True)
    st.markdown('<div style="text-align: center; color: #4ECDC4; font-size: 1.2rem; margin-bottom: 2rem;">Système Intelligent de Détection de Somnolence</div>', unsafe_allow_html=True)

    # Statistiques en temps réel
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f'''
        <div class="status-card">
            <div class="neon-counter">{st.session_state.stats.alertes_count}</div>
            <div style="text-align: center; color: #FF6B6B;">Alertes Déclenchées</div>
        </div>
        ''', unsafe_allow_html=True)

    with col2:
        session_duration = datetime.now() - st.session_state.stats.session_start
        minutes = int(session_duration.total_seconds() / 60)
        st.markdown(f'''
        <div class="status-card">
            <div class="neon-counter">{minutes}</div>
            <div style="text-align: center; color: #4ECDC4;">Minutes de Surveillance</div>
        </div>
        ''', unsafe_allow_html=True)

    with col3:
        st.markdown(f'''
        <div class="status-card">
            <div class="neon-counter">{st.session_state.stats.max_drowsiness_duration:.1f}s</div>
            <div style="text-align: center; color: #FFB800;">Max Somnolence</div>
        </div>
        ''', unsafe_allow_html=True)

    with col4:
        detections = len(st.session_state.stats.detection_history)
        st.markdown(f'''
        <div class="status-card">
            <div class="neon-counter">{detections}</div>
            <div style="text-align: center; color: #00FF88;">Détections Totales</div>
        </div>
        ''', unsafe_allow_html=True)

def create_advanced_controls():
    """Crée des contrôles avancés pour la détection"""
    st.sidebar.markdown("## ⚙️ Contrôles Avancés")

    # Profils prédéfinis
    profile = st.sidebar.selectbox(
        "🎯 Profil de Conduite",
        ["Standard", "Conduite Urbaine", "Autoroute", "Nuit", "Personnalisé"],
        help="Sélectionnez un profil adapté à votre situation"
    )

    # Ajustement des seuils selon le profil
    if profile == "Conduite Urbaine":
        default_threshold = 1.5
        sensitivity = 0.8
    elif profile == "Autoroute":
        default_threshold = 1.0
        sensitivity = 1.2
    elif profile == "Nuit":
        default_threshold = 2.0
        sensitivity = 0.6
    else:
        default_threshold = 2.5
        sensitivity = 1.0

    # Contrôles détaillés
    threshold = st.sidebar.slider(
        "⏰ Seuil d'Alerte (secondes)",
        0.5, 10.0, default_threshold, 0.1,
        help="Durée avant déclenchement de l'alerte"
    )

    sensitivity = st.sidebar.slider(
        "🎚️ Sensibilité",
        0.3, 2.0, sensitivity, 0.1,
        help="Ajuste la sensibilité de détection"
    )

    # Types d'alarmes
    alarm_progression = st.sidebar.checkbox(
        "📈 Alarme Progressive",
        value=True,
        help="L'alarme s'intensifie avec le temps"
    )

    # Fonctionnalités avancées
    st.sidebar.markdown("## 🚀 Fonctionnalités Avancées")

    voice_alerts = st.sidebar.checkbox(
        "🔊 Alertes Vocales",
        value=False,
        help="Activé les messages vocaux"
    )

    smart_pause = st.sidebar.checkbox(
        "⏸️ Pause Intelligente",
        value=True,
        help="Suggère des pauses régulières"
    )

    analytics = st.sidebar.checkbox(
        "📊 Analytics Temps Réel",
        value=True,
        help="Affiche les graphiques en temps réel"
    )

    return {
        'profile': profile,
        'threshold': threshold,
        'sensitivity': sensitivity,
        'alarm_progression': alarm_progression,
        'voice_alerts': voice_alerts,
        'smart_pause': smart_pause,
        'analytics': analytics
    }
def create_realtime_charts(detection_history):
    """Crée des graphiques en temps réel"""
    if len(detection_history) < 5:
        return

    # Préparer les données
    timestamps = [d['timestamp'] for d in detection_history[-60:]]  # 60 dernières secondes
    drowsiness_levels = [d['drowsiness_level'] for d in detection_history[-60:]]

    # Graphique de tendance
    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(
        x=timestamps,
        y=drowsiness_levels,
        mode='lines+markers',
        name='Niveau de Somnolence',
        line=dict(color='#00D4FF', width=3),
        marker=dict(size=6, color='#FF6B6B'),
        fill='tonexty',
        fillcolor='rgba(0, 212, 255, 0.1)'
    ))

    fig_trend.update_layout(
        title="🌊 Tendance de Somnolence (Temps Réel)",
        xaxis_title="Temps",
        yaxis_title="Niveau (%)",
        template='plotly_dark',
        height=400,
        showlegend=False
    )

    # Jauge de niveau actuel
    current_level = drowsiness_levels[-1] if drowsiness_levels else 0
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=current_level,
        title={'text': "🎯 Niveau Actuel"},
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "#00D4FF"},
            'steps': [
                {'range': [0, 30], 'color': "#00FF88"},
                {'range': [30, 70], 'color': "#FFB800"},
                {'range': [70, 100], 'color': "#FF4757"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 80
            }
        }
    ))

    fig_gauge.update_layout(
        template='plotly_dark',
        height=300
    )

    col1, col2 = st.columns([2, 1])
    with col1:
        st.plotly_chart(fig_trend, use_container_width=True, key=f"trend_chart_{int(time.time()*1000)}")
    with col2:
        st.plotly_chart(fig_gauge, use_container_width=True, key=f"gauge_chart_{int(time.time()*1000)}")
def main():
    # Initialize session state for stats BEFORE calling create_dashboard
    if 'stats' not in st.session_state:
        st.session_state.stats = AdvancedStats()
    
    # Initialisation de l'interface créative
    create_dashboard()

    # Contrôles avancés
    settings = create_advanced_controls()

    # Chargement des modèles
    with st.spinner("🔄 Initialisation du système Guardian Eye..."):
        model, model_error = load_drowsiness_model()
        face_cascade, eye_cascade, cascade_error = load_cascades()
        audio_ready, audio_error = initialize_audio()

    # ... reste du code inchangé

    # Interface de contrôle principal
    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        camera_active = st.checkbox("🎥 Activer Guardian Eye", value=False)

    with col2:
        if st.button("🔄 Reset Stats", help="Remet à zéro les statistiques"):
            st.session_state.stats.reset_daily_stats()
            st.success("✅ Statistiques réinitialisées")

    with col3:
        emergency_stop = st.button("🚨 ARRÊT D'URGENCE", help="Arrête immédiatement le système")

    # Rest of your code...


    # Zone de statut principal
    status_container = st.empty()

    # Zone pour les graphiques en temps réel
    if settings['analytics']:
        charts_container = st.empty()

    # Zone vidéo principale
    video_container = st.empty()

    # Messages intelligents et conseils
    advice_container = st.empty()

    if camera_active and not emergency_stop:
        # Variables de session étendues
        if 'eyes_closed_start' not in st.session_state:
            st.session_state.eyes_closed_start = None
        if 'alert_active' not in st.session_state:
            st.session_state.alert_active = False
        if 'last_alarm_time' not in st.session_state:
            st.session_state.last_alarm_time = 0
        if 'blink_counter' not in st.session_state:
            st.session_state.blink_counter = 0
        if 'last_blink_time' not in st.session_state:
            st.session_state.last_blink_time = time.time()
        if 'consecutive_detections' not in st.session_state:
            st.session_state.consecutive_detections = 0
        if 'last_break_suggestion' not in st.session_state:
            st.session_state.last_break_suggestion = time.time()

        # Initialiser la caméra
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)

        if not cap.isOpened():
            st.error("❌ Impossible d'ouvrir la caméra")
            return

        st.success("✅ Guardian Eye activé - Surveillance en cours")

        # Boucle principale de surveillance
        frame_count = 0
        last_analytics_update = time.time()

        while camera_active and not emergency_stop:
            ret, frame = cap.read()

            if not ret:
                st.error("❌ Erreur de capture vidéo")
                break

            frame_count += 1
            current_time = time.time()

            # Traitement de l'image avec détection avancée
            processed_frame, eyes_closed, face_data = enhanced_eye_detection(
                frame, model, face_cascade, eye_cascade
            )

            # Calcul du niveau de somnolence
            eyes_closed_duration = 0
            if eyes_closed and st.session_state.eyes_closed_start:
                eyes_closed_duration = current_time - st.session_state.eyes_closed_start

            # Calcul de la fréquence de clignement
            blink_frequency = st.session_state.blink_counter / max((current_time - st.session_state.last_blink_time), 1) * 60

            drowsiness_level = calculate_drowsiness_level(
                eyes_closed_duration, blink_frequency
            )

            # Ajustement avec la sensibilité
            drowsiness_level = min(int(drowsiness_level * settings['sensitivity']), 100)

            # Enregistrement des statistiques
            st.session_state.stats.add_detection(eyes_closed, drowsiness_level)

            # Gestion de la logique d'alerte avancée
            if eyes_closed:
                if st.session_state.eyes_closed_start is None:
                    st.session_state.eyes_closed_start = current_time

                elapsed_time = current_time - st.session_state.eyes_closed_start
                st.session_state.consecutive_detections += 1

                # Système d'alerte progressif
                if elapsed_time > settings['threshold']:
                    if not st.session_state.alert_active:
                        st.session_state.alert_active = True
                        st.session_state.stats.add_drowsiness_event(elapsed_time)

                        # Alarme progressive selon le temps
                        if settings['alarm_progression']:
                            if elapsed_time < settings['threshold'] * 2:
                                alarm_type = "gentle"
                                intensity = 0.5
                            elif elapsed_time < settings['threshold'] * 4:
                                alarm_type = "urgent"
                                intensity = 0.8
                            else:
                                alarm_type = "critical"
                                intensity = 1.0
                        else:
                            alarm_type = "standard"
                            intensity = 1.0

                        play_dynamic_alarm(alarm_type, intensity)
                        st.session_state.last_alarm_time = current_time

                    # Répétition d'alarme intelligente
                    elif current_time - st.session_state.last_alarm_time > (3.0 - settings['sensitivity']):
                        alarm_type = "critical" if elapsed_time > settings['threshold'] * 3 else "urgent"
                        play_dynamic_alarm(alarm_type, min(1.0, settings['sensitivity']))
                        st.session_state.last_alarm_time = current_time

                    # Effets visuels d'urgence
                    h, w = processed_frame.shape[:2]

                    # Cadre clignotant selon l'urgence
                    if int(current_time * 4) % 2:  # Clignotement rapide
                        thickness = 8 if elapsed_time > settings['threshold'] * 2 else 5
                        color = (0, 0, 255) if elapsed_time > settings['threshold'] * 2 else (0, 100, 255)
                        cv2.rectangle(processed_frame, (0, 0), (w, h), color, thickness)

                    # Messages d'urgence
                    urgency_msg = ""
                    if elapsed_time > settings['threshold'] * 3:
                        urgency_msg = "🆘 DANGER CRITIQUE - ARRÊTEZ-VOUS!"
                    elif elapsed_time > settings['threshold'] * 2:
                        urgency_msg = "⚠️ SOMNOLENCE DANGEREUSE"
                    else:
                        urgency_msg = "😴 ALERTE SOMNOLENCE"

                    # Overlay d'urgence
                    overlay = processed_frame.copy()
                    cv2.rectangle(overlay, (0, h-100), (w, h), (0, 0, 0), -1)
                    cv2.putText(overlay, urgency_msg, (20, h-50),
                              cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                    cv2.putText(overlay, f"Durée: {elapsed_time:.1f}s", (20, h-20),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    processed_frame = cv2.addWeighted(processed_frame, 0.7, overlay, 0.3, 0)

            else:
                # Détection de clignement
                if st.session_state.eyes_closed_start is not None:
                    blink_duration = current_time - st.session_state.eyes_closed_start
                    if 0.1 < blink_duration < 0.5:  # Clignement normal
                        st.session_state.blink_counter += 1

                st.session_state.eyes_closed_start = None
                st.session_state.alert_active = False
                st.session_state.consecutive_detections = 0

            # Affichage de l'image traitée
            processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            video_container.image(processed_frame_rgb, channels="RGB", use_container_width=True)

            # Mise à jour du statut principal
            if st.session_state.alert_active:
                elapsed = current_time - st.session_state.eyes_closed_start if st.session_state.eyes_closed_start else 0
                status_container.markdown(f'''
                <div class="danger-alert">
                    🚨 ALERTE SOMNOLENCE ACTIVE! 🔊<br>
                    Durée: {elapsed:.1f}s | Niveau: {drowsiness_level}%
                </div>
                ''', unsafe_allow_html=True)
            elif eyes_closed:
                elapsed = current_time - st.session_state.eyes_closed_start if st.session_state.eyes_closed_start else 0
                status_container.markdown(f'''
                <div class="status-card" style="background: linear-gradient(45deg, #FFB800, #FF6B6B);">
                    <div style="text-align: center; color: white; font-weight: bold;">
                        😴 Yeux fermés depuis {elapsed:.1f}s<br>
                        Niveau de somnolence: {drowsiness_level}%
                    </div>
                </div>
                ''', unsafe_allow_html=True)
            else:
                status_container.markdown(f'''
                <div class="monitoring-active">
                    👀 Surveillance Active | Niveau: {drowsiness_level}% | Clignements/min: {blink_frequency:.1f}
                </div>
                ''', unsafe_allow_html=True)

            # Conseils intelligents et suggestions de pause
            if settings['smart_pause'] and current_time - st.session_state.last_break_suggestion > 1800:  # 30 min
                if st.session_state.stats.alertes_count > 2:
                    advice_container.warning("💡 Conseil: Vous avez eu plusieurs alertes. Prenez une pause de 15-20 minutes.")
                    st.session_state.last_break_suggestion = current_time

            # Mise à jour des graphiques en temps réel
            if settings['analytics'] and current_time - last_analytics_update > 2:
                with charts_container.container():
                    create_realtime_charts(st.session_state.stats.detection_history)
                last_analytics_update = current_time

            # Messages vocaux (simulation)
            if settings['voice_alerts'] and st.session_state.alert_active:
                if random.randint(1, 100) == 1:  # Message occasionnel
                    advice_container.info("🔊 'Attention, signes de somnolence détectés. Veuillez vous arrêter dès que possible.'")

            # Optimisation de performance
            if frame_count % 3 == 0:  # Traiter 1 frame sur 3 pour de meilleures performances
                time.sleep(0.05)
            else:
                time.sleep(0.1)

            # Vérifier si l'utilisateur a arrêté
            if st.button("⏹️ Arrêter la surveillance", key=f"stop_{frame_count}"):
                break

        # Libération des ressources
        cap.release()

        # Rapport de session
        session_duration = datetime.now() - st.session_state.stats.session_start
        st.success(f"📊 Session terminée - Durée: {session_duration}")

        # Affichage du rapport final
        with st.expander("📈 Rapport de Session", expanded=True):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Alertes Totales", st.session_state.stats.alertes_count)
                st.metric("Détections", len(st.session_state.stats.detection_history))

            with col2:
                st.metric("Durée Max Somnolence", f"{st.session_state.stats.max_drowsiness_duration:.1f}s")
                st.metric("Clignements Totaux", st.session_state.blink_counter)

            with col3:
                avg_drowsiness = np.mean([d['drowsiness_level'] for d in st.session_state.stats.detection_history]) if st.session_state.stats.detection_history else 0
                st.metric("Niveau Moyen", f"{avg_drowsiness:.1f}%")

                # Évaluation de la conduite
                if avg_drowsiness < 20:
                    driving_quality = "Excellente 🌟"
                elif avg_drowsiness < 40:
                    driving_quality = "Bonne 👍"
                elif avg_drowsiness < 60:
                    driving_quality = "Attention requise ⚠️"
                else:
                    driving_quality = "Préoccupante 🚨"

                st.metric("Qualité Conduite", driving_quality)

            # Conseils personnalisés
            st.markdown("### 💡 Conseils Personnalisés")
            if st.session_state.stats.alertes_count > 5:
                st.warning("🛑 Nombreuses alertes détectées. Évitez de conduire si vous êtes fatigué.")
            elif st.session_state.stats.alertes_count > 2:
                st.info("⏸️ Prenez des pauses plus fréquentes lors de longs trajets.")
            else:
                st.success("✅ Bonne vigilance maintenue pendant la session.")

    elif emergency_stop:
        video_container.error("🚨 ARRÊT D'URGENCE ACTIVÉ")
        status_container.info("🛑 Système arrêté par l'utilisateur")

    else:
        video_container.markdown('''
        <div class="status-card">
            <div style="text-align: center; padding: 40px;">
                <h2>🚗 Guardian Eye Prêt</h2>
                <p>Activez la surveillance pour commencer la détection de somnolence</p>
                <div style="margin-top: 20px;">
                    <div style="color: #4ECDC4;">✅ Modèle IA chargé</div>
                    <div style="color: #4ECDC4;">✅ Détection faciale active</div>
                    <div style="color: #4ECDC4;">✅ Système audio prêt</div>
                </div>
            </div>
        </div>
        ''', unsafe_allow_html=True)

        status_container.empty()
        advice_container.empty()

    # Section d'aide et informations
    with st.sidebar:
        st.markdown("---")
        st.markdown("## ℹ️ Guide d'Utilisation")

        with st.expander("🎯 Profils de Conduite"):
            st.write("""
            **Standard**: Équilibre sensibilité/fiabilité
            **Urbaine**: Moins sensible (embouteillages)
            **Autoroute**: Plus sensible (vitesse élevée)
            **Nuit**: Adapté à la fatigue nocturne
            """)

        with st.expander("📊 Niveaux de Somnolence"):
            st.write("""
            **0-30%**: État normal ✅
            **30-50%**: Légère fatigue ⚠️
            **50-70%**: Attention requise 🔶
            **70-100%**: Danger immédiat 🚨
            """)

        with st.expander("🔧 Conseils Techniques"):
            st.write("""
            • Éclairage: Évitez les contre-jours
            • Position: Visage bien visible
            • Caméra: Stable et nette
            • Lunettes: Peuvent affecter la détection
            """)

        st.markdown("---")
        st.markdown("### 🆘 En cas d'urgence")
        st.error("Utilisez le bouton ARRÊT D'URGENCE")
        st.info("Guardian Eye ne remplace pas la vigilance du conducteur")

if __name__ == "__main__":
    main()
