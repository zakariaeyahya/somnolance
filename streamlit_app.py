import streamlit as st
import cv2
import numpy as np
import time
import tensorflow as tf
from tensorflow.keras.models import load_model
import tempfile
import os
from PIL import Image

# Configuration de la page
st.set_page_config(
    page_title="Détection de Somnolence",
    page_icon="👁️",
    layout="centered"
)

# CSS pour cacher les éléments Streamlit non nécessaires
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
.stApp > div:first-child {padding-top: 0rem;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

@st.cache_resource
def load_drowsiness_model():
    """Charge le modèle de détection de somnolence"""
    model_path = "saved_model/eye_state_model_final.h5"
    
    try:
        # Essayer de charger le modèle
        if os.path.exists(model_path):
            model = load_model(model_path, compile=False)
            return model, None
        else:
            return None, f"Modèle non trouvé: {model_path}"
    except Exception as e:
        return None, f"Erreur lors du chargement: {str(e)}"

@st.cache_resource
def load_cascades():
    """Charge les classificateurs Haar"""
    try:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        return face_cascade, eye_cascade, None
    except Exception as e:
        return None, None, f"Erreur cascade: {str(e)}"

def preprocess_eye(eye_frame):
    """Prétraite l'image de l'œil pour la prédiction"""
    try:
        resized = cv2.resize(eye_frame, (24, 24))
        normalized = resized / 255.0
        normalized = normalized.reshape(1, 24, 24, 1)
        return normalized
    except:
        return None

def detect_eyes_state(frame, model, face_cascade, eye_cascade):
    """Détecte l'état des yeux dans l'image"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    eye_states = []
    annotated_frame = frame.copy()
    
    for (x, y, w, h) in faces:
        # Rectangle autour du visage
        cv2.rectangle(annotated_frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        roi_gray = gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        
        for (ex, ey, ew, eh) in eyes:
            # Extraire l'œil
            eye_frame = roi_gray[ey:ey+eh, ex:ex+ew]
            
            if eye_frame.size > 0:
                processed_eye = preprocess_eye(eye_frame)
                
                if processed_eye is not None:
                    # Prédiction
                    prediction = model.predict(processed_eye, verbose=0)[0]
                    eye_state = np.argmax(prediction)  # 0: fermé, 1: ouvert
                    confidence = prediction[eye_state] * 100
                    
                    eye_states.append(eye_state)
                    
                    # Annotation sur l'image
                    eye_x, eye_y = x + ex, y + ey
                    status = "Ouvert" if eye_state == 1 else "Fermé"
                    color = (0, 255, 0) if eye_state == 1 else (0, 0, 255)
                    
                    cv2.rectangle(annotated_frame, (eye_x, eye_y), (eye_x+ew, eye_y+eh), color, 2)
                    cv2.putText(annotated_frame, f"{status}", 
                              (eye_x, eye_y-10), cv2.FONT_HERSHEY_SIMPLEX, 
                              0.6, color, 2)
    
    # Déterminer l'état global
    if eye_states:
        all_closed = all(state == 0 for state in eye_states)
        status = "YEUX FERMÉS" if all_closed else "YEUX OUVERTS"
        color = (0, 0, 255) if all_closed else (0, 255, 0)
    else:
        status = "AUCUN ŒIL DÉTECTÉ"
        color = (255, 255, 255)
        all_closed = False
    
    # Afficher le statut principal
    cv2.putText(annotated_frame, status, (50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
    return annotated_frame, all_closed

def main():
    # Titre centré
    st.markdown("<h1 style='text-align: center;'>👁️ Détection de Somnolence</h1>", 
                unsafe_allow_html=True)
    
    # Charger le modèle et les cascades
    model, model_error = load_drowsiness_model()
    face_cascade, eye_cascade, cascade_error = load_cascades()
    
    # Vérifier les erreurs
    if model_error:
        st.error(f"❌ {model_error}")
        st.info("💡 Assurez-vous que le modèle est entraîné et disponible dans 'saved_model/eye_state_model_final.h5'")
        return
    
    if cascade_error:
        st.error(f"❌ {cascade_error}")
        return
    
    st.success("✅ Modèle et cascades chargés avec succès")
    
    # Contrôles simples
    col1, col2 = st.columns(2)
    with col1:
        camera_active = st.checkbox("🎥 Activer la caméra", value=False)
    with col2:
        threshold = st.slider("⏰ Seuil d'alerte (secondes)", 1.0, 10.0, 2.5, 0.5)
    
    # Conteneur pour la vidéo
    video_container = st.empty()
    status_container = st.empty()
    
    if camera_active:
        # Variables de session
        if 'eyes_closed_start' not in st.session_state:
            st.session_state.eyes_closed_start = None
        if 'alert_active' not in st.session_state:
            st.session_state.alert_active = False
        
        # Initialiser la caméra
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            st.error("❌ Impossible d'ouvrir la caméra")
            return
        
        # Placeholder pour arrêter
        stop_button = st.button("🛑 Arrêter la caméra")
        
        while camera_active and not stop_button:
            ret, frame = cap.read()
            
            if not ret:
                st.error("❌ Erreur de capture vidéo")
                break
            
            # Traitement de l'image
            processed_frame, eyes_closed = detect_eyes_state(frame, model, face_cascade, eye_cascade)
            
            # Gestion du timing pour l'alerte
            current_time = time.time()
            
            if eyes_closed:
                if st.session_state.eyes_closed_start is None:
                    st.session_state.eyes_closed_start = current_time
                
                elapsed_time = current_time - st.session_state.eyes_closed_start
                
                if elapsed_time > threshold and not st.session_state.alert_active:
                    st.session_state.alert_active = True
                    # Ajouter un cadre rouge d'alerte
                    h, w = processed_frame.shape[:2]
                    cv2.rectangle(processed_frame, (0, 0), (w, h), (0, 0, 255), 5)
                    cv2.putText(processed_frame, "ALERTE SOMNOLENCE!", (50, 100), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                
                # Afficher le temps
                cv2.putText(processed_frame, f"Fermés: {elapsed_time:.1f}s", 
                          (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                st.session_state.eyes_closed_start = None
                st.session_state.alert_active = False
            
            # Convertir pour Streamlit
            processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            
            # Afficher l'image
            video_container.image(processed_frame_rgb, channels="RGB", use_column_width=True)
            
            # Statut en temps réel
            if st.session_state.alert_active:
                status_container.error("🚨 ALERTE SOMNOLENCE DÉTECTÉE!")
            elif eyes_closed:
                status_container.warning(f"😴 Yeux fermés depuis {elapsed_time:.1f}s")
            else:
                status_container.success("👀 Surveillance active")
            
            # Petite pause pour éviter la surcharge
            time.sleep(0.1)
        
        # Libérer la caméra
        cap.release()
        video_container.empty()
        status_container.info("📷 Caméra arrêtée")
    
    else:
        video_container.info("📷 Activez la caméra pour commencer la détection")
        status_container.empty()

if __name__ == "__main__":
    main()