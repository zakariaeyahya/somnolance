"""
Service de gestion du dashboard et visualisations
Interface utilisateur et graphiques temps réel
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class DashboardService:
    """Service pour la gestion du dashboard et des visualisations"""
    
    def __init__(self, analytics_service):
        self.analytics = analytics_service
        self.color_scheme = {
            'primary': '#00D4FF',
            'secondary': '#FF6B6B', 
            'accent': '#4ECDC4',
            'success': '#00FF88',
            'warning': '#FFB800',
            'danger': '#FF4757',
            'bg_dark': '#0E1117',
            'bg_card': '#1E2329',
            'text_light': '#FAFAFA'
        }
    
    def create_main_header(self):
        """Crée l'en-tête principal du dashboard"""
        st.markdown('''
        <div style="text-align: center; margin-bottom: 2rem;">
            <h1 style="
                font-family: 'Orbitron', monospace;
                font-size: 3rem;
                font-weight: 900;
                background: linear-gradient(45deg, #00D4FF, #4ECDC4);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
                text-shadow: 0 0 30px rgba(0, 212, 255, 0.5);
            ">🚗 GUARDIAN EYE</h1>
            <p style="
                color: #4ECDC4;
                font-size: 1.2rem;
                margin-top: -1rem;
            ">Système Intelligent de Détection de Somnolence</p>
        </div>
        ''', unsafe_allow_html=True)
    
    def create_metrics_cards(self):
        """Crée les cartes de métriques principales"""
        summary = self.analytics.get_session_summary()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            self._create_metric_card(
                title="Alertes Déclenchées",
                value=summary['total_alerts'],
                color=self.color_scheme['danger'],
                icon="🚨"
            )
        
        with col2:
            session_minutes = int(summary['session_duration'] / 60)
            self._create_metric_card(
                title="Minutes de Surveillance", 
                value=session_minutes,
                color=self.color_scheme['accent'],
                icon="⏱️"
            )
        
        with col3:
            self._create_metric_card(
                title="Max Somnolence",
                value=f"{summary['max_drowsiness_duration']:.1f}s",
                color=self.color_scheme['warning'],
                icon="😴"
            )
        
        with col4:
            self._create_metric_card(
                title="Score Vigilance",
                value=f"{summary['vigilance_score']:.0f}/100",
                color=self.color_scheme['success'],
                icon="👁️"
            )
    
    def _create_metric_card(self, title, value, color, icon=""):
        """Crée une carte de métrique individuelle"""
        st.markdown(f'''
        <div style="
            background: rgba(30, 35, 41, 0.8);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(0, 212, 255, 0.3);
            border-radius: 15px;
            padding: 20px;
            margin: 10px 0;
            box-shadow: 0 8px 32px rgba(0, 212, 255, 0.1);
            transition: all 0.3s ease;
            text-align: center;
        ">
            <div style="
                font-family: 'Orbitron', monospace;
                font-size: 2rem;
                font-weight: 700;
                color: {color};
                text-shadow: 0 0 10px {color};
            ">{icon} {value}</div>
            <div style="
                color: #FAFAFA;
                font-size: 0.9rem;
                margin-top: 5px;
            ">{title}</div>
        </div>
        ''', unsafe_allow_html=True)
    
    def create_status_display(self, drowsiness_analysis, alert_active=False):
        """Crée l'affichage du statut principal"""
        if alert_active:
            status_html = f'''
            <div style="
                background: linear-gradient(45deg, #FF4757, #FF6B6B);
                color: white;
                padding: 20px;
                border-radius: 15px;
                text-align: center;
                font-weight: bold;
                font-size: 1.5rem;
                animation: shake 0.5s ease-in-out infinite, pulse 1s ease-in-out infinite;
                box-shadow: 0 0 50px rgba(255, 71, 87, 0.5);
                margin: 20px 0;
            ">
                🚨 ALERTE SOMNOLENCE ACTIVE! 🔊<br>
                <span style="font-size: 1.2rem;">
                    Niveau: {drowsiness_analysis.get('drowsiness_level', 0):.1f}%
                </span>
            </div>
            '''
        elif drowsiness_analysis.get('any_eye_closed', False):
            status_html = f'''
            <div style="
                background: linear-gradient(45deg, #FFB800, #FF6B6B);
                color: white;
                padding: 15px;
                border-radius: 10px;
                text-align: center;
                font-weight: bold;
                margin: 20px 0;
            ">
                😴 Yeux détectés fermés<br>
                <span style="font-size: 1rem;">
                    Niveau de somnolence: {drowsiness_analysis.get('drowsiness_level', 0):.1f}%
                </span>
            </div>
            '''
        else:
            blink_freq = self.analytics.calculate_blink_frequency()
            drowsiness_level = drowsiness_analysis.get('drowsiness_level', 0)
            status_html = f'''
            <div style="
                background: linear-gradient(45deg, #00FF88, #4ECDC4);
                color: white;
                padding: 15px;
                border-radius: 10px;
                text-align: center;
                font-weight: bold;
                animation: pulse 2s ease-in-out infinite;
                margin: 20px 0;
            ">
                👀 Surveillance Active | Niveau: {drowsiness_level:.1f}% | Clignements/min: {blink_freq:.1f}
            </div>
            '''
        
        st.markdown(status_html, unsafe_allow_html=True)
    
    def create_realtime_charts(self):
        """Crée les graphiques en temps réel"""
        if len(self.analytics.detection_history) < 5:
            st.info("📊 Collecte de données en cours... (minimum 5 points requis)")
            return
        
        # Préparer les données des 60 dernières secondes
        recent_data = list(self.analytics.detection_history)[-60:]
        
        if not recent_data:
            return
        
        timestamps = [d['timestamp'] for d in recent_data]
        drowsiness_levels = []
        
        # Calculer les niveaux de somnolence par fenêtre glissante
        for i, _ in enumerate(recent_data):
            window_data = recent_data[max(0, i-10):i+1]  # Fenêtre de 10 points
            drowsy_count = sum(1 for d in window_data if d['eyes_closed'])
            level = (drowsy_count / len(window_data)) * 100 if window_data else 0
            drowsiness_levels.append(level)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Graphique de tendance
            fig_trend = go.Figure()
            fig_trend.add_trace(go.Scatter(
                x=timestamps,
                y=drowsiness_levels,
                mode='lines+markers',
                name='Niveau de Somnolence',
                line=dict(color=self.color_scheme['primary'], width=3),
                marker=dict(size=6, color=self.color_scheme['secondary']),
                fill='tonexty',
                fillcolor=f"rgba(0, 212, 255, 0.1)"
            ))
            
            fig_trend.update_layout(
                title="🌊 Tendance de Somnolence (Temps Réel)",
                xaxis_title="Temps",
                yaxis_title="Niveau (%)",
                template='plotly_dark',
                height=400,
                showlegend=False,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            
            st.plotly_chart(fig_trend, use_container_width=True)
        
        with col2:
            # Jauge de niveau actuel
            current_level = drowsiness_levels[-1] if drowsiness_levels else 0
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=current_level,
                title={'text': "🎯 Niveau Actuel", 'font': {'color': 'white'}},
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={
                    'axis': {'range': [None, 100], 'tickcolor': 'white'},
                    'bar': {'color': self.color_scheme['primary']},
                    'steps': [
                        {'range': [0, 30], 'color': self.color_scheme['success']},
                        {'range': [30, 70], 'color': self.color_scheme['warning']},
                        {'range': [70, 100], 'color': self.color_scheme['danger']}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 80
                    }
                },
                number={'font': {'color': 'white'}}
            ))
            
            fig_gauge.update_layout(
                template='plotly_dark',
                height=300,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            
            st.plotly_chart(fig_gauge, use_container_width=True)
    
    def create_advanced_analytics(self):
        """Crée les analyses avancées"""
        summary = self.analytics.get_session_summary()
        
        st.markdown("### 📈 Analyses Avancées")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Indice de risque
            risk_index = summary['risk_index']
            risk_color = self._get_risk_color(risk_index)
            
            st.markdown(f'''
            <div style="
                background: rgba(30, 35, 41, 0.8);
                border-radius: 10px;
                padding: 15px;
                text-align: center;
                border: 2px solid {risk_color};
            ">
                <h4 style="color: {risk_color};">⚠️ Indice de Risque</h4>
                <div style="font-size: 2rem; color: {risk_color}; font-weight: bold;">
                    {risk_index:.1f}/100
                </div>
                <div style="color: white; font-size: 0.9rem;">
                    {self._get_risk_level_text(risk_index)}
                </div>
            </div>
            ''', unsafe_allow_html=True)
        
        with col2:
            # Probabilité d'incident
            incident_prob = summary['incident_probability'] * 100
            prob_color = self._get_risk_color(incident_prob)
            
            st.markdown(f'''
            <div style="
                background: rgba(30, 35, 41, 0.8);
                border-radius: 10px;
                padding: 15px;
                text-align: center;
                border: 2px solid {prob_color};
            ">
                <h4 style="color: {prob_color};">🎯 Probabilité d'Incident</h4>
                <div style="font-size: 2rem; color: {prob_color}; font-weight: bold;">
                    {incident_prob:.1f}%
                </div>
                <div style="color: white; font-size: 0.9rem;">
                    Basé sur les modèles prédictifs
                </div>
            </div>
            ''', unsafe_allow_html=True)
        
        with col3:
            # Fatigue cumulée
            cumulative_fatigue = summary['cumulative_fatigue']
            fatigue_color = self._get_risk_color(cumulative_fatigue)
            
            st.markdown(f'''
            <div style="
                background: rgba(30, 35, 41, 0.8);
                border-radius: 10px;
                padding: 15px;
                text-align: center;
                border: 2px solid {fatigue_color};
            ">
                <h4 style="color: {fatigue_color};">😵 Fatigue Cumulée</h4>
                <div style="font-size: 2rem; color: {fatigue_color}; font-weight: bold;">
                    {cumulative_fatigue:.1f}%
                </div>
                <div style="color: white; font-size: 0.9rem;">
                    Persistance de la fatigue
                </div>
            </div>
            ''', unsafe_allow_html=True)
    
    def create_trend_analysis_chart(self):
        """Crée le graphique d'analyse des tendances"""
        trend_data = self.analytics.get_trend_analysis(window_minutes=10)
        
        if trend_data['trend'] == 'insufficient_data':
            st.info("📊 Données insuffisantes pour l'analyse des tendances")
            return
        
        # Créer un graphique de régression
        recent_data = list(self.analytics.detection_history)[-100:]  # 100 derniers points
        
        if len(recent_data) < 10:
            return
        
        # Préparer les données
        timestamps = [(d['timestamp'] - recent_data[0]['timestamp']).total_seconds() for d in recent_data]
        drowsiness_values = [1 if d['eyes_closed'] else 0 for d in recent_data]
        
        # Calculer la ligne de tendance
        z = np.polyfit(timestamps, drowsiness_values, 1)
        p = np.poly1d(z)
        trend_line = p(timestamps)
        
        # Créer le graphique
        fig = go.Figure()
        
        # Points de données
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=drowsiness_values,
            mode='markers',
            name='Détections',
            marker=dict(color=self.color_scheme['primary'], size=8, opacity=0.6)
        ))
        
        # Ligne de tendance
        trend_color = self._get_trend_color(trend_data['trend'])
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=trend_line,
            mode='lines',
            name=f'Tendance ({trend_data["trend"]})',
            line=dict(color=trend_color, width=3)
        ))
        
        fig.update_layout(
            title=f"📈 Analyse des Tendances - {trend_data['trend'].upper()}",
            xaxis_title="Temps (secondes)",
            yaxis_title="État de Somnolence",
            template='plotly_dark',
            height=400,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Afficher les statistiques de tendance
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Tendance", trend_data['trend'].replace('_', ' ').title())
        with col2:
            st.metric("Pente", f"{trend_data['slope']:.4f}")
        with col3:
            st.metric("Corrélation", f"{trend_data['correlation']:.3f}")
    
    def create_session_report(self):
        """Crée le rapport de session détaillé"""
        summary = self.analytics.get_session_summary()
        
        st.markdown("### 📊 Rapport de Session")
        
        # Métriques de base
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Durée Totale", f"{summary['session_duration']/60:.0f} min")
            st.metric("Frames Totales", summary['total_frames'])
        
        with col2:
            st.metric("Clignements", summary['total_blinks'])
            st.metric("Fréq. Clignement", f"{summary['blink_frequency']:.1f}/min")
        
        with col3:
            st.metric("Alertes", summary['total_alerts'])
            st.metric("Alertes/Heure", f"{summary['alerts_per_hour']:.1f}")
        
        with col4:
            st.metric("% Somnolence", f"{summary['drowsiness_percentage']:.1f}%")
            st.metric("Seuil Adaptatif", f"{summary['adaptive_threshold']:.1f}s")
        
        # Évaluation de la conduite
        vigilance_score = summary['vigilance_score']
        driving_quality, quality_color, recommendations = self._evaluate_driving_quality(vigilance_score, summary)
        
        st.markdown(f'''
        <div style="
            background: linear-gradient(45deg, rgba(30, 35, 41, 0.8), rgba(30, 35, 41, 0.6));
            border-radius: 15px;
            padding: 20px;
            margin: 20px 0;
            border-left: 5px solid {quality_color};
        ">
            <h4 style="color: {quality_color};">🏆 Évaluation de la Conduite: {driving_quality}</h4>
            <p style="color: white;">Score de Vigilance: {vigilance_score:.0f}/100</p>
            <div style="color: #CCCCCC;">
                <strong>Recommandations:</strong><br>
                {recommendations}
            </div>
        </div>
        ''', unsafe_allow_html=True)
    
    def create_control_panel(self):
        """Crée le panneau de contrôle"""
        st.sidebar.markdown("## ⚙️ Contrôles Avancés")
        
        # Profils prédéfinis
        profile = st.sidebar.selectbox(
            "🎯 Profil de Conduite",
            ["Standard", "Conduite Urbaine", "Autoroute", "Nuit", "Personnalisé"],
            help="Sélectionnez un profil adapté à votre situation"
        )
        
        # Paramètres selon le profil
        profile_settings = self._get_profile_settings(profile)
        
        threshold = st.sidebar.slider(
            "⏰ Seuil d'Alerte (secondes)",
            0.5, 10.0, profile_settings['threshold'], 0.1,
            help="Durée avant déclenchement de l'alerte"
        )
        
        sensitivity = st.sidebar.slider(
            "🎚️ Sensibilité",
            0.3, 2.0, profile_settings['sensitivity'], 0.1,
            help="Ajuste la sensibilité de détection"
        )
        
        # Options avancées
        st.sidebar.markdown("## 🚀 Fonctionnalités")
        
        alarm_progression = st.sidebar.checkbox(
            "📈 Alarme Progressive", value=True,
            help="L'alarme s'intensifie avec le temps"
        )
        
        voice_alerts = st.sidebar.checkbox(
            "🔊 Alertes Vocales", value=False,
            help="Active les messages vocaux"
        )
        
        smart_pause = st.sidebar.checkbox(
            "⏸️ Pause Intelligente", value=True,
            help="Suggère des pauses régulières"
        )
        
        analytics_display = st.sidebar.checkbox(
            "📊 Analytics Temps Réel", value=True,
            help="Affiche les graphiques en temps réel"
        )
        
        return {
            'profile': profile,
            'threshold': threshold,
            'sensitivity': sensitivity,
            'alarm_progression': alarm_progression,
            'voice_alerts': voice_alerts,
            'smart_pause': smart_pause,
            'analytics_display': analytics_display
        }
    
    def _get_risk_color(self, value):
        """Retourne la couleur selon le niveau de risque"""
        if value < 30:
            return self.color_scheme['success']
        elif value < 60:
            return self.color_scheme['warning']
        else:
            return self.color_scheme['danger']
    
    def _get_risk_level_text(self, risk_index):
        """Retourne le texte du niveau de risque"""
        if risk_index < 20:
            return "Risque Faible ✅"
        elif risk_index < 40:
            return "Risque Modéré ⚠️"
        elif risk_index < 70:
            return "Risque Élevé 🔶"
        else:
            return "Risque Critique 🚨"
    
    def _get_trend_color(self, trend):
        """Retourne la couleur selon la tendance"""
        if trend == 'decreasing':
            return self.color_scheme['success']
        elif trend == 'stable':
            return self.color_scheme['primary']
        else:
            return self.color_scheme['danger']
    
    def _evaluate_driving_quality(self, vigilance_score, summary):
        """Évalue la qualité de conduite et fournit des recommandations"""
        if vigilance_score >= 85:
            quality = "Excellente 🌟"
            color = self.color_scheme['success']
            recommendations = "Continuez ainsi ! Votre vigilance est exemplaire."
        elif vigilance_score >= 70:
            quality = "Bonne 👍"
            color = self.color_scheme['accent']
            recommendations = "Bonne conduite. Maintenez votre attention et prenez des pauses régulières."
        elif vigilance_score >= 50:
            quality = "Attention Requise ⚠️"
            color = self.color_scheme['warning']
            recommendations = "Soyez plus vigilant. Considérez une pause de 15-20 minutes."
        else:
            quality = "Préoccupante 🚨"
            color = self.color_scheme['danger']
            recommendations = "ARRÊTEZ-VOUS dès que possible. Votre état nécessite du repos."
        
        return quality, color, recommendations
    
    def _get_profile_settings(self, profile):
        """Retourne les paramètres selon le profil sélectionné"""
        profiles = {
            "Standard": {'threshold': 2.5, 'sensitivity': 1.0},
            "Conduite Urbaine": {'threshold': 1.5, 'sensitivity': 0.8},
            "Autoroute": {'threshold': 1.0, 'sensitivity': 1.2},
            "Nuit": {'threshold': 2.0, 'sensitivity': 0.6},
            "Personnalisé": {'threshold': 2.5, 'sensitivity': 1.0}
        }
        return profiles.get(profile, profiles["Standard"])