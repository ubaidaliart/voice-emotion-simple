import streamlit as st
import numpy as np
import soundfile as sf
import librosa
import io
import tempfile
import pandas as pd
import plotly.express as px
from transformers import pipeline
import torch

# Initialize emotion detector
@st.cache_resource
def load_emotion_model():
    return pipeline(
        "audio-classification", 
        model="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
    )

# Process audio without PyDub
def process_audio(uploaded_file):
    # Read audio file directly
    audio_bytes = uploaded_file.read()
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
    
    # Load and resample audio using librosa
    y, orig_sr = librosa.load(tmp_path, sr=None, mono=True)
    y_16k = librosa.resample(y, orig_sr=orig_sr, target_sr=16000)
    
    # Save processed audio
    processed_path = tmp_path.replace(".wav", "_processed.wav")
    sf.write(processed_path, y_16k, 16000)
    
    return processed_path

# Main app
def main():
    st.set_page_config(page_title="Voice Emotion Detector", layout="wide")
    st.title("🎤 Real-Time Emotion Detection")
    st.markdown("Upload customer call recordings to detect emotions using AI")
    
    # File upload
    uploaded_file = st.file_uploader("Upload Call Recording", 
                                    type=["wav", "mp3"],
                                    accept_multiple_files=False)
    
    # Initialize session state
    if 'results' not in st.session_state:
        st.session_state.results = []
    
    if uploaded_file:
        st.audio(uploaded_file)
        
        # Process and analyze button
        if st.button("🧠 Analyze Emotion", type="primary"):
            with st.spinner("Processing audio..."):
                # Process audio
                processed_path = process_audio(uploaded_file)
                
                # Load model
                emotion_classifier = load_emotion_model()
                
                # Analyze emotion
                with st.spinner("Detecting emotion..."):
                    try:
                        preds = emotion_classifier(processed_path)
                        result = {
                            "file": uploaded_file.name,
                            "emotion": preds[0]['label'],
                            "confidence": f"{preds[0]['score']:.0%}",
                            "score": preds[0]['score']
                        }
                        
                        # Add to results
                        st.session_state.results.append(result)
                        st.success("Analysis complete!")
                        
                    except Exception as e:
                        st.error(f"Error during analysis: {str(e)}")
    
    # Display results if available
    if st.session_state.results:
        latest = st.session_state.results[-1]
        
        # Emotion display
        emotion_color = {
            "angry": "#FF4B4B",
            "happy": "#00D26A",
            "sad": "#1C83E1",
            "neutral": "#7E7E7E"
        }.get(latest['emotion'].lower(), "#7E7E7E")
        
        st.subheader("Results")
        st.markdown(f"""
        <div style="background:{emotion_color};padding:20px;border-radius:10px">
            <h2 style="color:white;text-align:center;">{latest['emotion'].upper()}</h2>
            <h3 style="color:white;text-align:center;">Confidence: {latest['confidence']}</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Emotion distribution
        if len(st.session_state.results) > 1:
            st.subheader("Emotion History")
            df = pd.DataFrame(st.session_state.results)
            fig = px.pie(df, names='emotion', title='Emotion Distribution')
            st.plotly_chart(fig)
            
            # Results table
            st.dataframe(df[['file', 'emotion', 'confidence']])
            
            # Export button
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Export Results",
                csv,
                "emotion_results.csv",
                "text/csv"
            )

if __name__ == "__main__":
    main()
