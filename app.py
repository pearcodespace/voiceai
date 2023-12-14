import streamlit as st
import os
from tortoise.models.classifier import AudioMiniEncoderWithClassifierHead
from glob import glob
import io
import librosa
import plotly.express as px
import torch
import torch.nn.functional as F
import torchaudio
import numpy as np
from scipy.io.wavfile import read

st.image('logo.svg', width=150)

def load_audio(audiopath, sampling_rate=22000):
    if isinstance(audiopath, str): # If the input is a file path
        if audiopath.endswith('.wav','.mp3'):
            audio, lsr = librosa.load(audiopath, sr=sampling_rate)
            audio = torch.FloatTensor(audio)
        else:
            assert False, f"Unsupported audio format provided: {audiopath[-4:]}"
    elif isinstance(audiopath, io.BytesIO): # If the input is file content
        audio,lsr = torchaudio.load(audiopath)
        audio = audio[0] # Remove any channel data
    if lsr != sampling_rate:
         audio = torchaudio.functional.resample(audio, lsr, sampling_rate)
    if torch.any(audio > 2) or not torch.any(audio < 0):
        print(f"Error with audio data. Max={audio.max()} min={audio.min()}")
    audio.clip_(-1,1)
    return audio.unsqueeze(0)

#function for classifier
def classify_audio_clip(clip):
    """
    Returns whether or not the classifier thinks the given clip came from AI generation.
    :param clip: torch tensor containing audio waveform data (get it from load_audio)
    :return: The probability of the audio clip being AI-generated.
    """
    classifier = AudioMiniEncoderWithClassifierHead(2, spec_dim=1, embedding_dim=512, depth=5, downsample_factor=4,
                                                    resnet_blocks=2, attn_blocks=4, num_attn_heads=4, base_channels=32,
                                                    dropout=0, kernel_size=5, distribute_zero_label=False)
    state_dict = torch.load('classifier.pth', map_location=torch.device('cpu'))
    classifier.load_state_dict(state_dict)
    clip = clip.cpu().unsqueeze(0)
    results = F.softmax(classifier(clip), dim=-1)
    return results[0][0]

def main():
    col1, col2 = st.columns([5, 6])
    with col1:
        st.subheader("AI Deep Voice Detection")
        st.caption("AI Voice Detector: Uncover the Truth! \n"
                "Protect yourself from Voice Cloning \n"
                "Scams. Don't be a victim -you deserve \n"
                "authenticity!")
    with col2:
        #file uploader
        uploaded_file = st.file_uploader("Input your audio", type=["wav","mp3"])
    
    if uploaded_file is not None:
        if st.button("Analyze Audio"):
            col3, col4 = st.columns([1, 3])
            
            with col3:
                #load and classify the audio file
                audio_clip = load_audio(uploaded_file)
                result = classify_audio_clip(audio_clip)
                result = result.item()
                st.info(f"Result probability: {result}")
                st.success(f"The uploaded audio is {result * 100:.2f}% likely to be AI Generated.")
                st.balloons()
           
            with col4:
                st.text("Your uploaded audio is below")
                st.audio(uploaded_file)
                #create a waveform
                fig = px.line()
                fig.add_scatter(x=list(range(len(audio_clip.squeeze()))), y=audio_clip.squeeze())
                fig.update_layout(
                    title="Waveform Plot",
                    xaxis_title = "Time",
                    yaxis_title="Amplitude"
                )
                st.plotly_chart(fig, use_container_width=True)

if __name__=="__main__":
    main()