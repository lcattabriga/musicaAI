import numpy as np
import scipy.io.wavfile as wav
import os
from scipy.signal import butter, lfilter

# Mappa delle note per gli accordi
note_frequencies = {
    'C': [261.63, 329.63, 392.00],  
    'C7': [261.63, 329.63, 392.00, 466.16],  
    'Dm': [293.66, 349.23, 440.00],  
    'Am': [220.00, 261.63, 329.63, 440.00],  
    'Am7': [220.00, 261.63, 329.63, 392.00],  
    'Bb': [233.08, 293.66, 349.23],  
    'F': [174.61, 220.00, 261.63, 349.23],  
}

# Mappa delle toniche degli accordi per il basso
bass_frequencies = {
    'C': 130.81,  
    'C7': 130.81,  
    'Dm': 146.83,  
    'Am': 110.00,  
    'Am7': 110.00,  
    'Bb': 116.54,  
    'F': 87.31,  
}

# Progressione di accordi con durate corrette
progression = [
    ('F', 4), ('C', 2), ('C7', 2),
    ('Dm', 4), ('Am', 2), ('Am7', 2),
    ('Bb', 4), ('F', 4), ('C', 4), ('F', 4)
]

# Impostiamo il BPM e la durata delle misure
bpm = 124
beats_per_measure = 4  
seconds_per_beat = 60 / bpm  
seconds_per_measure = beats_per_measure * seconds_per_beat  
sr = 44100  # Sample rate

# Funzione per generare un accordo con sinusoidi di base + armoniche più calde
def generate_chord(frequencies, duration=0.5, sr=44100):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    chord = sum(np.sin(2 * np.pi * f * t) + 0.5 * np.sin(2 * np.pi * (f / 2) * t) for f in frequencies)
    chord = chord / np.max(np.abs(chord))  # Normalizzazione
    return chord

# Funzione per generare un basso (segue la tonica dell'accordo)
def generate_bass(note_freq, duration=0.5, sr=44100):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    bass = np.sin(2 * np.pi * note_freq * t) * 0.8  # Volume ridotto per non coprire il resto
    return bass

# Funzione per applicare un filtro passa-basso per un suono più morbido
def lowpass_filter(audio, cutoff=1000, sr=44100, order=6):
    nyquist = 0.5 * sr
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return lfilter(b, a, audio)

# Generiamo gli strumenti separatamente
# Suono degli accordi
audio = np.concatenate([
    generate_chord(note_frequencies[chord], duration=(beats * seconds_per_beat), sr=sr) 
    for chord, beats in progression
])

# Suono del basso
bass_audio = np.concatenate([
    generate_bass(bass_frequencies[chord], duration=(beats * seconds_per_beat), sr=sr)
    for chord, beats in progression
])

# Applichiamo il filtro passa-basso per un suono più morbido sugli accordi
audio = lowpass_filter(audio, cutoff=1200, sr=sr)

# Funzione per applicare un riverbero semplice
def add_reverb(audio, sr, reverb_amount=0.25):
    impulse_response = np.exp(-np.linspace(0, 3, int(sr * 0.3)))  
    impulse_response /= np.sum(impulse_response)  
    reverb_audio = np.convolve(audio, impulse_response, mode='full')[:len(audio)]
    return (1 - reverb_amount) * audio + reverb_amount * reverb_audio  

# Funzione per aggiungere rumore bianco più leggero
def add_white_noise(audio, noise_level=0.005):  # Ridotto da 0.01 a 0.005
    noise = np.random.normal(0, noise_level, audio.shape)
    return audio + noise

# Funzione per generare un kick e un snare sintetico
def generate_drum_pattern(sr, duration, pattern=['kick', 'snare', 'kick', 'snare']):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    kick = 0.5 * np.sin(2 * np.pi * 60 * t) * np.exp(-t * 10)  
    snare = (np.random.rand(len(t)) - 0.5) * np.exp(-t * 15)  

    pattern_audio = np.zeros(int(sr * duration * len(pattern)))  

    for i, hit in enumerate(pattern):
        start = int(i * sr * duration)
        if hit == 'kick':
            pattern_audio[start:start+len(kick)] += kick
        elif hit == 'snare':
            pattern_audio[start:start+len(snare)] += snare

    return pattern_audio

# Aggiungiamo riverbero e white noise agli accordi
audio_reverb = add_reverb(audio, sr, reverb_amount=0.25)  
audio_lofi = add_white_noise(audio_reverb, noise_level=0.005)  

# Generiamo la batteria
drum_pattern = generate_drum_pattern(sr, seconds_per_beat, pattern=['kick', 'snare', 'kick', 'snare'])
drum_pattern = np.tile(drum_pattern, int(len(audio_lofi) / len(drum_pattern)))[:len(audio_lofi)]

# Mixiamo batteria, accordi e basso
final_audio = audio_lofi + drum_pattern * 0.5 + bass_audio * 0.7  # Basso leggermente più presente

# Normalizzazione finale
final_audio = final_audio / np.max(np.abs(final_audio))
final_audio = (final_audio * 32767).astype(np.int16)

# Trova la cartella Desktop in modo automatico
desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
output_file = os.path.join(desktop_path, "progression_lofi.wav")

# Salviamo il file sul Desktop (sovrascrive)
wav.write(output_file, sr, final_audio)

print(f"✅ File aggiornato con basso e salvato sul Desktop: {output_file}")
