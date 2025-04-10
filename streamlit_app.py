# streamlit_app.py
import streamlit as st
import numpy as np
from itertools import product
from scipy.signal import butter, lfilter
from io import BytesIO
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# --- 기본 설정 ---
note_sequence = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
note_to_index = {note: i for i, note in enumerate(note_sequence)}
def note_from_fret(open_note, fret):
    start = note_to_index[open_note]
    return note_sequence[(start + fret) % 12]

def get_frequency(note, octave=4):
    base_freq = 440.0  # A4
    semitone_distance = note_to_index[note] - note_to_index['A'] + (octave - 4) * 12
    return base_freq * (2 ** (semitone_distance / 12))

# --- Streamlit UI ---
st.title("🎸Custom Tuning Guitar Chord Generator")
st.caption("Create chord voicings for any tuning")

st.subheader("1. Set Your Tuning (6th to 1st string)")
tuning = []
open_frequencies = []
tuning_columns = st.columns(6)
def_octaves = [2, 2, 3, 3, 3, 4]  # E2 A2 D3 G3 B3 E4
preset = st.selectbox("🎼 Choose a tuning preset", ["Standard E (EADGBE)", "Drop D (DADGBE)", "Custom"])

if preset == "Standard E (EADGBE)":
    default_notes = ['E', 'A', 'D', 'G', 'B', 'E']
elif preset == "Drop D (DADGBE)":
    default_notes = ['D', 'A', 'D', 'G', 'B', 'E']
else:
    default_notes = ['E', 'A', 'D', 'G', 'B', 'E']

for i in range(6):
    note = tuning_columns[i].selectbox(f"{6 - i} string", note_sequence, index=note_sequence.index(default_notes[i]))
    tuning.append(note)
    open_frequencies.append(get_frequency(note, def_octaves[i]))

allowed_mute_indices = {0, 1, 5}  # 6,5,1번 줄만 음소거 가능

chord_root = st.selectbox("Chord Root", note_sequence)
chord_type = st.selectbox("Chord Type", list({ 
    "maj": [0, 4, 7],
    "add2": [0, 2, 4, 7],
    "add#4" : [0, 4, 6, 7],
    "sus2": [0, 2, 7],
    "sus4": [0, 5, 7],
    "add6": [0, 4, 7, 9],
    "dim": [0, 3, 6],
    "dim7": [0, 3, 6, 9],
    "aug": [0, 4, 8],
    "maj7": [0, 4, 7, 11],
    "maj9": [0, 4, 7, 11, 14],
    "maj#11": [0, 4, 7, 11, 18],
    "maj9,#11": [0, 4, 7, 11, 14, 18],
    "maj13": [0, 4, 7, 11, 21],
    "maj9,13": [0, 4, 7, 11, 14, 21],
    "maj#11,13": [0, 4, 7, 11, 18, 21],
    "maj9,#11,13": [0, 4, 7, 11, 14, 18, 21],                
    "min": [0, 3, 7],
    "min7": [0, 3, 7, 10],
    "min_maj7": [0, 3, 7, 11],
    "min9": [0, 3, 7, 10, 14],
    "min11": [0, 3, 7, 10, 17],
    "min9,11": [0, 3, 7, 10, 14, 17],
    "min13": [0, 3, 7, 10, 21],
    "min9,13": [0, 3, 7, 10, 14, 21],
    "min11,13": [0, 3, 7, 10, 17, 21],
    "min9,11,13 :": [0, 3, 7, 10, 14, 17, 21],
    "7": [0, 4, 7, 10],
    "7,b9": [0, 4, 7, 10, 13],
    "7,9": [0, 4, 7, 10, 14],
    "7,#9": [0, 4, 7, 10, 15],
    "7,b9,#9": [0, 4, 7, 10, 13, 15],
    "7,#11": [0, 4, 7, 10, 18],
    "7,b9,#11": [0, 4, 7, 10, 13, 18],
    "7,9,#11" :[0, 4, 7, 10, 14, 18], 
    "7,#9,#11": [0, 4, 7, 10, 15, 18],
    "7,b13": [0, 4, 7, 10, 20],
    "7,b9,b13" : [0, 4, 7, 10, 13, 20],
    "7,9,b13": [0, 4, 7, 10, 14, 20],
    "7,#9,b13": [0, 4, 7, 10, 15, 20],
    "7,#11,b13": [0, 4, 7, 10, 18, 20],
    "7,9,#11,b13": [0, 4, 7, 10, 14, 18, 20],
    "7,13": [0, 4, 7, 10, 21],
    "7,b9,13": [0, 4, 7, 10, 13, 21],
    "7,9,13": [0, 4, 7, 10, 14, 21],
    "7,#9,13": [0, 4, 7, 10, 15, 21],
    "7,#11,13": [0, 4, 7, 10, 18, 21],
    "7,9,#11,13": [0, 4, 7, 10, 14, 18, 21],
    "7,b9,#11,13": [0, 4, 7, 10, 13, 18, 21],
    "7,#9,#11,13": [0, 4, 7, 10, 15, 18, 21],
    "sus4,7": [0, 5, 7, 10],
    "sus4,7,b9": [0, 5, 7, 10, 13],
    "sus4,7,b9,13": [0, 5, 7, 10, 13, 21],
    "sus4,7,b9,b13": [0, 5, 7, 10, 13, 20],
    "sus4,7,9": [0, 5, 7, 10, 14],
    "sus4,7,9,13": [0, 5, 7, 10, 14, 21],
    "sus4,7,9,b13": [0, 5, 7, 10, 14, 20]
}.keys()))
mode = st.radio("Display Mode", ["Show all voicings", "Show best voicing only"])

# --- 코드 폼 ---
chord_formulas = { 
    "maj": [0, 4, 7],
    "add2": [0, 2, 4, 7],
    "add#4" : [0, 4, 6, 7],
    "sus2": [0, 2, 7],
    "sus4": [0, 5, 7],
    "add6": [0, 4, 7, 9],
    "dim": [0, 3, 6],
    "dim7": [0, 3, 6, 9],
    "aug": [0, 4, 8],
    "maj7": [0, 4, 7, 11],
    "maj9": [0, 4, 7, 11, 14],
    "maj#11": [0, 4, 7, 11, 18],
    "maj9,#11": [0, 4, 7, 11, 14, 18],
    "maj13": [0, 4, 7, 11, 21],
    "maj9,13": [0, 4, 7, 11, 14, 21],
    "maj#11,13": [0, 4, 7, 11, 18, 21],
    "maj9,#11,13": [0, 4, 7, 11, 14, 18, 21],                
    "min": [0, 3, 7],
    "min7": [0, 3, 7, 10],
    "min_maj7": [0, 3, 7, 11],
    "min9": [0, 3, 7, 10, 14],
    "min11": [0, 3, 7, 10, 17],
    "min9,11": [0, 3, 7, 10, 14, 17],
    "min13": [0, 3, 7, 10, 21],
    "min9,13": [0, 3, 7, 10, 14, 21],
    "min11,13": [0, 3, 7, 10, 17, 21],
    "min9,11,13 :": [0, 3, 7, 10, 14, 17, 21],
    "7": [0, 4, 7, 10],
    "7,b9": [0, 4, 7, 10, 13],
    "7,9": [0, 4, 7, 10, 14],
    "7,#9": [0, 4, 7, 10, 15],
    "7,b9,#9": [0, 4, 7, 10, 13, 15],
    "7,#11": [0, 4, 7, 10, 18],
    "7,b9,#11": [0, 4, 7, 10, 13, 18],
    "7,9,#11" :[0, 4, 7, 10, 14, 18], 
    "7,#9,#11": [0, 4, 7, 10, 15, 18],
    "7,b13": [0, 4, 7, 10, 20],
    "7,b9,b13" : [0, 4, 7, 10, 13, 20],
    "7,9,b13": [0, 4, 7, 10, 14, 20],
    "7,#9,b13": [0, 4, 7, 10, 15, 20],
    "7,#11,b13": [0, 4, 7, 10, 18, 20],
    "7,9,#11,b13": [0, 4, 7, 10, 14, 18, 20],
    "7,13": [0, 4, 7, 10, 21],
    "7,b9,13": [0, 4, 7, 10, 13, 21],
    "7,9,13": [0, 4, 7, 10, 14, 21],
    "7,#9,13": [0, 4, 7, 10, 15, 21],
    "7,#11,13": [0, 4, 7, 10, 18, 21],
    "7,9,#11,13": [0, 4, 7, 10, 14, 18, 21],
    "7,b9,#11,13": [0, 4, 7, 10, 13, 18, 21],
    "7,#9,#11,13": [0, 4, 7, 10, 15, 18, 21],
    "sus4,7": [0, 5, 7, 10],
    "sus4,7,b9": [0, 5, 7, 10, 13],
    "sus4,7,b9,13": [0, 5, 7, 10, 13, 21],
    "sus4,7,b9,b13": [0, 5, 7, 10, 13, 20],
    "sus4,7,9": [0, 5, 7, 10, 14],
    "sus4,7,9,13": [0, 5, 7, 10, 14, 21],
    "sus4,7,9,b13": [0, 5, 7, 10, 14, 20]
}
guide_tone_intervals = {
    "maj": [0, 4, 7],
    "add2": [0, 2, 7],
    "add#4" : [0, 6, 7],
    "sus2": [0, 2, 7],
    "sus4": [0, 5, 7],
    "add6": [0, 4, 9],
    "dim": [0, 3, 6],
    "dim7": [0, 3, 6, 9],
    "aug": [0, 4, 8],
    "maj7": [0, 4, 11],
    "maj9": [0, 4, 11, 14],
    "maj#11": [0, 4, 11, 18],
    "maj9,#11": [0, 11, 14, 18],
    "maj13": [0, 4, 11, 21],
    "maj9,13": [0, 11, 14, 21],
    "maj#11,13": [0, 11, 18, 21],
    "maj9,#11,13": [0, 11, 14, 18, 21],                
    "min": [0, 3, 7],
    "min7": [0, 3, 10],
    "min_maj7": [0, 3, 7, 11],
    "min9": [0, 3, 10, 14],
    "min11": [0, 3, 10, 17],
    "min9,11": [0, 10, 14, 17],
    "min13": [0, 3, 10, 21],
    "min9,13": [0, 10, 14, 21],
    "min11,13": [0, 10, 17, 21],
    "min9,11,13 :": [0, 10, 14, 17, 21],
    "7": [0, 4, 7, 10],
    "7,b9": [0, 4, 10, 13],
    "7,9": [0, 4, 10, 14],
    "7,#9": [0, 4, 10, 15],
    "7,b9,#9": [0, 10, 13, 15],
    "7,#11": [0, 4, 10, 18],
    "7,b9,#11": [0, 10, 13, 18],
    "7,9,#11" :[0, 10, 14, 18], 
    "7,#9,#11": [0, 10, 15, 18],
    "7,b13": [0, 4, 10, 20],
    "7,b9,b13" : [0, 10, 13, 20],
    "7,9,b13": [0, 10, 14, 20],
    "7,#9,b13": [0, 10, 15, 20],
    "7,#11,b13": [0, 10, 18, 20],
    "7,9,#11,b13": [0, 10, 14, 18, 20],
    "7,13": [0, 4, 10, 21],
    "7,b9,13": [0, 10, 13, 21],
    "7,9,13": [0, 10, 14, 21],
    "7,#9,13": [0, 10, 15, 21],
    "7,#11,13": [0, 10, 18, 21],
    "7,9,#11,13": [0, 10, 14, 18, 21],
    "7,b9,#11,13": [0, 10, 13, 18, 21],
    "7,#9,#11,13": [0, 10, 15, 18, 21],
    "sus4,7": [0, 5, 7, 10],
    "sus4,7,b9": [0, 5, 10, 13],
    "sus4,7,b9,13": [0, 5, 10, 13, 21],
    "sus4,7,b9,b13": [0, 5, 10, 13, 20],
    "sus4,7,9": [0, 5, 10, 14],
    "sus4,7,9,13": [0, 5, 10, 14, 21],
    "sus4,7,9,b13": [0, 5, 10, 14, 20]
}

def get_full_chord_tones(chord_root, chord_type):
    intervals = chord_formulas[chord_type]
    root_index = note_to_index[chord_root]
    return {note_sequence[(root_index + iv) % 12] for iv in intervals}

def get_guide_tones(chord_root, chord_type):
    intervals = guide_tone_intervals[chord_type]
    root_index = note_to_index[chord_root]
    return {note_sequence[(root_index + iv) % 12] for iv in intervals}

def generate_voicings(chord_root, chord_type, max_fret=14, allow_muted=True):
    full_chord = get_full_chord_tones(chord_root, chord_type)
    required_tones = get_guide_tones(chord_root, chord_type)
    possible_options = []
    for idx, open_note in enumerate(tuning):
        opts = [(fret, note_from_fret(open_note, fret)) for fret in range(max_fret+1)
                if note_from_fret(open_note, fret) in full_chord]
        if allow_muted and idx in allowed_mute_indices:
            opts.append(("x", None))
        possible_options.append(opts)
    valid_voicings = []
    for comb in product(*possible_options):
        played = [c for c in comb if c[0] != "x"]
        if len(played) < 4: continue
        produced = {c[1] for c in played}
        if not required_tones.issubset(produced): continue
        frets = [c[0] for c in played if isinstance(c[0], int)]
        span = max(frets) - min(frets)
        if span > 3: continue
        lower_indices = [i for i in [0,1,2] if comb[i][0] != "x"]
        if not lower_indices: continue
        if comb[min(lower_indices)][1] != chord_root: continue
        frets_sum = sum(f for f, _ in played if isinstance(f, int))
        valid_voicings.append((comb, span, min(lower_indices), frets_sum))
    valid_voicings.sort(key=lambda x: (x[2], x[1], x[3]))
    return valid_voicings

def print_voicing(voicing):
    result = []
    for i, opt in enumerate(voicing):
        string_label = f"{6-i} ({tuning[i]})"
        fret, note = opt
        result.append(f"{string_label}: {'x' if fret == 'x' else f'{fret} → {note}' }")
    return "\n".join(result)

def draw_diagram(voicing, idx):
    fig, ax = plt.subplots(figsize=(5, 2.8), dpi=200)
    ax.set_title(f"Voicing {idx}", fontsize=10)
    ax.set_xlim(0.5, 6.5)
    ax.set_ylim(-1.5, 5)
    ax.set_xticks(range(1, 7))
    ax.set_xticklabels([f"{6 - i}" for i in range(6)])
    ax.set_yticks(range(0, 5))
    ax.set_yticklabels([f"Fret {i}" for i in range(0, 5)])
    ax.invert_yaxis()
    for i, (fret, note) in enumerate(voicing):
        if fret == "x":
            ax.text(i + 1, -1.0, 'x', ha='center', va='center', fontsize=12, color='red')
        elif fret == 0:
            ax.plot(i + 1, 0, 'o', color='black')
            ax.text(i + 1, 0.4, note, ha='center', fontsize=7)
        elif fret <= 4:
            color = 'orange' if note == chord_root else 'black'
            ax.plot(i + 1, fret, 'o', color=color)
            ax.text(i + 1, fret + 0.4, note, ha='center', fontsize=7)
    ax.axis('off')
    return fig

# --- 이하 동일 ---
