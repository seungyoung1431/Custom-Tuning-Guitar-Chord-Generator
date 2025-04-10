# streamlit_app.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from scipy.signal import butter, lfilter

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
st.title("🎸 Custom Tuning Guitar Chord Generator")
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
    "maj": [0, 4, 7], "min": [0, 3, 7], "dim7": [0, 3, 6, 9], "7": [0, 4, 7, 10], "maj7": [0, 4, 7, 11]
}.keys()))
mode = st.radio("Display Mode", ["Show all voicings", "Show best voicing only"])

# --- Generate Voicings Always Available ---
if st.button("🎵 Generate Voicings"):
    chord_formulas = {
        "maj": [0, 4, 7], "min": [0, 3, 7], "dim7": [0, 3, 6, 9], "7": [0, 4, 7, 10], "maj7": [0, 4, 7, 11]
    }
    guide_tone_intervals = {
        "maj": [0, 4], "min": [0, 3], "dim7": [0, 3, 6], "7": [0, 4, 10], "maj7": [0, 4, 11]
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

    def draw_fretboard_diagram(voicing, chord_name):
        fig, ax = plt.subplots(figsize=(4, 5))
        ax.set_facecolor('white')
        ax.set_xlim(0.5, 6.5)
        ax.set_ylim(0, 5)
        ax.axis('off')

        # Fret lines (horizontal)
        for y in range(1, 5):
            ax.plot([0.5, 6.5], [y, y], color='black', lw=1)
        # String lines (vertical)
        for x in range(1, 7):
            ax.plot([x, x], [0, 5], color='black', lw=3 if x == 1 else 1)

        min_fret = min([fret for fret, note in voicing if fret != 'x'])

        for i, (fret, note) in enumerate(voicing):
            x = 6 - i
            if fret != 'x':
                y = fret - min_fret + 1
                ax.plot(x, y, 'o', markersize=14, color='black', zorder=2)
                ax.text(x, y, note, fontsize=9, color='white', ha='center', va='center', zorder=3)
            else:
                ax.text(x, 4.6, 'x', fontsize=10, ha='center', va='center', color='black')

        ax.text(6.7, 2.5, chord_name, fontsize=12, rotation=270, va='center')
        st.pyplot(fig)

    def print_voicing(voicing):
        result = []
        for i, opt in enumerate(voicing):
            string_label = f"{6-i} ({tuning[i]})"
            fret, note = opt
            result.append(f"{string_label}: {'x' if fret == 'x' else f'{fret} → {note}' }")
        return "\n".join(result)

    def bandpass_filter(data, rate, low, high):
        nyq = 0.5 * rate
        b, a = butter(4, [low/nyq, high/nyq], btype='band')
        return lfilter(b, a, data)

    def envelope(duration, rate, attack_ms, release_ms):
        samples = int(duration * rate)
        a, r = int(attack_ms / 1000 * rate), int(release_ms / 1000 * rate)
        s = samples - a - r
        return np.concatenate([
            np.linspace(0, 1, a),
            np.ones(s),
            np.linspace(1, 0, r)
        ]) if s > 0 else np.concatenate([np.linspace(0, 1, a), np.linspace(1, 0, r)])

    def synthesize_voicing(voicing, sample_rate=44100):
        duration = 1.0
        delay = 0.12
        total = duration + delay * 5
        samples = int(total * sample_rate)
        audio = np.zeros(samples)
        for i, (fret, _) in enumerate(voicing):
            if fret == "x": continue
            freq = open_frequencies[i] * (2 ** (fret / 12))
            offset = int(delay * i * sample_rate)
            t = np.linspace(0, duration, int(sample_rate * duration), False)
            env = envelope(duration, sample_rate, 60, 500)
            wave = 0.2 * np.sin(2 * np.pi * freq * t) * env
            audio[offset:offset+len(wave)] += wave[:samples-offset]
        audio = bandpass_filter(audio, sample_rate, 80, 5000)
        audio /= np.max(np.abs(audio)) if np.max(np.abs(audio)) > 0 else 1
        return audio

    voicings = generate_voicings(chord_root, chord_type)
    st.write(f"Number of Voicings: {len(voicings)}")
    st.write(f"Chord: {chord_root}{chord_type}")
    if not voicings:
        st.warning("No matching voicings were found.")
    elif mode == "Show best voicing only":
        v, *_ = voicings[0]
        st.subheader("Best Voicing")
        st.text(print_voicing(v))
        draw_fretboard_diagram(v, chord_root + chord_type)
        st.audio(synthesize_voicing(v), sample_rate=44100)
    else:
        for idx, (v, *_rest) in enumerate(voicings, 1):
            st.subheader(f"Voicing {idx}")
            st.text(print_voicing(v))
            draw_fretboard_diagram(v, chord_root + chord_type)
            st.audio(synthesize_voicing(v), sample_rate=44100)
