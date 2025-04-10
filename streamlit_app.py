# streamlit_app.py
import streamlit as st
import numpy as np
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
st.title("Custom Tuning Guitar Chord Generator")
tuning = []
open_frequencies = []
tuning_columns = st.columns(6)
def_octaves = [2, 2, 3, 3, 3, 4]  # E2 A2 D3 G3 B3 E4
default_notes = ['E', 'A', 'D', 'G', 'B', 'E']
for i in range(6):
    note = tuning_columns[i].selectbox(
        f"{6 - i} string", 
        note_sequence, 
        index=note_sequence.index(default_notes[i])
    )
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
    "7,13": [0, 4, 7, 10, 21],
    "7,b9,13": [0, 10, 13, 21],
    "7,9,13": [0, 10, 14, 21],
    "7,#9,13": [0, 10, 15, 21],
    "7,#11,13": [0, 10, 18, 21],
    "7,9,#11,13": [0, 10, 14, 18, 21],
    "7,b9,#11,13": [0, 10, 13, 18, 21],
    "7,#9,#11,13": [0, 10, 15, 18, 21],
    "sus4,7": [0, 5, 7, 10],
    "sus4,7,b9": [0, 5, 7, 10, 13],
    "sus4,7,b9,13": [0, 5, 7, 10, 13, 21],
    "sus4,7,b9,b13": [0, 5, 7, 10, 13, 20],
    "sus4,7,9": [0, 5, 7, 10, 14],
    "sus4,7,9,13": [0, 5, 7, 10, 14, 21],
    "sus4,7,9,b13": [0, 5, 7, 10, 14, 20],
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
        opts = [
            (fret, note_from_fret(open_note, fret)) 
            for fret in range(max_fret+1)
            if note_from_fret(open_note, fret) in full_chord
        ]
        if allow_muted and idx in allowed_mute_indices:
            opts.append(("x", None))
        possible_options.append(opts)
    valid_voicings = []
    for comb in product(*possible_options):
        played = [c for c in comb if c[0] != "x"]
        if len(played) < 4:
            continue
        produced = {c[1] for c in played}
        # 필수(가이드) 톤 체크
        if not required_tones.issubset(produced):
            continue
        # 프렛 스팬 제한 (편의상 3프렛 이내로 제한)
        frets = [c[0] for c in played if isinstance(c[0], int)]
        span = max(frets) - min(frets)
        if span > 3:
            continue
        # 가장 낮은 3줄(6,5,4번) 중 실제로 소리나는 줄(뮤트, x 제외) 중 최저음이 루트여야 한다
        lower_indices = [i for i in [0,1,2] if comb[i][0] != "x"]
        if not lower_indices:
            continue
        if comb[min(lower_indices)][1] != chord_root:
            continue
        # 보이싱 정렬 기준: (가장낮은줄 인덱스, 프렛 스팬, frets 합)
        frets_sum = sum(f for f, _ in played if isinstance(f, int))
        valid_voicings.append((comb, span, min(lower_indices), frets_sum))
    valid_voicings.sort(key=lambda x: (x[2], x[1], x[3]))
    return valid_voicings

# 변경: 간단히 1 x 6 형태로 한 줄에 프렛(또는 x)만 표시
def print_voicing_simple(voicing):
    # voicing은 6개의 (fret, note) 튜플 (위에서부터 6->5->4->3->2->1)
    # 예: [(10, 'D'), ('x', None), (10, 'C'), ...]
    # -> "10 x 10 10 10 x" 형태 출력
    result = []
    for fret, _ in voicing:
        if fret == 'x':
            result.append('x')
        else:
            result.append(str(fret))
    return " ".join(result)

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
        if fret == "x":
            continue
        freq = open_frequencies[i] * (2 ** (fret / 12))
        offset = int(delay * i * sample_rate)
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        env = envelope(duration, sample_rate, 60, 500)
        wave = 0.2 * np.sin(2 * np.pi * freq * t) * env
        audio[offset:offset+len(wave)] += wave[:samples-offset]
    audio = bandpass_filter(audio, sample_rate, 80, 5000)
    audio /= np.max(np.abs(audio)) if np.max(np.abs(audio)) > 0 else 1
    return audio

# --- 실행 ---
if st.button("Generate Voicings"):
    voicings = generate_voicings(chord_root, chord_type)
    st.write(f"Number of Voicings: {len(voicings)}")
    st.write(f"Chord: {chord_root}{chord_type}")
    if not voicings:
        st.warning("No matching voicings were found.")
    elif mode == "Show best voicing only":
        v, *_ = voicings[0]
        st.subheader("Best Voicing")
        # 간단히 1 x 6 형태로 출력
        st.text(print_voicing_simple(v))
        # 오디오
        st.audio(synthesize_voicing(v), sample_rate=44100)
    else:
        for idx, (v, *_rest) in enumerate(voicings, 1):
            st.subheader(f"Voicing {idx}")
            # 간단히 1 x 6 형태로 출력
            st.text(print_voicing_simple(v))
            # 오디오
            st.audio(synthesize_voicing(v), sample_rate=44100)
