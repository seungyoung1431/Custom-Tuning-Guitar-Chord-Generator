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
st.title("🎸Custom Tuning Guitar Chord Generator")
st.caption("Create Chord Voicings for Any Tuning")

st.subheader("1. Set Your Tuning (6th to 1st string)")
tuning = []
open_frequencies = []
tuning_columns = st.columns(6)
def_octaves = [2, 2, 3, 3, 3, 4]  # E2 A2 D3 G3 B3 E4
default_notes = ['E', 'A', 'D', 'G', 'B', 'E']
for i in range(6):
    note = tuning_columns[i].selectbox(f"{6 - i} string", note_sequence, index=note_sequence.index(default_notes[i]))
    tuning.append(note)
    open_frequencies.append(get_frequency(note, def_octaves[i]))

allowed_mute_indices = {0, 1, 5}  # 6번, 5번, 1번 줄만 음소거 가능

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
    "sus4,7,9,b13": [0, 5, 10, 14, 20],
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

# --- 추가: 기타 코드 다이어그램 그리기 ---
def draw_chord_diagram(voicing, chord_root, tuning):
    """
    voicing: 각 줄(6~1번)에 대한 (fret, note) 튜플로 구성된 6요소 튜플.
    chord_root: 선택된 코드 루트 (예, "C")
    tuning: 각 줄의 오픈 음 (예, ['E','A','D','G','B','E'])
    
    - 뮤트("x")와 오픈(0)은 Nut 위에 "X" 또는 "O"로 표시
    - fretted된 경우, 해당 frets에 원을 그리며, 루트 음이면 검은색으로 채움
    - 최소 fretted fret이 0 또는 1이면 Nut(굵은 선)를 그리고, 2이상이면 다이어그램 왼쪽에 시작 프렛 번호를 표시
    """
    # 각 스트링의 표시 정보를 구성 (index 0: 6번, index 5: 1번)
    positions = []
    finger_frets = []
    for i, (fret, note) in enumerate(voicing):
        if fret == "x":
            positions.append({"type": "mute"})
        elif fret == 0:
            positions.append({"type": "open"})
        else:
            positions.append({"type": "finger", "fret": fret, "note": note})
            finger_frets.append(fret)
    
    if finger_frets:
        min_fret = min(finger_frets)
    else:
        min_fret = 1  # 오픈 코드인 경우

    # nut (오픈 코드의 경우) 여부 결정
    nut = (min_fret <= 1)
    n_rows = 5  # 다이어그램에 표시할 frets 수
    fig, ax = plt.subplots(figsize=(3, 4))
    
    # 세로줄 (스트링) 그리기: 왼쪽부터 6번 ~ 1번까지 (인덱스 0~5)
    for i in range(6):
        ax.plot([i, i], [0, n_rows], color='black', linewidth=1)
    
    # 가로줄 (frets) 그리기
    for j in range(n_rows + 1):
        # Nut인 경우, 제일 위의 줄을 굵게 표시
        if j == 0 and nut:
            ax.plot([-0.1, 5.1], [j, j], color='black', linewidth=3)
        else:
            ax.plot([-0.1, 5.1], [j, j], color='black', linewidth=1)
    
    # 각 스트링의 오픈/뮤트, 혹은 fretted된 마커 그리기
    # 오픈/뮤트 표시는 Nut 위쪽 (y = -0.3) 에 배치
    for i, pos in enumerate(positions):
        x = i  # 왼쪽부터 0~5
        if pos["type"] == "mute":
            ax.text(x, -0.3, "X", ha='center', va='center', fontsize=12)
        elif pos["type"] == "open":
            ax.text(x, -0.3, "O", ha='center', va='center', fontsize=12)
        elif pos["type"] == "finger":
            fret = pos["fret"]
            # fretted된 위치의 y 좌표 계산: 
            # - Nut가 그려지는 경우 (min_fret <=1): 첫 fret 셀의 중앙은 0.5, 즉 (fret - 1) + 0.5
            # - Nut가 없는 경우: 상단이 min_fret에 해당하므로 (fret - min_fret) + 0.5
            if nut:
                y = (fret - 1) + 0.5
            else:
                y = (fret - min_fret) + 0.5
            # 코드 루트인 경우 원을 채운 검은색으로, 아닐 경우 흰색 내부에 검은 테두리로 표시
            if pos["note"] == chord_root:
                circle = plt.Circle((x, y), 0.3, color='black', zorder=10)
            else:
                circle = plt.Circle((x, y), 0.3, facecolor='white', edgecolor='black', zorder=10)
            ax.add_patch(circle)
    
    # 만약 Nut를 그리지 않는 경우 (즉, 최소 fretted fret이 2 이상이면) 시작 프렛 번호를 다이어그램 왼쪽에 표시.
    if not nut:
        ax.text(-0.7, 0.5, str(min_fret), ha='center', va='center', fontsize=12)
    
    # 좌표 설정: 일반적으로 기타 다이어그램은 위쪽이 Nut이므로 y축을 반전
    ax.set_xlim(-1, 6)
    ax.set_ylim(n_rows, -1)
    ax.axis('off')
    
    return fig

# --- 실행 ---
if st.button("Generate Voicings"):
    voicings = generate_voicings(chord_root, chord_type)
    st.write(f"Number of Voicings: {len(voicings)}")
    st.write(f"Chord: {chord_root}{chord_type}")
    if not voicings:
        st.warning("No matching voicings were found.")
    elif mode == "Show best voicing only":
        best_voicing, *_ = voicings[0]
        st.subheader("Best Voicing")
        st.text(print_voicing(best_voicing))
        # 기타 다이어그램 출력
        fig = draw_chord_diagram(best_voicing, chord_root, tuning)
        st.pyplot(fig)
        st.audio(synthesize_voicing(best_voicing), sample_rate=44100)
    else:
        for idx, (v, *_rest) in enumerate(voicings, 1):
            st.subheader(f"Voicing {idx}")
            st.text(print_voicing(v))
            # 각 보이싱에 대해 기타 다이어그램 표시
            fig = draw_chord_diagram(v, chord_root, tuning)
            st.pyplot(fig)
            st.audio(synthesize_voicing(v), sample_rate=44100)
