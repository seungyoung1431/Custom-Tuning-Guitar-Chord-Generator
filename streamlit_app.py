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
    "add#4": [0, 4, 6, 7],
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
    "7,9,#11": [0, 4, 7, 10, 14, 18], 
    "7,#9,#11": [0, 4, 7, 10, 15, 18],
    "7,b13": [0, 4, 7, 10, 20],
    "7,b9,b13": [0, 4, 7, 10, 13, 20],
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
    "add#4": [0, 4, 6, 7],
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
    "7,9,#11": [0, 4, 7, 10, 14, 18], 
    "7,#9,#11": [0, 4, 7, 10, 15, 18],
    "7,b13": [0, 4, 7, 10, 20],
    "7,b9,b13": [0, 4, 7, 10, 13, 20],
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
    "add#4": [0, 6, 7],
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
    "7,9,#11": [0, 10, 14, 18],
    "7,#9,#11": [0, 10, 15, 18],
    "7,b13": [0, 4, 10, 20],
    "7,b9,b13": [0, 10, 13, 20],
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
        if len(played) < 4:
            continue
        produced = [c[1] for c in played]
        # 필수 가이드 톤이 모두 포함되어야 함.
        if not required_tones.issubset(set(produced)):
            continue
        # 손가락으로 누른 프렛들만 고려
        frets = [c[0] for c in played if isinstance(c[0], int) and c[0] != 0]
        # 오픈 스트링은 프렛 0으로 간주
        if frets:
            span = max(frets) - min(frets)
        else:
            span = 0
        if span > 3:
            continue
        lower_indices = [i for i in [0,1,2] if comb[i][0] != "x"]
        if not lower_indices:
            continue
        # 가장 낮은 fretted 음(또는 오픈)이 chord root여야 함.
        if comb[min(lower_indices)][1] != chord_root:
            continue
        # 근음(루트)의 등장 횟수가 정확히 1번이어야 함.
        if produced.count(chord_root) != 1:
            continue
        frets_sum = sum(f for f, _ in played if isinstance(f, int))
        valid_voicings.append((comb, span, min(lower_indices), frets_sum))
    valid_voicings.sort(key=lambda x: (x[2], x[1], x[3]))
    return valid_voicings

def print_voicing(voicing):
    result = []
    for i, opt in enumerate(voicing):
        string_label = f"{6-i} ({tuning[i]})"
        fret, note = opt
        result.append(f"{string_label}: {'x' if fret == 'x' else f'{fret} → {note}'}")
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

# --- 추가: 기타 코드 다이어그램 (전통적인 기타 코드표 형태) ---
def draw_chord_diagram(voicing, chord_root, tuning):
    """
    전통적인 기타 코드표 형태로 출력합니다.
    - 좌측(왼쪽)에는 Nut (오픈/프렛 기준)가 있으며, 만약 최소 fretted fret가 0 또는 1이면 Nut를 두꺼운 선으로 그립니다.
    - 최소 fretted fret이 2 이상이면 좌측에 시작 프렛 번호를 표시합니다.
    - 6행(위:6번 스트링, 아래:1번 스트링)과 n열(여기서는 5 frets)을 사용합니다.
    - 오픈 스트링은 Nut 왼쪽에 'O', 뮤트는 'X'로 표시합니다.
    - 손가락 누른 fretted note는 해당 셀 중앙에 원을 그림. 단, chord_root(근음)는 단 한 개만 검은색(채움)으로 표시합니다.
    """
    n_cols = 5  # 다이어그램에 표시할 프렛 수
    n_rows = 6  # 6줄 (6번 ~ 1번)
    
    # fretted(손가락)인 프렛들만 따져 최소 프렛 결정 (0은 오픈취급)
    finger_frets = [fret for fret, note in voicing if isinstance(fret, int) and fret > 0]
    if finger_frets:
        min_fret = min(finger_frets)
    else:
        min_fret = 1
    # Nut 조건: 최소 fretted fret이 0 또는 1이면 Nut를 표시
    if min_fret <= 1:
        offset = 1
        show_nut = True
    else:
        offset = min_fret
        show_nut = False

    # 각 fretted note가 들어갈 열: 열 번호 = fret - offset + 1, 1-based cell (왼쪽 가장자리: nut 혹은 빈 여백)
    # 각 스트링(행): 0번행: 6번 스트링, ... 5번행: 1번 스트링
    # dot center 좌표: (x_center, y_center) = ( (fret - offset + 1) - 0.5, row + 0.5 )
    
    # 우선, designated root (한 번만 검출)
    designated_root = None
    for idx, (fret, note) in enumerate(voicing):
        if fret != "x" and fret != 0 and note == chord_root:
            designated_root = idx
            break

    fig, ax = plt.subplots(figsize=(4, 6))
    
    # 배경: 수직(프렛) 선과 수평(스트링) 선 그리기
    # 수직선: x= 0 ~ n_cols (0이면 Nut 또는 여백)
    for x in range(n_cols + 1):
        if x == 0 and show_nut:
            ax.plot([x, x], [0, n_rows], color='black', linewidth=3)  # Nut: 굵은 선
        else:
            ax.plot([x, x], [0, n_rows], color='black', linewidth=1)
            
    # 수평선: 각 스트링의 경계: y= 0 ~ n_rows
    for y in range(n_rows + 1):
        ax.plot([0, n_cols], [y, y], color='black', linewidth=1)
    
    # 각 스트링 왼쪽에 오픈/뮤트 마커 표시 (Nut 왼쪽)
    for row, (fret, note) in enumerate(voicing):
        y_center = row + 0.5
        if fret == "x":
            ax.text(-0.3, y_center, "X", ha='center', va='center', fontsize=12)
        elif fret == 0:
            ax.text(-0.3, y_center, "O", ha='center', va='center', fontsize=12)
    
    # 손가락으로 누른 fretted note 표시 (동그라미)  
    # 좌표: x_center = (fret - offset + 1) - 0.5, y_center = row + 0.5 
    for row, (fret, note) in enumerate(voicing):
        if isinstance(fret, int) and fret > 0:
            col = fret - offset + 1
            x_center = col - 0.5
            y_center = row + 0.5
            # designated root만 검은 원으로 표시, 나머지는 흰색 원 (테두리 검정)
            if note == chord_root and row == designated_root:
                circle = plt.Circle((x_center, y_center), 0.3, color='black', zorder=10)
            else:
                circle = plt.Circle((x_center, y_center), 0.3, facecolor='white', edgecolor='black', zorder=10)
            ax.add_patch(circle)
    
    # 만약 Nut가 없는 경우(즉, offset>=2) 왼쪽에 시작 프렛 번호 표시
    if not show_nut:
        ax.text(-0.7, n_rows - 0.5, str(offset), ha='center', va='center', fontsize=12)
    
    ax.set_xlim(-1, n_cols)
    ax.set_ylim(0, n_rows)
    # invert y 축 if needed so that 6번 스트링가 위쪽
    ax.invert_yaxis()
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
        fig = draw_chord_diagram(best_voicing, chord_root, tuning)
        st.pyplot(fig)
        st.audio(synthesize_voicing(best_voicing), sample_rate=44100)
    else:
        for idx, (v, *_rest) in enumerate(voicings, 1):
            st.subheader(f"Voicing {idx}")
            st.text(print_voicing(v))
            fig = draw_chord_diagram(v, chord_root, tuning)
            st.pyplot(fig)
            st.audio(synthesize_voicing(v), sample_rate=44100)
