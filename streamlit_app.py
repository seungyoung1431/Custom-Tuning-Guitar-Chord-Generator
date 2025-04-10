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
    note = tuning_columns[i].selectbox(
        f"{6 - i} string", note_sequence, index=note_sequence.index(default_notes[i])
    )
    tuning.append(note)
    open_frequencies.append(get_frequency(note, def_octaves[i]))

allowed_mute_indices = {0, 1, 5}  # 6,5,1번 줄만 음소거 가능

chord_root = st.selectbox("Chord Root", note_sequence)
chord_type = st.selectbox("Chord Type", list({
    "maj": [0, 4, 7],
    "min": [0, 3, 7],
    "dim7": [0, 3, 6, 9],
    "7": [0, 4, 7, 10],
    "maj7": [0, 4, 7, 11]
}.keys()))
mode = st.radio("Display Mode", ["Show all voicings", "Show best voicing only"])

# --- Generate Voicings Always Available ---
if st.button("🎵 Generate Voicings"):
    chord_formulas = {
        "maj": [0, 4, 7],
        "min": [0, 3, 7],
        "dim7": [0, 3, 6, 9],
        "7": [0, 4, 7, 10],
        "maj7": [0, 4, 7, 11]
    }
    guide_tone_intervals = {
        "maj": [0, 4],
        "min": [0, 3],
        "dim7": [0, 3, 6],
        "7": [0, 4, 10],
        "maj7": [0, 4, 11]
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
                for fret in range(max_fret + 1)
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
            if not required_tones.issubset(produced):
                continue
            frets = [c[0] for c in played if isinstance(c[0], int)]
            span = max(frets) - min(frets) if frets else 0
            if span > 3:
                continue
            lower_indices = [i for i in [0,1,2] if comb[i][0] != "x"]
            if not lower_indices:
                continue
            # 근음이 6~4번 줄 중 가장 낮은 인덱스 현에 있어야
            if comb[min(lower_indices)][1] != chord_root:
                continue
            frets_sum = sum(f for f, _ in played if isinstance(f, int))
            valid_voicings.append((comb, span, min(lower_indices), frets_sum))

        valid_voicings.sort(key=lambda x: (x[2], x[1], x[3]))
        return valid_voicings

    def bandpass_filter(data, rate, low, high):
        nyq = 0.5 * rate
        b, a = butter(4, [low / nyq, high / nyq], btype='band')
        return lfilter(b, a, data)

    def envelope(duration, rate, attack_ms, release_ms):
        samples = int(duration * rate)
        a, r = int(attack_ms / 1000 * rate), int(release_ms / 1000 * rate)
        s = samples - a - r
        if s < 0:
            s = 0
        env = np.concatenate([
            np.linspace(0, 1, a, endpoint=False),
            np.ones(s),
            np.linspace(1, 0, r, endpoint=False)
        ])
        if len(env) < samples:
            env = np.pad(env, (0, samples - len(env)), mode='constant')
        return env

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
            end_idx = offset + len(wave)
            if end_idx > samples:
                wave = wave[: samples - offset]
            audio[offset:offset + len(wave)] += wave

        audio = bandpass_filter(audio, sample_rate, 80, 5000)
        mx = np.max(np.abs(audio))
        if mx > 0:
            audio /= mx
        return audio

    def print_voicing(voicing):
        lines = []
        for i, (fret, note) in enumerate(voicing):
            string_label = f"{6 - i} ({tuning[i]})"
            if fret == "x":
                lines.append(f"{string_label}: x")
            else:
                lines.append(f"{string_label}: {fret} → {note}")
        return "\n".join(lines)

    # ******************** 핵심: 다이어그램 함수 ******************** #
    def draw_fretboard_diagram(voicing, chord_name):
        """
        - 맨 위가 6번줄, 맨 아래가 1번줄
        - Nut(왼쪽) 선 두껍게
        - x 표시: 해당 줄(가장 왼쪽부분)에
        - 음표(●)는 칸 정중앙
        - 음 이름은 점 내부 (white) or 우측
        - 0~5프렛만 표시 (가장 흔함)
        """
        fig_w, fig_h = 3, 5  # 세로로 좀 더 길게
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        ax.set_facecolor("white")

        # frets 범위: 0~5
        # strings: 6번 ~ 1번 (위->아래)
        # x축: frets (왼->오른), y축: strings (위->아래)
        # 그러나 matplotlib 좌표는 (왼->오른, 아래->위)이므로 뒤집는 테크닉 사용
        # 우린 깔끔하게: x=프렛, y=줄. 6번줄 = y=0, 1번줄 = y=5
        # Fret lines(수직) => x=0..5
        # String lines(수평) => y=0..6

        # 크기 설정
        ax.set_xlim(-0.2, 5.2)   # frets 0..5
        ax.set_ylim(-0.5, 6.5)   # strings 0..6 (total 6, 6번 ~ 1번)
        ax.invert_yaxis()        # 6번줄이 위, 1번줄이 아래

        # 그리드 없이
        ax.axis("off")

        # 프렛 선: 세로 (x=0..5)
        for fret_i in range(6):
            lw = 3 if fret_i == 0 else 1  # 왼쪽 nut(0번 프렛)은 굵게
            ax.plot([fret_i, fret_i], [0, 6], color="black", lw=lw)

        # 줄 선: 가로 (y=0..6)
        for string_i in range(7):
            ax.plot([0, 5], [string_i, string_i], color="black", lw=1)

        # 음 배치
        # 6번줄 => y=0, 5번줄 => y=1 ... 1번줄 => y=5
        # fret n => x= n
        # 점은 fret+0.5, string+0.5 위치에 둔다 (칸 중앙)
        for i, (fret, note) in enumerate(voicing):
            string_y = i  # i=0 => 6번줄 y=0
            if fret == "x":
                # x 표시는 (x= -0.3 위치) 등?
                ax.text(-0.3, string_y + 0.5, "x", fontsize=12,
                        ha="center", va="center", color="red", weight="bold")
            else:
                # (fret, string_y) => 중앙 = (fret+0.5, string_y+0.5)
                cx = fret + 0.5
                cy = string_y + 0.5
                # 점
                circ_size = 12
                ax.plot(cx, cy, "o", markersize=circ_size, color="black")
                # 음표 안 글씨
                ax.text(cx, cy, note,
                        fontsize=9, color="white",
                        ha="center", va="center", weight="bold")

        # 코드 이름은 상단 오른쪽에 세로로
        ax.text(5.3, 0.5, chord_name, rotation=270, fontsize=12,
                ha="left", va="bottom", color="black")

        st.pyplot(fig)

    # ************************************************************** #

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
