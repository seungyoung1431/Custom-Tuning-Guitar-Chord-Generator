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
chord_type = st.selectbox("Chord Type", ["maj","min","dim7","7","maj7"])
mode = st.radio("Display Mode", ["Show all voicings", "Show best voicing only"])

# ---- Button ----
if st.button("🎵 Generate Voicings"):
    # 간단한 코드 테이블 (직접)
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

    def get_full_chord_tones(root, ctype):
        intervals = chord_formulas[ctype]
        r_idx = note_to_index[root]
        return { note_sequence[(r_idx + iv) % 12] for iv in intervals}

    def get_guide_tones(root, ctype):
        intervals = guide_tone_intervals[ctype]
        r_idx = note_to_index[root]
        return { note_sequence[(r_idx + iv) % 12] for iv in intervals}

    from itertools import product

    def generate_voicings(root, ctype, max_fret=14, allow_muted=True):
        full_chord = get_full_chord_tones(root, ctype)
        required = get_guide_tones(root, ctype)

        possible_options = []
        for idx, open_note in enumerate(tuning):
            opts = []
            for fret in range(max_fret+1):
                n = note_from_fret(open_note, fret)
                if n in full_chord:
                    opts.append((fret, n))
            if allow_muted and idx in allowed_mute_indices:
                opts.append(("x", None))
            possible_options.append(opts)

        valid = []
        for comb in product(*possible_options):
            played = [c for c in comb if c[0] != "x"]
            if len(played) < 4:
                continue
            produced = {p[1] for p in played}
            if not required.issubset(produced):
                continue
            frets = [p[0] for p in played if isinstance(p[0], int)]
            if frets:
                span = max(frets) - min(frets)
            else:
                span = 0
            if span > 3:
                continue
            # 근음 조건
            lower_indices = [i for i in [0,1,2] if comb[i][0] != "x"]
            if not lower_indices:
                continue
            if comb[min(lower_indices)][1] != root:
                continue
            fret_sum = sum(x for x in frets)
            valid.append((comb, span, min(lower_indices), fret_sum))

        valid.sort(key=lambda x: (x[2], x[1], x[3]))
        return valid

    # ---------- 사운드 -----------
    def bandpass_filter(data, sr, low, high):
        from scipy.signal import butter, lfilter
        ny = 0.5 * sr
        b,a = butter(4, [low/ny, high/ny], btype='band')
        return lfilter(b,a,data)

    def envelope(duration, sr, atk, rel):
        samples = int(duration*sr)
        A, R = int(atk/1000*sr), int(rel/1000*sr)
        s = samples - A - R
        if s<0: s=0
        env = np.concatenate([
            np.linspace(0,1,A,endpoint=False),
            np.ones(s),
            np.linspace(1,0,R,endpoint=False)
        ])
        if len(env)<samples:
            env = np.pad(env,(0,samples-len(env)),'constant')
        return env

    def synthesize_voicing(voicing, sr=44100):
        dur = 1.0
        delay = 0.12
        total = dur + delay*5
        smp = int(total*sr)
        audio = np.zeros(smp)
        for i,(fr,nt) in enumerate(voicing):
            if fr=="x":continue
            freq = open_frequencies[i]*(2**(fr/12))
            off = int(delay*i*sr)
            t = np.linspace(0,dur,int(sr*dur),False)
            env = envelope(dur,sr,60,500)
            wave = 0.2*np.sin(2*np.pi*freq*t)*env
            endi = off+len(wave)
            if endi> smp:
                wave=wave[:smp-off]
            audio[off:off+len(wave)] += wave
        audio = bandpass_filter(audio, sr,80,5000)
        m = np.max(np.abs(audio))
        if m>0:
            audio/=m
        return audio

    def print_voicing(voicing):
        lines=[]
        for i,(fr,nt) in enumerate(voicing):
            label = f"{6-i}({tuning[i]})"
            if fr=="x":
                lines.append(f"{label}: x")
            else:
                lines.append(f"{label}: {fr}→{nt}")
        return "\n".join(lines)

    # --------- 다이어그램 (4칸 x 5칸) ----------
    def draw_4x5_diagram(voicing, chord_name):
        """
        - 표 크기: 가로 4칸, 세로 5칸 => 수직선 5개(x=0..4), 수평선 6개(y=0..5)
        - 가장 낮은 프렛 = minF.
        - 0번(왼쪽 굵게), 1..4번 가는 선
        - 점: (fret-minF +0.5, string +0.5)
          (string=0 => 6번줄 맨위, string=5=>1번줄 맨아래)
        - x=0..4, y=0..5 (칸은 4x5)
        - x=0 굵게
        """
        # voicing 중 x는 제외한 frets
        frets_only = [f for f,n in voicing if f!="x"]
        if not frets_only:
            # 전부 x인 경우 => 그냥 0을 minF라 치자
            minF=0
        else:
            minF = min(frets_only)
        # 그림
        fig,ax=plt.subplots(figsize=(4,5))
        ax.set_facecolor("white")
        ax.set_xlim(-0.1,4.1)
        ax.set_ylim(-0.1,5.1)
        ax.invert_yaxis()
        ax.axis("off")

        # 세로선 (0..4)
        for x in range(5):
            lw = 3 if x==0 else 1
            ax.plot([x,x],[0,5],color='black',lw=lw)
        # 가로선 (0..5)
        for y in range(6):
            ax.plot([0,4],[y,y],color='black',lw=1)

        # 각 음 점찍기
        for i,(fret,note) in enumerate(voicing):
            # i=0 => 6번줄= top => y=0
            string_y = i
            if fret=="x":
                # x표시는 왼쪽 바깥
                ax.text(-0.3,string_y+0.5,"X", fontsize=12,
                    ha='center',va='center',color='red', weight='bold')
            else:
                # 만약 fret가 minF+4 넘어가면 이 4칸 표로 표현불가 => 그냥 생략
                if fret> minF+3:
                    continue
                x_col = (fret - minF)+0.5
                y_row = string_y+0.5
                # 점
                ax.plot(x_col,y_row,"o",color='black',markersize=15)
                # 음표 안 글씨
                ax.text(x_col,y_row,note,fontsize=9,color='white',
                    ha='center',va='center')

        # 코드 이름: 오른쪽 위쪽
        ax.text(4.2,0.5,chord_name,fontsize=12, rotation=270,
            ha='left',va='bottom',color='black')

        st.pyplot(fig)

    # -------------------
    voics = generate_voicings(chord_root, chord_type)
    st.write(f"Number of Voicings: {len(voics)}")
    st.write(f"Chord: {chord_root}{chord_type}")

    if not voics:
        st.warning("No matching voicings were found.")
    elif mode=="Show best voicing only":
        best, *_=voics[0]
        st.subheader("Best Voicing")
        st.text(print_voicing(best))
        draw_4x5_diagram(best, chord_root+chord_type)
        st.audio(synthesize_voicing(best),sample_rate=44100)
    else:
        for idx,(v,*rest) in enumerate(voics,1):
            st.subheader(f"Voicing {idx}")
            st.text(print_voicing(v))
            draw_4x5_diagram(v, chord_root+chord_type)
            st.audio(synthesize_voicing(v),sample_rate=44100)
