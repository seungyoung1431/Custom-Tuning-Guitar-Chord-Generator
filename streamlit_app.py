# streamlit_app.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from scipy.signal import butter, lfilter

##################
#  기본 함수들   #
##################
note_sequence = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
note_to_index = {note: i for i, note in enumerate(note_sequence)}

def note_from_fret(open_note, fret):
    start = note_to_index[open_note]
    return note_sequence[(start + fret) % 12]

def get_frequency(note, octave=4):
    base_freq = 440.0  # A4
    semitone_distance = note_to_index[note] - note_to_index['A'] + (octave - 4) * 12
    return base_freq * (2 ** (semitone_distance / 12))

def bandpass_filter(data, rate, low, high):
    from scipy.signal import butter, lfilter
    nyq = 0.5 * rate
    b, a = butter(4, [low/nyq, high/nyq], btype='band')
    return lfilter(b, a, data)

def envelope(duration, rate, attack_ms, release_ms):
    samples = int(duration * rate)
    a, r = int(attack_ms / 1000 * rate), int(release_ms / 1000 * rate)
    s = samples - a - r
    if s < 0: s = 0
    env = np.concatenate([
        np.linspace(0, 1, a, endpoint=False),
        np.ones(s),
        np.linspace(1, 0, r, endpoint=False)
    ])
    if len(env) < samples:
        env = np.pad(env, (0, samples - len(env)), mode='constant')
    return env

def synthesize_voicing(voicing, sr=44100, open_frequencies=None):
    dur = 1.0
    dly = 0.12
    total = dur + dly*5
    smp = int(total*sr)
    audio = np.zeros(smp)
    for i,(fr,_) in enumerate(voicing):
        if fr=="x": 
            continue
        freq = open_frequencies[i]*(2**(fr/12))
        off = int(dly*i*sr)
        t = np.linspace(0,dur,int(sr*dur),False)
        env = envelope(dur,sr,60,500)
        wave = 0.2*np.sin(2*np.pi*freq*t)*env
        endi = off+len(wave)
        if endi> smp:
            wave=wave[:smp-off]
        audio[off:off+len(wave)] += wave
    audio = bandpass_filter(audio,sr,80,5000)
    mx = np.max(np.abs(audio))
    if mx>0:
        audio/=mx
    return audio

######################
#  Streamlit UI 부분  #
######################
st.title("🎸 Custom Tuning Guitar Chord Generator")
st.caption("Create chord voicings for any tuning")

# 1. Tuning 설정
st.subheader("1. Set Your Tuning (6th to 1st string)")
tuning = []
open_frequencies = []
tuning_columns = st.columns(6)
def_octaves = [2, 2, 3, 3, 3, 4]

preset = st.selectbox("🎼 Choose a tuning preset", ["Standard E (EADGBE)", "Drop D (DADGBE)", "Custom"])
if preset == "Standard E (EADGBE)":
    default_notes = ['E','A','D','G','B','E']
elif preset == "Drop D (DADGBE)":
    default_notes = ['D','A','D','G','B','E']
else:
    default_notes = ['E','A','D','G','B','E']

for i in range(6):
    note = tuning_columns[i].selectbox(
        f"{6 - i} string", note_sequence, 
        index=note_sequence.index(default_notes[i])
    )
    tuning.append(note)
    open_frequencies.append(get_frequency(note, def_octaves[i]))

allowed_mute_indices = {0,1,5}  # 6,5,1번 줄 음소거 가능

# 2. 코드 Root / Type
chord_root = st.selectbox("Chord Root", note_sequence)
chord_type = st.selectbox("Chord Type", ["maj","min","dim7","7","maj7"])
mode = st.radio("Display Mode", ["Show all voicings","Show best voicing only"])

#######################
#  Generate Button    #
#######################
if st.button("🎵 Generate Voicings"):
    # 아주 단순 코드 정의
    chord_formulas = {
        "maj": [0,4,7],
        "min": [0,3,7],
        "dim7":[0,3,6,9],
        "7":[0,4,7,10],
        "maj7":[0,4,7,11]
    }
    guide_tone_intervals = {
        "maj":[0,4],
        "min":[0,3],
        "dim7":[0,3,6],
        "7":[0,4,10],
        "maj7":[0,4,11]
    }

    def get_full_chord_tones(croot, ctype):
        intervals = chord_formulas[ctype]
        rdx = note_to_index[croot]
        return { note_sequence[(rdx + iv) % 12] for iv in intervals }

    def get_guide_tones(croot, ctype):
        intervals = guide_tone_intervals[ctype]
        rdx = note_to_index[croot]
        return { note_sequence[(rdx + iv) % 12] for iv in intervals }

    def generate_voicings(croot, ctype, max_fret=14, allow_muted=True):
        full_chord = get_full_chord_tones(croot, ctype)
        required = get_guide_tones(croot, ctype)

        possible_opts = []
        for idx, open_note in enumerate(tuning):
            opts = []
            for f in range(max_fret+1):
                n = note_from_fret(open_note, f)
                if n in full_chord:
                    opts.append((f,n))
            if allow_muted and idx in allowed_mute_indices:
                opts.append(("x",None))
            possible_opts.append(opts)

        valids = []
        for comb in product(*possible_opts):
            played = [c for c in comb if c[0]!="x"]
            if len(played)<4: 
                continue
            produced = {p[1] for p in played}
            if not required.issubset(produced):
                continue
            frets = [p[0] for p in played if isinstance(p[0],int)]
            if frets:
                span = max(frets)-min(frets)
            else:
                span=0
            if span>3: 
                continue
            # 근음 조건
            lower_idx = [i for i in [0,1,2] if comb[i][0]!="x"]
            if not lower_idx: 
                continue
            if comb[min(lower_idx)][1] != croot:
                continue
            fret_sum = sum(x for x in frets)
            valids.append((comb, span, min(lower_idx), fret_sum))
        valids.sort(key=lambda x:(x[2],x[1],x[3]))
        return valids

    def print_voicing(voicing):
        lines=[]
        for i,(fr,nt) in enumerate(voicing):
            label = f"{6-i}({tuning[i]})"
            if fr=="x":
                lines.append(f"{label}: x")
            else:
                lines.append(f"{label}: {fr}→{nt}")
        return "\n".join(lines)

    ###############################
    # 다이어그램(4칸x5칸), 2:1비율
    # 음표 크기 2배, X도 2배
    # 음표와 X는 가로선 자체에 중앙 배치
    # 전체 크기는 1/3
    # 코드명 표시 제거
    ###############################
    def draw_4x5_diagram(voicing):
        # 기존 대비 1/3: ex. (8,4)->(8/3,4/3)->(2.67,1.33)
        fig, ax = plt.subplots(figsize=(2.67,1.33))
        ax.set_facecolor("white")
        # 수직선 x=0..4, 수평선 y=0..5
        ax.set_xlim(-0.1,4.1)
        ax.set_ylim(-0.1,5.1)
        # 뒤집지 않음(상단=6번줄), 옆으로 길게
        ax.axis("off")

        # 세로줄
        for x in range(5):
            lw=3 if x==0 else 1
            ax.plot([x,x],[0,5],color='black',lw=lw)
        # 가로줄
        for y in range(6):
            ax.plot([0,4],[y,y],color='black',lw=1)

        frets_only = [f for f,n in voicing if f!="x"]
        minF = min(frets_only) if frets_only else 0

        # 음표(●)와 X 표시
        # "가로선에 걸친 중앙" => y를 정수로 해서 점을 띄우지 않음
        # i=0 -> y=0 (6번줄), i=5->y=5(1번줄)
        # x_col = (f-minF)+0.5
        # markersize= "2배" => ex 15->30
        # X도 font 크기 2배
        for i,(f,n) in enumerate(voicing):
            y = i  # 정수
            if f=="x":
                ax.text(-0.3,y,"X",fontsize=20,
                    ha='center',va='center',color='red',weight='bold')
            else:
                if f> minF+3:
                    continue
                cx = (f-minF)+0.5
                # 점: y정수 + 0 => y자리에
                ax.plot(cx,y,"o",markersize=30,color='black')
                ax.text(cx,y,n,fontsize=12,color='white',
                    ha='center',va='center',weight='bold')

        st.pyplot(fig)

    # --- 보이싱 생성 & 출력 ---
    from itertools import product

    voics = generate_voicings(chord_root,chord_type)
    st.write(f"Number of Voicings: {len(voics)}")

    if not voics:
        st.warning("No matching voicings were found.")
    elif mode=="Show best voicing only":
        best,*_ = voics[0]
        st.subheader("Best Voicing")
        st.text(print_voicing(best))
        draw_4x5_diagram(best)
        audio = synthesize_voicing(best,open_frequencies=open_frequencies)
        st.audio(audio, sample_rate=44100)
    else:
        for idx,(v,*_) in enumerate(voics,1):
            st.subheader(f"Voicing {idx}")
            st.text(print_voicing(v))
            draw_4x5_diagram(v)
            audio = synthesize_voicing(v,open_frequencies=open_frequencies)
            st.audio(audio, sample_rate=44100)
