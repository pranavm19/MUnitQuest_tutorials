"""
Generate synthetic source data for the MUnitQuest EMG-BIDS tutorial.

Simulates 10 HD-sEMG recordings across 2 subjects and multiple tasks using a
trapezoidal contraction protocol (rest → ramp up → plateau → ramp down → rest).
For each recording it writes:
  .npy  — raw EMG array, shape (n_channels, n_samples), float16, µV
  .npz  — spike trains, keys MU_00…MU_NN, each an int32 array of sample indices
  .csv  — behavioural event annotations (onset, duration, mvc_level, event_type)

Also regenerates recordings.csv (linking each recording to its three data files)
into source_data/. The other CSV files (channels_electrodes, participants, etc.)
are edited manually and are not overwritten by this script.

Run this script first, then run assemble_bids.py to build the BIDS dataset.

Usage:
    python generate_source_data.py

Requires: numpy
"""

import os
import csv
import numpy as np

# Constants 
FS        = 2048           # Hz
DURATION  = 40             # seconds
N_SAMPLES = FS * DURATION  # 81920

# Trapezoidal segments (samples): rest | ramp_up | plateau | ramp_down | rest
SEG = [5*FS, 5*FS, 20*FS, 5*FS, 5*FS]

OUT_DIR = os.path.join(os.path.dirname(__file__), 'source_data')

SETUPS = {
    'VL_3x4s_1i':  (13, 5),  # 12 surface + 1 intramuscular EMG, 20 MUs
    'TA_dual_3x3': (18, 10),  # 9+9 surface EMG, 30 MUs
    'TA_4x4':      (16, 10),  # 16 surface EMG, 30 MUs
}

TASK_MVC = {
    'rest':                 0,
    'isometric30percentMVC': 30,
    'isometric50percentMVC': 50,
}

RECORDINGS = [
    # sub   ses    task                        run  setup          path_stem
    ('01', '',   'rest',                  '1', 'VL_3x4s_1i',  'sub1/vl_trial1'),
    ('01', '',   'isometric30percentMVC', '1', 'VL_3x4s_1i',  'sub1/vl_trial2'),
    ('01', '',   'isometric50percentMVC', '1', 'VL_3x4s_1i',  'sub1/vl_trial3'),
    ('01', '',   'isometric50percentMVC', '2', 'VL_3x4s_1i',  'sub1/vl_trial4'),
    ('02', '01', 'rest',                  '1', 'TA_dual_3x3', 'sub2/ta_2grids_trial1'),
    ('02', '01', 'isometric30percentMVC', '1', 'TA_dual_3x3', 'sub2/ta_2grids_trial2'),
    ('02', '01', 'isometric30percentMVC', '2', 'TA_dual_3x3', 'sub2/ta_2grids_trial3'),
    ('02', '02', 'isometric30percentMVC', '1', 'TA_4x4',      'sub2/ta_1grid_trial1'),
    ('02', '02', 'isometric30percentMVC', '2', 'TA_4x4',      'sub2/ta_1grid_trial2'),
    ('02', '02', 'isometric30percentMVC', '3', 'TA_4x4',      'sub2/ta_1grid_trial3'),
]


def make_events(pct_mvc: int, fs: int) -> list:
    """Return event rows for one trapezoidal contraction recording."""
    seg = [5*fs, 5*fs, 20*fs, 5*fs, 5*fs]

    if pct_mvc == 0:
        return [{"onset": 0.0, "duration": float(DURATION), "sample": 0,
                 "mvc_level": 0.0, "event_type": "rest",
                 "description": "Resting EMG recording"}]

    rate = pct_mvc / 5  # % MVC/s (5-second ramp)
    s_on   = seg[0]
    s_hold = seg[0] + seg[1]
    s_down = seg[0] + seg[1] + seg[2]
    s_off  = seg[0] + seg[1] + seg[2] + seg[3]
    return [
        {"onset": s_on/fs,   "duration": 0.0,         "sample": s_on,
         "mvc_level": 0.0,   "event_type": "muscle_on",
         "description": "Start of voluntary contraction"},
        {"onset": s_on/fs,   "duration": seg[1]/fs,    "sample": s_on,
         "mvc_level": 0.0,   "event_type": "linear_ramp",
         "description": f"Ramp up to {pct_mvc}% MVC at {rate:.0f}% MVC/s"},
        {"onset": s_hold/fs, "duration": seg[2]/fs,    "sample": s_hold,
         "mvc_level": float(pct_mvc), "event_type": "steady_hold",
         "description": f"Steady isometric contraction at {pct_mvc}% MVC"},
        {"onset": s_down/fs, "duration": seg[3]/fs,    "sample": s_down,
         "mvc_level": float(pct_mvc), "event_type": "linear_ramp",
         "description": f"Ramp down from {pct_mvc}% MVC at {rate:.0f}% MVC/s"},
        {"onset": s_off/fs,  "duration": 0.0,          "sample": s_off,
         "mvc_level": 0.0,   "event_type": "muscle_off",
         "description": "End of voluntary contraction"},
    ]


def write_events_csv(path: str, events: list) -> None:
    fields = ["onset", "duration", "sample", "mvc_level", "event_type", "description"]
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(events)


def make_envelope(pct_mvc: int) -> np.ndarray:
    """Trapezoidal force envelope, peak = pct_mvc / 100."""
    amp = pct_mvc / 100.0
    return np.concatenate([
        np.zeros(SEG[0]),
        np.linspace(0, amp, SEG[1]),
        np.full(SEG[2], amp),
        np.linspace(amp, 0, SEG[3]),
        np.zeros(SEG[4]),
    ])


def make_spike_trains(envelope: np.ndarray, n_units: int, rng: np.random.RandomState) -> dict:
    """
    Poisson-like spike trains with firing rate proportional to force above
    each unit's recruitment threshold.

    Returns dict  MU_XX -> int32 array of spike sample indices.
    """
    n_samples  = len(envelope)
    max_force  = envelope.max() if envelope.max() > 0 else 1.0
    # Recruitment thresholds spread over 0..70% of peak force
    thresholds = np.linspace(0.0, 0.70 * max_force, n_units)

    BASE_RATE = 1.0   # Hz at threshold
    MAX_RATE  = 2.0  # Hz at max force
    REFRACTORY = int(0.050 * FS)  # 20 ms absolute refractory

    spike_trains = {}
    for u in range(n_units):
        thr    = thresholds[u]
        spikes = []
        t      = rng.randint(0, max(1, int(0.05 * FS)))  # small random offset
        while t < n_samples:
            excess = max(0.0, float(envelope[t]) - thr)
            if excess > 0:
                rate = BASE_RATE + (MAX_RATE - BASE_RATE) * excess / max(max_force - thr, 1e-9)
                isi  = max(REFRACTORY, int(rng.exponential(FS / rate)))
                t   += isi
                if t < n_samples:
                    spikes.append(t)
            else:
                t += int(0.010 * FS)  # 10 ms skip when silent
        spike_trains[f'MU_{u:02d}'] = np.array(spikes, dtype=np.int32)
    return spike_trains


def _muap_template(length: int = 80) -> np.ndarray:
    """Biphasic MUAP waveform, peak ≈ 1 (will be scaled per unit/channel)."""
    t = np.linspace(0, length / FS * 1000, length)  # ms
    wave = -np.sin(2 * np.pi * t / 1.8) * np.exp(-t / 1.8)
    return wave / np.abs(wave).max()


MUAP = _muap_template(80)


def render_emg(spike_trains: dict, n_emg: int, envelope: np.ndarray,
               rng: np.random.RandomState) -> np.ndarray:
    """
    Render surface EMG (µV) by superimposing scaled MUAP waveforms.

    Each MU has a random amplitude vector across channels (rough proxy for
    spatial variation on the grid) and a random overall scale.
    """
    n_units  = len(spike_trains)
    muap_len = len(MUAP)
    emg      = np.zeros((n_emg, N_SAMPLES), dtype=np.float64)

    # Per-unit amplitude profiles across channels
    scales       = rng.uniform(80, 350, size=n_units)            # µV peak
    chan_profiles = rng.randn(n_units, n_emg) * 0.35 + 1.0       # channel variation
    chan_profiles = np.clip(chan_profiles, 0.1, 2.0)

    for u, (mu_name, spikes) in enumerate(spike_trains.items()):
        amp_vec = scales[u] * chan_profiles[u]   # (n_emg,)
        for s in spikes:
            end  = min(s + muap_len, N_SAMPLES)
            tlen = end - s
            emg[:, s:end] += np.outer(amp_vec, MUAP[:tlen])

    # Baseline noise + contraction-scaled noise
    noise_base = rng.randn(n_emg, N_SAMPLES) * 25.0
    noise_cont = rng.randn(n_emg, N_SAMPLES) * 60.0 * envelope[np.newaxis, :]
    emg += noise_base + noise_cont

    return emg


def write_recordings_csv(path: str) -> None:
    rows = [['sub', 'ses', 'task_name', 'run', 'setup',
             'path_to_emg_file', 'path_to_labels_file', 'path_to_events_file']]
    for sub, ses, task, run, setup, stem in RECORDINGS:
        rows.append([sub, ses, task, run, setup,
                     stem + '.npy', stem + '_spike_trains.npz', stem + '_events.csv'])
    with open(path, 'w', newline='') as f:
        csv.writer(f).writerows(rows)

def main():
    rng = np.random.RandomState(42)

    for sub, ses, task, run, setup, stem in RECORDINGS:
        pct_mvc    = TASK_MVC[task]
        n_emg, n_mu = SETUPS[setup]

        envelope    = make_envelope(pct_mvc)
        spike_trains = make_spike_trains(envelope, n_mu, rng)
        emg_data    = render_emg(spike_trains, n_emg, envelope, rng)

        # Torque channel (Nm), scaled to ~1.2 × pct_mvc + noise
        torque = envelope * (pct_mvc * 1.2 + 1.0) + rng.randn(N_SAMPLES) * 0.4

        # Stack: (n_emg + 1, n_samples) — save as fp16 to save disk space
        full = np.vstack([emg_data, torque[np.newaxis, :]]).astype(np.float16)

        folder = os.path.join(OUT_DIR, os.path.dirname(stem))
        os.makedirs(folder, exist_ok=True)

        emg_path    = os.path.join(OUT_DIR, stem + '.npy')
        labels_path = os.path.join(OUT_DIR, stem + '_spike_trains.npz')

        np.save(emg_path, full)
        np.savez_compressed(labels_path, **spike_trains)

        events_path = os.path.join(OUT_DIR, stem + '_events.csv')
        write_events_csv(events_path, make_events(pct_mvc, FS))

        total_spikes = sum(len(v) for v in spike_trains.values())
        print(f"  {stem:35s}  shape={str(full.shape):18s}  {n_mu} MUs  {total_spikes} spikes")

    # Write CSVs
    write_recordings_csv(os.path.join(OUT_DIR, 'recordings.csv'))
    print(f"\nDone. Source data written to: {OUT_DIR}/")


if __name__ == '__main__':
    print("Generating synthetic source data...\n")
    main()
