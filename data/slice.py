import glob
import os
import pickle

import librosa as lr
import numpy as np
import soundfile as sf
from tqdm import tqdm

#zxy_edit新增加
def delete_extra_motion(motion_file, out_dir, num_motion_slices, num_audio_slices):

    file_name = os.path.splitext(os.path.basename(motion_file))[0]

    # slice until done or until matching audio slices
    while num_motion_slices - num_audio_slices >= 0:
        file_path = f"{out_dir}/{file_name}_slice{num_motion_slices}.pkl"
        if os.path.exists(file_path):
            os.remove(file_path)

        num_motion_slices = num_motion_slices -1


def slice_audio(audio_file, stride, length, out_dir, num_slices):
    # stride, length in seconds
    audio, sr = lr.load(audio_file, sr=None)
    file_name = os.path.splitext(os.path.basename(audio_file))[0]
    start_idx = 0
    idx = 0
    window = int(length * sr)
    stride_step = int(stride * sr)
    while start_idx <= len(audio) - window and idx < num_slices:
        audio_slice = audio[start_idx : start_idx + window]
        sf.write(f"{out_dir}/{file_name}_slice{idx}.wav", audio_slice, sr)
        start_idx += stride_step
        idx += 1
    return idx


def slice_motion(motion_file, stride, length, out_dir):
    motion = pickle.load(open(motion_file, "rb"))
    pos, q, joint_offset = motion["pos"], motion["q"][:, :72], motion["joint_offset"]

    beta = motion["beta"]

    n = pos.shape[0]

    beta = np.repeat(beta, n).reshape((n, 16))

    file_name = os.path.splitext(os.path.basename(motion_file))[0]
    # normalize root position
    # pos /= scale
    start_idx = 0
    window = int(length * 60)
    stride_step = int(stride * 60)
    slice_count = 0
    # slice until done or until matching audio slices
    while start_idx <= len(pos) - window * 2:
        pos_slice, q_slice, beta_slice, joint_offset_slice = (
            pos[start_idx : start_idx + window * 2 : 2],
            q[start_idx : start_idx + window * 2 : 2],
            beta[start_idx : start_idx + window * 2 : 2],
            joint_offset[start_idx : start_idx + window * 2 : 2]
        )
        out = {"pos": pos_slice, "q": q_slice, "beta": beta_slice, "joint_offset": joint_offset_slice}
        start_idx += stride_step
        slice_count += 1
    return slice_count


def slice_aistpp(motion_dir, wav_dir, stride=0.5, length=5):
    wavs = sorted(glob.glob(f"{wav_dir}/*.wav"))
    motions = sorted(glob.glob(f"{motion_dir}/*.pkl"))
    wav_out = wav_dir + "_sliced"
    motion_out = motion_dir + "_sliced"
    os.makedirs(wav_out, exist_ok=True)
    os.makedirs(motion_out, exist_ok=True)
    assert len(wavs) == len(motions)
    for wav, motion in tqdm(zip(wavs, motions)):
        # make sure name is matching
        m_name = os.path.splitext(os.path.basename(motion))[0]
        w_name = os.path.splitext(os.path.basename(wav))[0]
        assert m_name == w_name, str((motion, wav))
        # audio_slices = slice_audio(wav, stride, length, wav_out)
        # motion_slices = slice_motion(motion, stride, length, audio_slices, motion_out)
        
        motion_slices = slice_motion(motion, stride, length, motion_out)
        audio_slices = slice_audio(wav, stride, length, wav_out, motion_slices)
        
        if audio_slices < motion_slices:
            delete_extra_motion(motion, motion_out, motion_slices, audio_slices)
            motion_slices = audio_slices
        # make sure the slices line up
        assert audio_slices == motion_slices, str(
            (wav, motion, audio_slices, motion_slices)
        )


def slice_audio_folder(wav_dir, stride=0.5, length=5):
    wavs = sorted(glob.glob(f"{wav_dir}/*.wav"))
    wav_out = wav_dir + "_sliced"
    os.makedirs(wav_out, exist_ok=True)
    for wav in tqdm(wavs):
        audio_slices = slice_audio(wav, stride, length, wav_out)
