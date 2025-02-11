import glob
import os
from functools import cmp_to_key
from pathlib import Path
from tempfile import TemporaryDirectory
import random
import math

import clip
import numpy as np
import jittor as jt
from tqdm import tqdm

from args import parse_test_opt_by_text
from data.slice import slice_audio
from EDGE import EDGE
from data.audio_extraction.baseline_features import extract as baseline_extract
from data.audio_extraction.jukebox_features import extract as juke_extract

# sort filenames that look like songname_slice{number}.ext
key_func = lambda x: int(os.path.splitext(x)[0].split("_")[-1].split("slice")[-1])

def load_and_freeze_clip(clip_version,device):
        
    clip_model, clip_preprocess = clip.load(clip_version, device,
                                                jit=False)  # Must set jit=False for training

    clip.model.convert_weights(
        clip_model)  # Actually this line is unnecessary since clip by default already on float16

    # Freeze CLIP weights
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad = False

    return clip_model

def encode_text(raw_text,device):
    max_text_len = 20 
    if max_text_len is not None:
        default_context_length = 77
        context_length = max_text_len + 2 # start_token + 20 + end_token
        assert context_length < default_context_length
        texts = clip.tokenize(raw_text).to(device) # [bs, context_length] # if n_tokens > context_length -> will truncate  
    else:
        texts = clip.tokenize(raw_text).to(device) # [bs, context_length] # if n_tokens > 77 -> will truncate


    clip_version = 'ViT-B/32'
    clip_model = load_and_freeze_clip(clip_version,device)
    return clip_model.encode_text(texts).float()


def read_beta_from_npz_file(file_path):
    try:
        data = np.load(file_path)
        beta = data['beta']
        return beta
    except (IOError, KeyError) as e:
        print(f"Error reading npz file: {e}")
        return None

def stringintcmp_(a, b):
    aa, bb = "".join(a.split("_")[:-1]), "".join(b.split("_")[:-1])
    ka, kb = key_func(a), key_func(b)
    if aa < bb:
        return -1
    if aa > bb:
        return 1
    if ka < kb:
        return -1
    if ka > kb:
        return 1
    return 0


stringintkey = cmp_to_key(stringintcmp_)


def test(opt):
    feature_func = juke_extract if opt.feature_type == "jukebox" else baseline_extract
    sample_length = opt.out_length
    sample_size = int(sample_length / 2.5) - 1
    temp_dir_list = []
    all_cond = []
    all_filenames = []
    if opt.use_cached_features:
        print("Using precomputed features")

        # all subdirectories
        dir_list = glob.glob(os.path.join(opt.feature_cache_dir, "*/"))
        for dir in dir_list:
            file_list = sorted(glob.glob(f"{dir}/*.wav"), key=stringintkey)
            juke_file_list = sorted(glob.glob(f"{dir}/*.npy"), key=stringintkey)
            assert len(file_list) == len(juke_file_list)

            # random chunk after sanity check
            rand_idx = random.randint(0, len(file_list) - sample_size)
            file_list = file_list[rand_idx : rand_idx + sample_size]
            juke_file_list = juke_file_list[rand_idx : rand_idx + sample_size]
            cond_list = [np.load(x) for x in juke_file_list]
            all_filenames.append(file_list)
            all_cond.append(jt.array(np.array(cond_list)))
    else:
        print("Computing features for input music")
        for wav_file in glob.glob(os.path.join(opt.music_dir, "*.wav")):

            # create temp folder (or use the cache folder if specified)
            if opt.cache_features:
                songname = os.path.splitext(os.path.basename(wav_file))[0]
                save_dir = os.path.join(opt.feature_cache_dir, songname)
                Path(save_dir).mkdir(parents=True, exist_ok=True)
                dirname = save_dir
            else:
                temp_dir = TemporaryDirectory()
                temp_dir_list.append(temp_dir)
                dirname = temp_dir.name

            # slice the audio file
            print(f"Slicing {wav_file}")
            num_slice = 1000
            slice_audio(wav_file, 2.5, 5.0, dirname, num_slice)
            file_list = sorted(glob.glob(f"{dirname}/*.wav"), key=stringintkey)

            # randomly sample a chunk of length at most sample_size
            if len(file_list) >= sample_size:
                rand_idx = random.randint(0, len(file_list) - sample_size)
            else:
                rand_idx = 0
            cond_list = []

            # generate juke representations
            print(f"Computing features for {wav_file}")
            for idx, file in enumerate(tqdm(file_list)):

                # if not caching then only calculate for the interested range
                if (not opt.cache_features) and (not (rand_idx <= idx < rand_idx + sample_size)):
                    continue

                reps, _ = feature_func(file)

                # save reps
                if opt.cache_features:
                    featurename = os.path.splitext(file)[0] + ".npy"
                    np.save(featurename, reps)

                # if in the random range, put it into the list of reps we want
                # to actually use for generation
                if rand_idx <= idx < rand_idx + sample_size:
                    cond_list.append(reps)
            cond_list = jt.array(np.array(cond_list))
            all_cond.append(cond_list)
            all_filenames.append(file_list[rand_idx : rand_idx + sample_size])


    model = EDGE(opt.feature_type, opt.checkpoint)
    model.eval()

    # directory for optionally saving the dances for eval
    fk_out = None
    if opt.save_motions:
        fk_out = opt.motion_save_dir

    print("Generating dances")
    for i in range(len(all_cond)):
        mydevice = 'cuda:' + opt.gpu_for_motionclip
        text = opt.text_prompt
        emb_enc_text = encode_text(text,mydevice)
        emb_enc_text = emb_enc_text.cpu().numpy()
        emb_enc_text = jt.array(emb_enc_text)

        beta_file_path = './data/test/beta/50002.npz'

        data_include_beta = np.load(beta_file_path, allow_pickle=True)
        data_tuple = None, all_cond[i], all_filenames[i]
        m, n, _ = all_cond[i].shape 
        beta = data_include_beta['betas']
        joint_offset = data_include_beta['joint_offset'][0]
        beta = np.repeat(beta, m * n).reshape((m, n, 16))
        joint_offset = jt.array(joint_offset)
        joint_offset = jt.reshape(joint_offset, (1, 1, 24, 3))
        beta = jt.array(beta)

        emb_enc_text_copies = emb_enc_text.repeat(m, 1)
        motionclip_features = emb_enc_text_copies.reshape(m, 512)

        model.render_sample(
            data_tuple, beta, motionclip_features, joint_offset ,"test", render_dir = opt.render_dir, render_count=-1, fk_out=fk_out, render=not opt.no_render
        )
    print("Done")

    for temp_dir in temp_dir_list:
        temp_dir.cleanup()


if __name__ == "__main__":
    opt = parse_test_opt_by_text()
    test(opt)
