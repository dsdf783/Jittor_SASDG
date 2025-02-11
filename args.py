import argparse


def parse_train_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--continue_train", action="store_true", help="Resume training from the latest checkpoint")
    parser.add_argument("--project", default="runs/train", help="project/name")
    parser.add_argument("--exp_name", default="exp", help="save to project/name")
    parser.add_argument("--data_path", type=str, default="data/", help="raw data path")
    parser.add_argument(
        "--processed_data_dir",
        type=str,
        default="data/dataset_backups/",
        help="Dataset backup path",
    )
    parser.add_argument(
        "--render_dir", type=str, default="renders/", help="Sample render path"
    )

    parser.add_argument("--feature_type", type=str, default="jukebox")
    parser.add_argument(
        "--wandb_pj_name", type=str, default="EDGE", help="project name"
    )
    parser.add_argument("--batch_size", type=int, default=64, help="batch size")
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument(
        "--force_reload", action="store_true", help="force reloads the datasets"
    )
    parser.add_argument(
        "--no_cache", action="store_true", help="don't reuse / cache loaded dataset"
    )
    parser.add_argument(
        "--save_interval",
        type=int,
        default=100,
        help='Log model after every "save_period" epoch',
    )
    parser.add_argument("--ema_interval", type=int, default=1, help="ema every x steps")
    parser.add_argument(
        "--checkpoint", type=str, default="", help="trained checkpoint path (optional)"
    )
    opt = parser.parse_args()
    return opt


def parse_test_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_type", type=str, default="jukebox")
    parser.add_argument("--out_length", type=float, default=30, help="max. length of output, in seconds")
    parser.add_argument(
        "--processed_data_dir",
        type=str,
        default="data/dataset_backups/",
        help="Dataset backup path",
    )
    parser.add_argument(
        "--render_dir", type=str, default="renders/test_exp_relative_positions/wild_music", help="Sample render path"
    )
    parser.add_argument(
        "--checkpoint", type=str, default="checkpoint.pt", help="checkpoint"
    )
    parser.add_argument(
        "--music_dir",
        type=str,
        default="data/test/wavs",
        help="folder containing input music",
    )
    parser.add_argument(
        "--save_motions", action="store_true", help="Saves the motions for evaluation"
    )
    parser.add_argument(
        "--motion_save_dir",
        type=str,
        default="eval/motions",
        help="Where to save the motions",
    )
    parser.add_argument(
        "--cache_features",
        action="store_true",
        help="Save the jukebox features for later reuse",
    )
    parser.add_argument(
        "--no_render",
        action="store_true",
        help="Don't render the video",
    )
    parser.add_argument(
        "--use_cached_features",
        action="store_true",
        help="Use precomputed features instead of music folder",
    )
    parser.add_argument(
        "--feature_cache_dir",
        type=str,
        default="cached_features/",
        help="Where to save/load the features",
    )
    opt = parser.parse_args()
    return opt

def parse_test_opt_by_text():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_type", type=str, default="jukebox")
    parser.add_argument("--out_length", type=float, default=60, help="max. length of output, in seconds")
    parser.add_argument(
        "--processed_data_dir",
        type=str,
        default="data/dataset_backups/",
        help="Dataset backup path",
    )
    parser.add_argument(
        "--render_dir", type=str, default="renders/test_long_text", help="Sample render path"
    )
    parser.add_argument(
        "--checkpoint", type=str, default="checkpoint_style_wo_dropout.pt", help="checkpoint"
    )
    parser.add_argument(
        "--music_dir",
        type=str,
        default="data/test/wavs",
        help="folder containing input music",
    )
    parser.add_argument(
        "--save_motions", action="store_true", help="Saves the motions for evaluation"
    )
    parser.add_argument(
        "--motion_save_dir",
        type=str,
        default="eval/motions_style_by_long_text",
        help="Where to save the motions",
    )
    parser.add_argument(
        "--cache_features",
        action="store_true",
        help="Save the jukebox features for later reuse",
    )
    parser.add_argument(
        "--no_render",
        action="store_true",
        help="Don't render the video",
    )
    parser.add_argument(
        "--use_cached_features",
        action="store_true",
        help="Use precomputed features instead of music folder",
    )
    parser.add_argument(
        "--feature_cache_dir",
        type=str,
        default="cached_features/",
        help="Where to save/load the features",
    )
    parser.add_argument("--text_prompt", default='', type=str,
                       help="A text prompt to be generated. If empty, will take text prompts from dataset.")
    parser.add_argument('--gpu_for_motionclip', type=str, default='0')
    parser.add_argument(
        "--smpl_model_dir", type=str, default="smpl/", help="SMPL Model path"
    )
    opt = parser.parse_args()
    return opt