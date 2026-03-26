"""
Audio processing utilities shared by single-synthesis and batch-synthesis.
"""
import numpy as np
import torch


def concat_generated_wavs(results_dict: dict) -> np.ndarray:
    """
    Concatenate generated wav segments into a single audio array.

    Args:
        results_dict: Output from model.forward_longform(),
                      must contain 'generated_wavs' key.

    Returns:
        1-D numpy float32 array of the concatenated audio.
    """
    wavs = results_dict["generated_wavs"]
    if not wavs:
        return np.array([], dtype=np.float32)
    target_audio = wavs[0]
    for wav in wavs[1:]:
        target_audio = torch.concat([target_audio, wav], axis=1)
    return target_audio.cpu().squeeze(0).numpy()
