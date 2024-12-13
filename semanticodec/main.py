from configparser import NoSectionError
import torch
import torch.nn as nn
import os
import torchaudio
import math

from semanticodec.modules.encoder.encoder import AudioMAEConditionQuantResEncoder
from semanticodec.modules.decoder.latent_diffusion.models.ddpm import (
    extract_encoder_state_dict,
    overlap_add_waveform,
)
from semanticodec.config import get_config
from semanticodec.modules.decoder.latent_diffusion.util import instantiate_from_config
from semanticodec.utils import extract_kaldi_fbank_feature
from huggingface_hub import hf_hub_download

# Constants
SAMPLE_RATE = 16000
SEGMENT_DURATION = 10.24
MEL_TARGET_LENGTH = 1024
AUDIOMAE_PATCH_DURATION = 0.16
SEGMENT_OVERLAP_RATIO = 0.0625


class SemantiCodec(nn.Module):
    def __init__(
        self,
        token_rate,
        semantic_vocab_size,
        ddim_sample_step=50,
        cfg_scale=2.0,
        checkpoint_path = None,
        cache_path="pretrained",
    ):
        super().__init__()
        self.token_rate = token_rate
        self.stack_factor_K = 100 / self.token_rate
        self.ddim_sample_step = ddim_sample_step
        self.cfg_scale = cfg_scale

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available(): 
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        # Initialize encoder and decoder
        config, checkpoint_path, feature_dim, lstm_layers, semanticodebook = get_config(
            token_rate, semantic_vocab_size, checkpoint_path
        )
        encoder_checkpoint_path = os.path.join(checkpoint_path, "encoder.ckpt")
        if not os.path.exists(encoder_checkpoint_path):
            if not os.path.exists(cache_path):
                os.makedirs(cache_path)
                print(f"checkpoint cache dir '{cache_path}' was created.")
            encoder_checkpoint_path = hf_hub_download(repo_id="haoheliu/SemantiCodec",filename=checkpoint_path+"/encoder.ckpt",cache_dir=cache_path)
        decoder_checkpoint_path = os.path.join(checkpoint_path, "decoder.ckpt")
        if not os.path.exists(decoder_checkpoint_path):
            decoder_checkpoint_path = hf_hub_download(repo_id="haoheliu/SemantiCodec",filename=checkpoint_path+"/decoder.ckpt",cache_dir=cache_path)

        if not os.path.exists(semanticodebook):
            semanticodebook = "/".join(semanticodebook.split("/")[-3:])
            semanticodebook = hf_hub_download(repo_id="haoheliu/SemantiCodec",filename=semanticodebook,cache_dir=cache_path)

        # Initialize encoder
        print("🚀 Loading SemantiCodec encoder")
        state_dict = torch.load(encoder_checkpoint_path, map_location="cpu")
        self.encoder = AudioMAEConditionQuantResEncoder(
            feature_dimension=feature_dim,
            lstm_layer=lstm_layers,
            centroid_npy_path=semanticodebook,
        )
        self.encoder.load_state_dict(state_dict)
        self.encoder = self.encoder.to(self.device)
        print("✅ Encoder loaded")

        # Initialize decoder
        print("🚀 Loading SemantiCodec decoder")
        self.decoder = instantiate_from_config(config["model"])
        checkpoint = torch.load(decoder_checkpoint_path, map_location="cpu")
        self.decoder.load_state_dict(checkpoint)
        self.decoder = self.decoder.to(self.device)
        print("✅ Decoder loaded")

    def load_audio(self, filepath):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"{filepath} does not exist")

        assert isinstance(filepath, str)
        waveform, sr = torchaudio.load(filepath)
        print(f"1. {waveform.shape}, {sr}")
        # resample to 16000
        if sr != SAMPLE_RATE:
            waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
            sr = SAMPLE_RATE
            print(f"2. {waveform.shape}, {sr}")
        # if stereo to mono
        if waveform.shape[0] > 1:
            waveform = waveform[0:1]
            print(f"3. {waveform.shape}, {sr}")
        # Calculate the original duration
        original_duration = waveform.shape[1] / sr
        print(f"4. {original_duration}")
        # This is to pad the audio to the multiplication of 0.16 seconds so that the original audio can be reconstructed
        original_duration = original_duration + (
            AUDIOMAE_PATCH_DURATION - original_duration % AUDIOMAE_PATCH_DURATION
        )
        print(f"5. {AUDIOMAE_PATCH_DURATION}, {original_duration % AUDIOMAE_PATCH_DURATION}, {AUDIOMAE_PATCH_DURATION - original_duration % AUDIOMAE_PATCH_DURATION}, {original_duration}")
        # Calculate the token length in theory
        target_token_len = (
            8 * original_duration / AUDIOMAE_PATCH_DURATION / self.stack_factor_K
        )
        print(f"6. {target_token_len}, {self.stack_factor_K}")

        segment_sample_length = int(SAMPLE_RATE * SEGMENT_DURATION)
        # Pad audio to the multiplication of 10.24 seconds for easier segmentations
        print("7. {segment_sample_length}")

        if waveform.shape[1] % segment_sample_length < segment_sample_length:
            print(f"8. {waveform.shape[1] % segment_sample_length}")
            waveform = torch.cat(
                [
                    waveform,
                    torch.zeros(
                        1,
                        int(
                            segment_sample_length
                            - waveform.shape[1] % segment_sample_length
                        ),
                    ),
                ],
                dim=1,
            )
            print(f"8.  {int(
                            segment_sample_length
                            - waveform.shape[1] % segment_sample_length
                        )}, {waveform.shape}")


        mel_target_length = MEL_TARGET_LENGTH * int(
            waveform.shape[1] / segment_sample_length
        )
        print(f"9. {mel_target_length}, {MEL_TARGET_LENGTH}, {waveform.shape[1] / segment_sample_length}")
        # Calculate the mel spectrogram
        mel = extract_kaldi_fbank_feature(
            waveform, sr, target_length=mel_target_length
        )["ta_kaldi_fbank"].unsqueeze(0)
        print(f"10. {mel.shape}")

        mel = mel.squeeze(1)
        print(f"11. {mel.shape}")

        print(f"12. {mel.shape}, {target_token_len}")
        assert mel.shape[-1] == 128 and mel.shape[-2] % 1024 == 0
        return mel, target_token_len

    def encode(self, filepath):
        mel, target_token_len = self.load_audio(filepath)
        tokens = self.encoder(mel.to(self.device))
        print(f"13. {tokens.shape}")
        tokens = tokens[:, : math.ceil(target_token_len), :]
        print(f"14. {tokens.shape} ,{math.ceil(target_token_len)}")
        return tokens

    def decode(self, tokens):
        windowed_token_list = self.encoder.long_token_split_window(
            tokens,
            window_length=int(512 / self.stack_factor_K),
            overlap=SEGMENT_OVERLAP_RATIO,
        )
        print(f"15. window_length={int(512 / self.stack_factor_K)}, overlap={SEGMENT_OVERLAP_RATIO}, tokens={tokens.shape}")
        print(f"16. {windowed_token_list.shape}, {windowed_token_list[0].shape}, {windowed_token_list[0]}")

        windowed_waveform = []
        for _, windowed_token in enumerate(windowed_token_list):
            print(f"17. {_}, {windowed_token}")
            latent = self.encoder.token_to_quantized_feature(windowed_token)
            print(f"18. {latent.shape}")
            print(f"19. {latent.shape[0],latent.shape[1], int(512 / self.stack_factor_K) - latent.shape[1], latent.shape[2]}")
            latent = torch.cat(
                [
                    latent,
                    torch.ones(
                        latent.shape[0],
                        int(512 / self.stack_factor_K) - latent.shape[1],
                        latent.shape[2],
                    ).to(latent.device)
                    * -1,
                ],
                dim=1,
            )
            print(f"20. {latent.shape}")

            waveform = self.decoder.generate_sample(
                latent,
                ddim_steps=self.ddim_sample_step,
                unconditional_guidance_scale=self.cfg_scale,
            )
            windowed_waveform.append(waveform)
            
            print(f"21. appended waveform: {waveform.shape}")

        output = overlap_add_waveform(
            windowed_waveform, overlap_duration=SEGMENT_DURATION * SEGMENT_OVERLAP_RATIO
        )
        print(f"22. {len(windowed_waveform)}, overlap_duration={SEGMENT_DURATION * SEGMENT_OVERLAP_RATIO}")
        print(f"23. {output.shape}")
        # Each patch step equal 16 mel time frames, which have 0.01 second
        trim_duration = (tokens.shape[1] / 8) * 16 * 0.01 * self.stack_factor_K
        print(f"24. {(tokens.shape[1] / 8)}, {16 * 0.01 * self.stack_factor_K}")
        print(f"25. {int(trim_duration * SAMPLE_RATE)}")
        return output[..., : int(trim_duration * SAMPLE_RATE)]

    def forward(self, filepath):
        tokens = self.encode(filepath)
        waveform = self.decode(tokens)
        return waveform
