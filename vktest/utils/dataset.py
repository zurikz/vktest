import os
import librosa
import torch
import torchaudio
from torch.utils.data import Dataset
from typing import Tuple
from torch import Tensor

from torch import nn
from torchaudio.transforms import (
	Vad, Resample, MelSpectrogram
)

class VCTK(Dataset):
	"""Create custom VCTK 0.92 Dataset for AGAIN-VC training.

	Args:
		root (str): Root directory where the dataset's top level directory is found.
		train (bool): What split to use: if 'True', then 80 train speakers are being used,
					  else -- validation speakers.
		mic_id (str): Microphone ID. Either "mic1" or "mic2". (default: "mic2")
		audio_ext (str, optional): Audio extension. (default: ".flac")
	"""
	def __init__(
		self,
		root: str,
		train: bool,
		mic_id: str = "mic2",
		audio_ext: str = ".flac"
	) -> None:
		self._path = os.path.join(root, "VCTK-Corpus-0.92")
		self._txt_dir = os.path.join(self._path, "txt")
		self._audio_dir = os.path.join(self._path, "wav48_silence_trimmed")
		self._mic_id = mic_id
		self._audio_ext = audio_ext	
		self.segment_len = 128

		self.wav2melspec = nn.Sequential(
			Resample(orig_freq=48000, new_freq=22050),
			Vad(sample_rate=22050, 
				trigger_level=-2, 
				noise_reduction_amount=0),
			MelSpectrogram(
				sample_rate=22050,
				win_length=1024,
				n_fft=1024,
				hop_length=256,
				n_mels=80,
				f_min=0,
				f_max=11025
			)
		)

		# Extracting speaker IDs from the folder structure
		speaker_ids = os.listdir(self._txt_dir)
		if train:
			self._speaker_ids = speaker_ids[:80]
		else:
			self._speaker_ids = speaker_ids[80:]
		self._sample_ids = []

		for speaker_id in self._speaker_ids:

			if speaker_id == "p280" and mic_id == "mic2":
				continue

			utterance_dir = os.path.join(self._txt_dir, speaker_id)

			for utterance_file in sorted(
				f for f in os.listdir(utterance_dir) if f.endswith(".txt")
			):
				utterance_id = os.path.splitext(utterance_file)[0]
				audio_path_mic = os.path.join(
					self._audio_dir,
					speaker_id,
					f"{utterance_id}_{mic_id}{self._audio_ext}"
				)
				if speaker_id == "p362" and not os.path.isfile(audio_path_mic):
					continue
				splitted_utterance_id = utterance_id.split("_")
				# for each speaker 200 utterances are chosen arbitrarily
				if int(splitted_utterance_id[1]) > 200:
					break
				self._sample_ids.append(splitted_utterance_id)

	def _load_melspec(self, file_path) -> Tensor:
		waveform, _ = torchaudio.load(file_path)
		return self.wav2melspec(waveform[0])

	def _load_sample(
		self, speaker_id: str, utterance_id: str, mic_id: str
	) -> Tuple[Tensor, str, int]:
		audio_path = os.path.join(
			self._audio_dir,
			speaker_id,
			f"{speaker_id}_{utterance_id}_{mic_id}{self._audio_ext}"
		)

		melspec = self._load_melspec(audio_path)

		if melspec.shape[1] < self.segment_len:
			# circular (wrap) padding at the end of a spectrogram
			tail = melspec[:, 0:(self.segment_len - melspec.shape[1])]
			melspec = torch.cat([melspec, tail], axis=1)
		else:
			melspec = melspec[:, 0:self.segment_len]

		return (melspec, speaker_id, utterance_id)

	def __getitem__(self, n: int) -> Tuple[Tensor, str, int]:
		speaker_id, utterance_id = self._sample_ids[n]
		return self._load_sample(speaker_id, utterance_id, self._mic_id)

	def __len__(self) -> int:
		return len(self._sample_ids)
