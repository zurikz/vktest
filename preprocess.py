import os
from tqdm import tqdm


path = "./dataset/VCTK-Corpus-0.92/wav48_silence_trimmed"
speakers_dirs = sorted(os.listdir(path))
temp1, temp2, temp3 = "temp1.flac", "temp2.flac", "temp3.flac"

for dir in tqdm(speakers_dirs):
	speaker_path = os.path.join(path, dir)
	if not os.path.isdir(speaker_path):
		continue
	files = [x for x in os.listdir(speaker_path)]
	for file in tqdm(files):
		file = os.path.join(speaker_path, file)
		if file.endswith(".flac"):
			os.popen("ffmpeg -loglevel quiet -i " + file + " -af silenceremove=1:0:-40dB -y " + temp1).read()
			os.popen("ffmpeg -loglevel quiet -i " + temp1 + " -af areverse -y " + temp2).read()
			os.popen("ffmpeg -loglevel quiet -i " + temp2 + " -af silenceremove=1:0:-60dB -y " + temp3).read()
			os.popen("ffmpeg -loglevel quiet -i " + temp3 + " -af areverse -y " + file).read()
			os.remove(temp1)
			os.remove(temp2)
			os.remove(temp3)