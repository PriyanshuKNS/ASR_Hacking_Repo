import numpy as np
import librosa
import soundfile as sf
import torch
import contextlib
import warnings
import random
warnings.filterwarnings("ignore")



# logging.getLogger().setLevel(logging.ERROR)

from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer

# Load ASR model and tokenizer from the local directory
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
print("\n==================================================================\n") 


def transcribe(audio_vector):
	global model 
	global tokenizer
	input_values = tokenizer(audio_vector, return_tensors="pt", padding="longest").input_values
	with torch.no_grad():
		logits = model(input_values).logits
	predicted_ids = torch.argmax(logits, dim=-1)
	transcription = tokenizer.batch_decode(predicted_ids)[0]
	return transcription 




# original_audio_path = "2.wav"
# original_audio, sr = librosa.load(original_audio_path, sr=None)
# print("original_audio.size : ", len(original_audio))
# print(original_audio)
# print("max: ", max(original_audio), ", min: ", min(original_audio))
# modified_audio = librosa.util.normalize(modified_audio)
# sf.write(modified_audio_path, modified_audio, sr)

audio, sr = librosa.load("260-123286-0022.wav") 


"""
	modified_audio = attacker_model(audio) 

"""
def attacker_model1(audio):
    noise_range = 1e-2
    return audio + [random.uniform(-noise_range, noise_range) for _ in range(len(audio))]

def attacker_model2(audio):
    st = 1e-4  
    end = 1e-2
    diff = end - st
    return audio + [(st + i*diff / float(len(audio))) for i in range(len(audio))]


modified_audio = attacker_model2(audio) 

t_audio = transcribe(audio) 
t_modified_audio = transcribe(modified_audio) 



## 1. QUANTIZING THE AUDIO TO DEFEND AGAINST THE ATTACK 
recovered_audio = audio.copy()
K = 1e-2  # Making recovered_audio[i] = n*K, where n is an integer
for i in range(len(audio)):
    recovered_audio[i] = int(modified_audio[i]/K) * K 
    
t_recovered_audio = transcribe(recovered_audio)  


print("t_audio: ",  t_audio)
print("t_modified_audio: ",  t_modified_audio)
print("t_recovered_audio, Quantized: ",  t_recovered_audio)



## 2. SMOOTHING THE AUDIO FOR DEFENSE 
recovered_audio = modified_audio.copy() 
K = 10   # The width of sliding window
n = len(audio)  
for i in range(K,n-K):
     recovered_audio[i] = sum(recovered_audio[i-K:i+K]) / (2*K) 
     
t_recovered_audio = transcribe(recovered_audio) 
print("t_recovered_audio, Smoothing: ", t_recovered_audio) 

  
    














