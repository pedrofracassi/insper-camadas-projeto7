import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
import wave
from funcoes_LPF import filtro
from scipy.fftpack import fft
from scipy import signal as window
import sounddevice as sd
from  scipy.io import wavfile 

def calcFFT(signal, fs):
  N  = len(signal)
  W = window.hamming(N)
  T  = 1/fs
  xf = np.linspace(0.0, 1.0/(2.0*T), N//2)
  yf = fft(signal*W)
  return(xf, np.abs(yf[0:N//2]))

arquivo = 'gravado.wav'

print('Lendo arquivo...')
sound, sampletime = sf.read(arquivo)
with wave.open(arquivo, "rb") as wave_file:
  samplerate = wave_file.getframerate()

fig, ((a1, b1), (a2, b2)) = plt.subplots(2, 2)

a1.plot(sound)
a1.title.set_text('Sinal gravado')

print('Fazendo FFT...')
xf1, yf1 = calcFFT(sound, samplerate)

a2.plot(xf1, yf1)
a2.title.set_text('FFT do sinal gravado')

print('Demodulando...')
portadora = np.sin(2*np.pi*14000*np.arange(len(sound))/samplerate)

demodulado = sound*portadora # por que multiplicar e não dividir? demodular não seria o reverso?

b1.plot(demodulado)
b1.title.set_text('Sinal demodulado')

print('Filtrando...')
filtered = filtro(demodulado, samplerate, 4000)
b2.plot(filtered)
b2.title.set_text('Sinal filtrado')

sd.play(filtered, samplerate)
sd.wait()

wavfile.write('demodulado.wav', samplerate, filtered)

plt.show()