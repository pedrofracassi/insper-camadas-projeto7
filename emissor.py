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

arquivo = 'among.wav'

sound, sampletime = sf.read(arquivo)
with wave.open(arquivo, "rb") as wave_file:
  samplerate = wave_file.getframerate()

print(samplerate)

fig, ((a1, b1, c1), (a2, b2, c2)) = plt.subplots(2, 3)

# juntar num canal só
sound = np.mean(sound, axis=1)

a1.plot(sound)
a1.title.set_text('Sinal original')

print('Aplicando low pass filter...')
filtered = filtro(sound, samplerate, 4000)
b1.plot(filtered)
b1.title.set_text('Sinal filtrado')

print('Calculando FFTs')
xf1, yf1 = calcFFT(sound, samplerate)
xf2, yf2 = calcFFT(filtered, samplerate)

a2.plot(xf1, yf1)
a2.plot([4000], [0], 'o', color='red')
a2.title.set_text('FFT do sinal original')

b2.plot(xf2, yf2)
b2.plot([4000], [0], 'o', color='red')
b2.title.set_text('FFT do sinal filtrado')

portadora = np.sin(2*np.pi*14000*np.arange(len(filtered))/samplerate)

modulado = filtered*portadora

c1.plot(modulado)
c1.title.set_text('Sinal modulado')

normalizado = modulado/np.max(np.abs(modulado))
c2.plot(normalizado)
c2.title.set_text('Sinal modulado normalizado')

sd.play(sound, samplerate)
sd.wait()

sd.play(filtered, samplerate)
sd.wait()

sd.play(modulado, samplerate)
sd.wait()

wavfile.write('modulado.wav', samplerate, modulado)

plt.show()