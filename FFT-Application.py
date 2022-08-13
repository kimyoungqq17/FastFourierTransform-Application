import numpy as np
import cv2
import math
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import time

from matplotlib import pyplot as plt

from argparse import ArgumentParser
def main():
  parser = ArgumentParser()
  parser.add_argument('-m', type=int, default=1, action='store', help='mode to select 1: fast, 2: denoise, 3:compress, 4:plot runtime')
  parser.add_argument('-i', type=str, default='moonlanding.png', action='store', help='moonlanding.png')
  args = parser.parse_args()
  
  mode = args.m
  filename = args.i

  if args.m == 1:
    mode_1(filename)
  elif args.m == 2:
    mode_2(filename)
  elif args.m == 3:
    mode_3(filename)
  elif args.m == 4:
    mode_4()
  else:
    raise ValueError("Invalid mode! Must be either 1,2,3, or 4")
  plt.show()

##### EXECUTION MODES #####

def mode_1(filename='moonlanding.png', title1='original', title2='fft 2d with lognorm'):
  img = cv2.imread(filename,0)
  img = img_size_fix(img)

  image_array = np.array(img)

  fft_image = fft2(image_array)

  write_fft_csv(fft_image, filename.split('.')[0] + '_mode_1')

  fig, ax = plt.subplots(1, 2)
  ax[0].imshow(image_array)
  ax[0].set_title(title1)
  ax[1].imshow(np.abs(fft_image), norm=colors.LogNorm())
  ax[1].set_title(title2)

def mode_2(filename='moonlanding.png', title1='original',title2='denoised', threshold = 0.09):
  img = cv2.imread(filename,0)
  img = img_size_fix(img)

  image_array = np.array(img)

  #denoise the image
  fft_image = fft2(image_array)
  copy_image = fft_image.copy()
  row, column = copy_image.shape

  top, bottom = int(row * threshold), int(row*(1-threshold))
  left, right = int(column * threshold), int(column*(1 - threshold))

  print('Number of coefficients used:', int(threshold*row*column))
  print('Fraction to keep:', threshold)
  
  copy_image[top:bottom] = 0
  copy_image[:,left:right] = 0
  denoise = ifft2(copy_image).real

  write_fft_csv(denoise, filename.split('.')[0] + '_mode_2')

  #display the image
  fig, ax = plt.subplots(1, 2)
  fft_inv_img = fft2(image_array)
  ax[0].imshow(image_array, plt.cm.gray)
  ax[0].set_title(title1)
  ax[1].imshow(denoise, plt.cm.gray)
  ax[1].set_title(title2)

def mode_3(filename='moonlanding.png', title='Compression'):
  img = cv2.imread(filename,0)
  img = img_size_fix(img)

  image_array = np.array(img)

  fft_image = fft2(image_array)
  
  compression = [0, 10, 30, 50, 70, 95] # 0 to 95 (incl) 15 step interval

  fig, ax = plt.subplots(2, 3)

  fig.set_size_inches(15,15)
  print(np.max(fft_image), np.abs(np.min(fft_image)))
  for factor, axis in zip(compression, ax.flatten()):
    copy_image = fft_image.copy()
    threshold = np.percentile(abs(copy_image), factor)
    print(threshold)
    copy_image[abs(copy_image) < threshold] = 0
    compressed_image = ifft2(copy_image).real
    write_fft_csv(copy_image.real, filename.split('.')[0] + '_mode_3_p' + str(factor))
    axis.set_title(title + ' ' + str(factor) + ' percentile')
    axis.imshow(compressed_image)


def mode_4(maxsize=9):
  x = np.arange(5, max(5, maxsize)+1)

  test_arrays = [ np.random.rand(2**size, 2**size) for size in x ]
  result = []

  print("Testing two dimensional arrays of squared sizes:\n", [2**v for v in x])
  for j, arr in zip(x, test_arrays):
    time_res = []
    result.append(time_res)
    print("Test array of size ",'2^' + str(int(np.log2(arr.shape[0]))) ,arr.shape)
    for i in range(10):
      print("iteration ",i,end=' ')
      start = time.perf_counter()
      #### Program Execution ####
      calc = fft2(arr)
      end = time.perf_counter()
      print("time:", end-start)

      time_res.append(end-start)
  
  y = [ np.mean(res) for res in result ]
  e = [ np.std(res) for res in result ]

  print("Test results:",*[ str(round(dx,4)) + '-> avg: ' + str(round(dy,4)) + ' std: ' + str(round(de,4)) for dx, dy, de in zip(x,y,e) ],sep='\n')

  points = np.asarray([ [v,w] for res, v in zip(result, x) for w in res ])

  fig, tst = plt.subplots(1,2)
  fig.set_size_inches(12,6)
  ax, pts = tst
  ax.set_xlabel('size')
  ax.set_ylabel('runtime (mean & std deviation in seconds)')
  ax.errorbar(x, y, e, linestyle='None', marker='^')

  pts.set_xlabel('size')
  pts.set_ylabel('runtime (each execution in seconds)')
  pts.scatter(points[:,0], points[:,1])

############################


#####  DFT FUNCTIONS  #####

# DFT and inverse
def dft(x):
  x = np.asarray(x)
  N = x.shape[0]
  return np.asarray([ np.sum(x * np.exp(-2j * np.pi * k * np.arange(N) / N )) for k in range(N)])

def idft(X):
  X = np.asarray(X)
  N = X.shape[0]
  return np.asarray([ 1/N * np.sum( X * np.exp(2j * np.pi * k * np.arange(N) / N) ) for k in range(N) ])

# FFT and inverse

def fft(x):
  x = np.asarray(x, dtype=complex)
  N = x.shape[0]

  if N % 2 != 0:
    raise Exception("Must be even value N: " + str(N))
  if N <= 8:
    return dft(x)

  Eseries = fft(x[::2])  # dft(x[::2]) or fft(x[::2])
  Oseries = fft(x[1::2]) # dft(x[1::2]) or fft(x[1::2])

  half_N = N//2

  Cseries = np.exp(-2j * math.pi * np.arange(half_N) / N ) 
  return np.concatenate([ Eseries + Cseries * Oseries, Eseries - Cseries * Oseries])

def ifft(X):
  X = np.asarray(X)
  N = X.shape[0]

  if N % 2 != 0:
    raise Exception("Must be even value N: " + str(N))
  if N <= 8:
    return idft(X)


  Eseries = ifft(X[::2])
  Oseries = ifft(X[1::2])
  half_N = N//2
  Cseries =  np.exp(2j * np.pi * np.arange(half_N) / N)

  return 1/2 * np.concatenate([ Eseries + Cseries * Oseries, Eseries - Cseries * Oseries])

# DFT and FFT but for 2d arrays

def dft2(f):
  f = np.asarray(f, dtype=complex)
  M = f.shape[0]
  N = f.shape[1]

  result = np.array([ dft(row) for row in f ])
  result = result.transpose()
  result = np.array([ dft(col) for col in result])
  result = result.transpose()
  
  return result

def idft2(F):
  F = np.asarray(F, dtype=complex)
  M = F.shape[0]
  N = F.shape[1]

  result = np.array([ idft(row) for row in F ])
  result = result.transpose()
  result = np.array([ idft(col) for col in result])
  result = result.transpose()
  
  return result

def fft2(f):
  f = np.asarray(f, dtype=complex)
  M = f.shape[0]
  N = f.shape[1]

  result = np.array([ fft(row) for row in f ])
  result = result.transpose()
  result = np.array([ fft(col) for col in result])
  result = result.transpose()

  return result

def ifft2(F):
  F = np.asarray(F, dtype=complex)
  M = F.shape[0]
  N = F.shape[1]

  result = np.array([ ifft(row) for row in F ])
  result = result.transpose()
  result = np.array([ ifft(col) for col in result])
  result = result.transpose()

  return result


# Image helper method

def img_size_fix(img):
  arr = np.asarray(img)
  height = int(2**( np.round(np.log2(arr.shape[0])) ))
  width = int(2**( np.round(np.log2(arr.shape[1])) ))
  return cv2.resize(img, (height, width))

def write_fft_csv(arr2d, filename):
  with open(filename + ".csv", 'w') as f:
    for row in arr2d:
      f.write(','.join([ str(c) for c in row ]))


if __name__ == '__main__':
  main()