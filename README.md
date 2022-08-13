# FastFourierTransform-Application

## Task: 
1. Implement different fourier transform algorithms  
2. Apply the algorithms and denoise/compress the given image
3. Plot the complexity of the algorithm; Analyze the runtime of the algorithms 

## Background: 
* Applications like X-Ray, electrical circuits deal with waves.
* Waves can be represented with frequencies. 
* Fequency: how fast a signal(information is contained in a signal) is changing. 
* Fourier transform allows us to switch from frequencies domain to time domain(vice versa)
* sometimes when we translate from one domain into another, we can detect 
nosies that we couldnt see in the other domain. 

## 
1. implemented fourier transform algorithm 
* Brute Force
![fourier tranform formula](https://github.com/kimyoungqq17/FastFourierTransform-Application/blob/main/README_IMG/fourier_formula.PNG)
* The cool turkey FFT 
![the cool turkey fft formula](https://github.com/kimyoungqq17/FastFourierTransform-Application/blob/main/README_IMG/fourier_formula2.PNG)
> Divide and Conquer method.
> Eventually the two terms are just a DFT of the even and odd indicies.
> Dividing a big array in two equal parts recursively untill the smaller parts became small enough to compute with the naive method without suffering large fficiency penalties.
* The two dimensional fourier transform 
![the two dimensional fourier transform formula](https://github.com/kimyoungqq17/FastFourierTransform-Application/blob/main/README_IMG/fourier_formula3.PNG)
> We have to keep in mind that images are 2D/ Discrete pixel value. 

2.  Image Processing
* Each point in an image can be represented with x,y. (Spatial Domain)
* We use fourier transform such that we can go from spatial domain to frequency domain(the rate of change of pixel values). 
* In order to have the image be compatible with the algorithms based on the power of 2, the images are resized according to the closest power of 2 independently for width and height. h = 2^(logh) / w = 2^(logw)

![fft applied](https://github.com/kimyoungqq17/FastFourierTransform-Application/blob/main/README_IMG/fft_image.PNG)

3. Image Denoising
* Removed a lot of high frequencies, which is anything that is centered around the value of pi. 0 and 2pi(2pi=0) are lower frequencies.
* Removing frequencies > 9% : didn't have a significant effect on the noiseness of the image.
* Removing frequencies < 9% : made the image blurry 

![image denoised](https://github.com/kimyoungqq17/FastFourierTransform-Application/blob/main/README_IMG/denoised.PNG)

4. Image Compression
* Reduced the amount of information stored for the image. 
* Removed the magnitude according to the percentile to keep. 

![image compressed](https://github.com/kimyoungqq17/FastFourierTransform-Application/blob/main/README_IMG/compressed.PNG)

5. Testing
* the NumPy FFT algorithm and np.allclose(arr1, arr2) were utilized to ensure the accuracy of the calculations performed by the implemented algorithms.

6. Time Complexity 

6.1 Brute Force DFT: O(N^2)

* for each k=0...N-1, iterate through all elements of the array of length N. N*N operations

6.2 The cool turkey FFT: O(NlogN)

* ![complexity](https://github.com/kimyoungqq17/FastFourierTransform-Application/blob/main/README_IMG/FFT_Complexity.PNG)

6.3 DFT2D: O(N^3)

* The 2D fourier transform runs 1D fourier transform on each row of the 2D arrray first, then performs the 1D fourier transform on each column of the 2D array. So if the number of columns is C and the number of rows is R, then the complexity canbe determined by O(C * R^2 + R * C^2) => O(2*N^3) => O(N^3)

6.4 FFT2D: O(N^2log(N))

* Similar approach as the above is applied here. O(2*N*Nlog(N)) => N^2log(N) 
