# FCGPHOG; Fused-Color Gabor Pyramidal Histogram of Oriented Gradients.
FCGPHOG is a variant of HOG descriptor which uses multiple Gabor filters by rotating with a constant angle in the kernel. It uses multiple color spaces and principal component analysis (PCA) for the dimensionality reduction.

## Features:
1. Can generate HOG
2. Can generate PHOG
3. Can generate GPHOG
4. Color conversion algorithms : (RGB to HSV, ORGB, YIQ, YCbCr and DCS)
5. Can generate FC-GPHOG

## How to Use:
This implementation uses openCV 2.4 but can be used with openCV 2.x.x and c++. The code contain an entry method (main method) with a code of sample usage. There for this file can be built to run individually.

## More Details:
This is an implementation of FC-GPHOG mentioned in the below given paper. I am not an author of the paper and the code is implemented according to the best of my understanding therefore this particular implementation may not reflect the original implementation of FC-GPHOG used in the experiments mentioned in the paper. I hope this code may helpful for the research community. You may need to get permission from the original authors to use this code or the their original algorithm for commercial purposes.

Related Paper:
Sinha, Atreyee, Sugata Banerji, and Chengjun Liu. "New color GPHOG descriptors for object and scene image classification." Machine vision and applications 25, no. 2 (2014): 361-375.
