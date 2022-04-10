1st Place HackMIT 2021

# SenseEye

Being able to extract heart rate contactless with an RGB camera under generalized conditions (e.g. robust to face view, motion, illumination, can handle many people in the scene) is a crucial keystone to solving many problems, with applications ranging from elderly monitoring to security. We propose and flesh out the foundations of SenseEye to make headway on this issue. 

SenseEye consists of two modules: a face-processing module, which extracts the relevant features needed downstream, and a heart rate detection module. First, all faces in the scene are detected with the state-of-the-art MTCNN and cropped. Next, we use FaceNet to compute face embeddings for each crop and query our face database to perform facial recognition. The face database essentially keeps track of an average embedding for each person and, given an embedding, we want to find the closest average embedding. 

We combine several previous works to get a robust heart rate detection algorithm. We first detect the face landmarks in order to segment the ROI. that is invariant to motion (e.g. smiles), specifically the cheek regions. Next, we perform skin tone normalization to make the image invariant to illumination (https://www.es.ele.tue.nl/~dehaan/pdf/169_ChrominanceBasedPPG.pdf), modifying it so it runs in a streaming fashion. Third, we remove regions that indicate the impossibility of heart rate detection (e.g. face looking down) by segmentation of bad regions in the raw signal (https://www.cv-foundation.org/openaccess/content_cvpr_2014/papers/Li_Remote_Heart_Rate_2014_CVPR_paper.pdf). Finally, we perform a fourier transform on our signal and bandpass the acceptable range of heart rate. 

If the face is not visible, we can use multiple cameras, which is feasible if the application allows the creator control of the scene (e.g. surveillance or monitoring applications).

## Engineering Optimizations:
The face database needs to be very efficient for a large number of people (i.e. if the camera is used in surveillance). We propose a novel setup: there are two databases an all-encompassing database with all faces and a cache. The cache contains only recently seen people and is maintained via LRU. The intuition is that usually people move around frequently in a scene before disappearing for a large amount of time.

We need to determine if a pixel is in the ROI (manually defined using facial landmarks). This is a classic problem-- determine if point is in polygon. Taking advantage of the fact that the face regions are small, we break the polygons into triangles and line sweep the x-axis, adding the relevant pixels based on y-axis. 

Another possible optimization that can be explored is doing sparse sampling and relying on optical flow for face detection and face occlusion detection. 

## Applications:
It is possible to add other health analytic features (lots of NIH papers on using camera stuff to get blood pressure etc)

Elderly monitoring- lots of old people die because they are alone and no one finds out until weeks later. This system can help. Same idea for workers in high-altitude regions.

Animal farming-- instead of using human faces, do it on animals-- cut down costs i.e. have a camera swing around on a pole every morning in the chicken pen-- and possibly remove the need for cages. 

Security-- people wear fake faces or hold 2d images/3d morphs to attack camera-based systems. Use heart rate to verify face is real. 
