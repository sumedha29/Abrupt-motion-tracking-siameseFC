# Abrupt-motion-tracking-siameseFC

Dependencies to be installed :
-----------------------------------------
1.	Python – 3.7.3
2.	Tensorflow – 1.13.1
3.	Pillow – 6.1.0
4.	Numpy – 1.18.1
5.	Matplotlib – 3.1.3
6.	Scipy – 1.4.1
7.	H5py – 2.10.0
8.	Hdf5storage – 0.1.15
9.	Hdf5 – 1.10.4
10.	Opencv-python – 4.2.0.32

Repository folders description :
-------------------------------------------
1.	‘parameters’ : Contains different parameters which can be changed to configure the execution environment
2.	‘pretrained’ : Contains pretrained cfnet network weights
3.	‘src’ : Contains list of procedures for the Siamese FC implementation

Running the tracker :
-------------------------------------------
1.	All the initial parameters regarding loading the data, enabling the visualization, loading the pretrained cfnet weights and the hyperparameters have been assigned in the code.
2.	Run the ‘run_tracker_evaluation_meanshift.py’ file to execute the SiameseFC tracker integrated with the Mean Shift filter. 
3.	Run the ‘run_tracker_evaluation_lucaskanade.py’ file to execute the SiameseFC tracker integrated with the Lucas Kanade algorithm.
