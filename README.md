# NNframeworkgroup

Goal: Modular Python Framework to fit neural networks to energy/force landscapes

Module 1 (Red Team):
Completed:
-obtained data set for unary metal cluster (Gold, via Henkelman group)

Development goals (short term):
-obtain data set for molecules
-obtain data set for bulk structures (e.g. binary from GASP)

Package goals (long term):
-functions for data generation (random generation, symmetry groups, etc)
*-interfacing with existing databases?
*-prebuilt (included) testing data sets?
*-interfacing with VASP and LAMMPS optional?

Module 2 (Blue Team):
Completed:
-implemented molML's behler-parinello representation
-implemented conversion from structure files to normalized G1/G2 arrays

Development goals (short term):
-investigate other representation schemes

Package goals (long term):
-multiple, user-selectable data representation schemes
-output NN-formatted data, with transformations as necessary

Module 3 (Yellow Team):
Completed:
-created preliminary 3-layer back-propagation neural network

Development goals (short term):
-master Keras, TensorFlow
-integrate NeuPy and scikit-learn

Package goals (long term):
-multiple, user-selectable neural networks
-benchmarking for comparison?
*-active learning functionality (network trains itself)?
