# **Self Driving Car Simulation** 
### Purpose
* Develop a self-drving car model based on convolutional neural network. 
* Use Keras On top of Tensor Flow as our machine learning API. 
* Use two diffrent data inputs (Steering Angle and Car Camera). 
* Implement the enviournment as a simulation in Udacity's car simulator app. 
### Challenges
* Training the data takes large amout of time. (One solution would be to use Google Cloud Platform). 
### Expectation
* The car being able to drive itself with minimal crashes. 
### Concepts
This project is based on a paper by  Mariusz Bojarski, Ben Firner, Beat Flepp, Larry Jackel, Urs Muller and Karol Zieb for Nvidia called End-to-End Deep Learning for Self-Driving Cars [Linked Here](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/)
#### **Training the Neural Network**
![Image of Neural Net Model](https://devblogs.nvidia.com/parallelforall/wp-content/uploads/2016/08/training-768x358.png)
#### **Neural Network Architecture** 
Please note that the 1164 layer neural network and the 200 filter convolutional neural net is ommited from the project due to computatinal limitations. 
![Image of Net Architacture](https://devblogs.nvidia.com/parallelforall/wp-content/uploads/2016/08/cnn-architecture-768x1095.png)
