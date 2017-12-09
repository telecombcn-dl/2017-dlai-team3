## FACE GENERATION USING GENERATIVE ADVERSARIAL NETWORKS

### INTRODUCTION
<img src="docs/2.png" alt="hi" class="inline"/>
Our idea is to generate a talking face from an audio segment by using Generative Adversarial Networks (GANs). The architecture used is the proposed by David Berthelot, Thomas Schumm and Luke Metz in their work [Boundary Equilibrium Generative Adversarial Network (BEGAN)](https://arxiv.org/pdf/1703.10717.pdf).
In our case, the input to the Generator network will be the audio features of an audio segment. Those features are obtained by passing the MFCC coefficients of the audio segment through a shallow CNN architecture. By doing that, we provide information to the generator network to be able to generate a face according to the provided audio segment.

### DATASET
<img src="docs/1" alt="hi" class="inline"/>
In order to train our network, we have also created our own dataset.
First we extract videos from the youtube platform. We wanted to have well centered faces and with the cleanest audio possible, for that reason we chosed to download videos from Donald Trump public speeches.
Once the videos were selected, we had to process them to obtain both the face and the audio corresponding to the face. 
BLAH BLAH BLAH BLAH BLAH


### ARCHITECTURE
The architecture proposed in the [Boundary Equilibrium Generative Adversarial Network (BEGAN)](https://arxiv.org/pdf/1703.10717.pdf) is quite different to the general way of building GANs. In this case, we have that the architecture used for the Discriminator network is an autoencoder, while the Generator has only the decoder part. In order to conditioned the Generator network to generate talking faces according to an audio segment, we input to the generator audio features instead of a random noise. Those features have been extracted from a CNN which had as an input the audio segment. This CNN is composed by: Conv (3x3,64) + Conv (3x3,128) + Pool(2)  + Conv (3x3,256) + Conv (3x3,512) + Dense(512) + 
Dense (256).
<img src="docs/1" alt="hi" class="inline"/>
