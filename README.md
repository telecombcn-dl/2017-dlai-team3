## FACE GENERATION USING GENERATIVE ADVERSARIAL NETWORKS

### INTRODUCTION
<img src="docs/2.png" alt="hi" class="inline"/>
Our idea is to generate a talking face from an audio segment by using Generative Adversarial Networks (GANs). The architecture used is the proposed by David Berthelot, Thomas Schumm and Luke Metz in their work [Boundary Equilibrium Generative Adversarial Network (BEGAN)](https://arxiv.org/pdf/1703.10717.pdf).
In our case, the input to the Generator network will be the audio features of an audio segment. Those features are obtained by passing the MFCC coefficients of the audio segment through a shallow CNN architecture. By doing that, we provide information to the generator network to be able to generate a face according to the provided audio segment.

### DATASET
<img src="docs/3.png" alt="hi" class="inline"/>
In order to train our network, we have also created our own dataset.
First we extract videos from the youtube platform. We wanted to have well centered faces and with the cleanest audio possible, for that reason we chosed to download videos from Donald Trump public speeches.
Once the videos were selected, we had to process them to obtain both the face and the audio corresponding to the face. 
BLAH BLAH BLAH BLAH BLAH


### AUDIO FEATURE EXTRACTION
<img src="docs/6.png" alt="hi" class="inline"/>
In order to train our network, we have also created our own dataset.
First we extract videos from the youtube platform. We wanted to have well centered faces and with the cleanest audio possible, for that reason we chosed to download videos from Donald Trump public speeches.
Once the videos were selected, we had to process them to obtain both the face and the audio corresponding to the face. 
BLAH BLAH BLAH BLAH BLAH


### ARCHITECTURE
The architecture proposed in the [Boundary Equilibrium Generative Adversarial Network (BEGAN)](https://arxiv.org/pdf/1703.10717.pdf) is quite different to the general way of building GANs. In this case, we have that the architecture used for the Discriminator network is an autoencoder, while the Generator has only the decoder part. In order to conditioned the Generator network to generate talking faces according to an audio segment, we input to the generator audio features instead of a random noise. Those features have been extracted from a CNN which had as an input the audio segment. This CNN is composed by: Conv (3x3,64) + Conv (3x3,128) + Pool(2)  + Conv (3x3,256) + Conv (3x3,512) + Dense(512) + 
Dense (256).
<img src="docs/1.png" alt="hi" class="inline"/>

### PROBLEMS and SOLUTIONS
The training of the system was not as straightforward as we thought. We had to face some problems during the training phase and the generation of the dataset.
During training, the Generator's output was always the same in a batch, so we had a mode collapse problem. 
<img src="docs/4.png" alt="hi" class="inline"/>
In this image we can see at the left, the faces generated by the Generator network. Each one of the rows correspond to a batch, so we are showing two samples for each batch. As can be seen, the faces generated in each of the batches are the same within the batch. To the right, we have at the top the input image and below the images generated by the Discriminator network (Remember that the Discriminator in the BEGAN architecture is an autoencoder, so it also generates images). 

In order to overcome the situation, we concatenate to the input audio features a random noise. The reason is that audio features were very similar between each other, so maybe the network was not able to obtain enough information and so it fails in generating different images. The introduction of a random sequence to the input, was thought to provide this variability and so improve the results. But... it also failed.
<img src="docs/5.png" alt="hi" class="inline"/>
Here we can see that we still generate the same samples within the batch, and what is more, the Generator network at some point fails completely, and it is not able to ouput faces anymore. 

We decided to finally simplify the network, and input only noise to the generator network. That would be, generate different Trump faces without any audio condition. 
We can see here the evolution of the variable Kt and MGlobal. As can be seen MGlobal decreases, so the generator is able to generate more realistic faces, but, we still generate the same face...

<img src="docs/7.png" alt="hi" class="inline"/> <img src="docs/8.png" alt="hi" class="inline"/> <img src="docs/output_XxVvWq.gif" alt="hi" class="inline"/>


