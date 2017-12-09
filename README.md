## FACE GENERATION USING GENERATIVE ADVERSARIAL NETWORKS

### INTRODUCTION
Our idea is to generate a talking face from an audio segment by using Generative Adversarial Networks (GANs). The architecture used is the proposed by David Berthelot, Thomas Schumm and Luke Metz in their work [Boundary Equilibrium Generative Adversarial Network (BEGAN)](https://arxiv.org/pdf/1703.10717.pdf).
In our case, the input to the Generator network will be the audio features of an audio segment. Those features are obtained by passing the MFCC coefficients of the audio segment through a shallow CNN architecture. By doing that, we provide information to the generator network to be able to generate a face according to the provided audio segment.


### ARCHITECTURE
The architecture proposed in the [Boundary Equilibrium Generative Adversarial Network (BEGAN)](https://arxiv.org/pdf/1703.10717.pdf) is quite different to the general way of building GANs. In this case, we have that the architecture used for the Discriminator network is an autoencoder, while the Generator has only the decoder part.
<img src="docs/Captura de pantalla 2017-12-09 a las 10.21.16.png" alt="hi" class="inline"/>
