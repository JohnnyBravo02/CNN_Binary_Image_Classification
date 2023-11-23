# CNN_Binary_Image_Classification

### Goal
 _Design and train a **Small Convolutional Neural Network** in Julia and FluxML that accurately classifies images into either 2 classes:_
 - _Dog (1)_
 - _Cat (2)_

### Design
<details>
  <summary>Neural Network Architecture</summary>
  
  - Convolutional Layers: _3_
    - Input Feature Maps: _3_
    - Ouput Feature Maps: _16_
    - Activation: _ReLU_
  - Pooling Layers: _3_
    - Kernel Size: _2x2_
    - Position: _Directly after Convolutional Layer_
  - Flattening Layer: _1_
    - Position: _After last Polling Layer_
  - Dense Layers: _3_
    - $1^{st}$ nodes: _16384_
      - Activation: _ReLU_
    - $2^{nd}$ nodes: _5250_
      - Activation: _Sigmoid_
    - $3^{rd}$ nodes: _1_
</details>

<details>
  <summary>Hyperparameters</summary>
  
  - Learning Rate ($\alpha$): _0.01_
  - Momentum ($\psi$): _0.0001_
  - Kernel Size ($\kappa$): _3x3_
  - Stride ($\zeta$): _1_
  - Padding ($\rho$): _0_
  - Weight Decay ($\lambda$): _0.0004_
  - Batch Size: 128
</details>

<details>
  <summary>Training</summary>
  
  - Loss Function: _Log Cross Entropy_
  - Optimizer: _Gradient Descent ($\alpha$, $\psi$)_
</details>

### Training Metrics
<details>
  <summary>Loss Log</summary>

  At Last Epoch
  
  Training Loss: _0.1_
  
  Validation Loss: _0.1_
  
  ![Screenshot 2023-11-23 174355](https://github.com/JohnnyBravo02/CNN_Binary_Image_Classification/assets/140975510/bfc0d9ea-e94e-4ffd-8b7a-78673ad2d5af)

</details>

<details>
  <summary>Accuracy Log</summary>

  At Last Epoch
  
  Training Accuracy: _97.07_
  
  Validation Accuracy: _97.1_
  
  ![Screenshot 2023-11-23 174342](https://github.com/JohnnyBravo02/CNN_Binary_Image_Classification/assets/140975510/aff72e80-14af-467b-96cd-51b166745d5c)


</details>

### Test
Test Accuracy: _97.15%_
