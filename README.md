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
  - Padding ($rho$): _0_
  - Batch Size: 128
</details>

<details>
  <summary>Training</summary>
  
  - Loss Function: _Log Cross Entropy_
  - Optimizer: _Gradient Descent ($\alpha$, $\psi$)_
</details>

### Test
Test Accuracy: _89.06%_
