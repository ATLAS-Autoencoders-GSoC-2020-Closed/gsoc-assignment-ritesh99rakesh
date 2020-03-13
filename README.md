# ATLAS-autoencoder-evaluation

This project contains the solution to the excercise. I have tried 2 approaches here:

1. Reproduce the original work [https://github.com/Skelpdar/HEPAutoencoders](https://github.com/Skelpdar/HEPAutoencoders) in the file `4D/fastai_AE_3D_200.ipynb`. Here the network is of the form:
  
    `layer1 -> tanh() -> layer2 -> tanh() -> layer3 ->tanh() -> layer4 ->...`
2. Change the network: I changed the network in `4D/fastai_AE_3D_200_LeakyReLU.ipynb` in the following way:
    
    `layer1 -> tanh() -> layer2 -> LeakyReLU() -> layer3 ->tanh() -> layer4 ->...`

    By using LeakyReLU I was able to get better result. For comparison between original and my model, see `FASTAI Autoencoder.pdf` file or the Google Slides submitted in the Google Form.
  
Reasons for LeakyReLU network performing better than original network:
1. The biggest advantage of ReLu is indeed non-saturation of its gradient, which greatly accelerates the convergence of stochastic gradient descent compared to the sigmoid / tanh functions.
2. Sparsity effects of ReLu activations and induced regularization.
3. Compared to tanh / sigmoid neurons that involve expensive operations (exponentials, etc.), the ReLU can be implemented by simply thresholding a matrix of activations at zero.

### Code

1. The 2 jupyter notebooks contains the code and plot for the excercise: `4D/fastai_AE_3D_200.ipynb` and `4D/fastai_AE_3D_200_LeakyReLU.ipynb`.
2. The new network that I proposed can be found in `nn_utils.py`, here is the code:

    ```python3
    class AE_3D_200_leaky(nn.Module):
    def __init__(self, n_features=4):
        super(AE_3D_200_leaky, self).__init__()
        self.en1 = nn.Linear(n_features, 200)
        self.en2 = nn.Linear(200, 100)
        self.en3 = nn.Linear(100, 50)
        self.en4 = nn.Linear(50, 3)
        self.de1 = nn.Linear(3, 50)
        self.de2 = nn.Linear(50, 100)
        self.de3 = nn.Linear(100, 200)
        self.de4 = nn.Linear(200, n_features)
        self.tanh = nn.Tanh()
        self.leakyReLU = nn.LeakyReLU(negative_slope=0.02)

    def encode(self, x):
        return self.en4(self.tanh(self.en3(self.leakyReLU(self.en2(self.tanh(self.en1(x)))))))

    def decode(self, x):
        return self.de4(self.tanh(self.de3(self.leakyReLU(self.de2(self.tanh(self.de1(self.tanh(x))))))))

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)

    def describe(self):
        return 'in-200-100-50-3-50-100-200-out'
    ```

### Notes

1. Due to large training time, I had to reduce the total number of iterations and hence did not get the results as good the fully trained network in the [https://github.com/Skelpdar/HEPAutoencoders](https://github.com/Skelpdar/HEPAutoencoders)
