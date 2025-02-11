import jittor as jt
from jittor import nn
from model.blocks import Upsample
from model.blocks import AdaptiveInstanceNorm1d

def test_upsample():
    print("Testing Upsample...")
    upsample = Upsample(scale_factor=2, mode='nearest')
    x = jt.array([[1, 2], [3, 4]]).float().unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, 2, 2)
    output = upsample.execute(x)
    print("Input shape:", x.shape)
    print("Output shape:", output.shape)
    assert output.shape == (1, 1, 4, 4), "Upsample output shape mismatch!"


def test_adaptive_instance_norm():
    print("Testing AdaptiveInstanceNorm1d...")
    norm_layer = AdaptiveInstanceNorm1d(num_features=3)
    norm_layer.weight = jt.ones(3)
    norm_layer.bias = jt.zeros(3)

    x = jt.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]])  # Shape: (1, 3, 3)
    output = norm_layer.execute(x)
    print("Input:", x)
    print("Output:", output)
    assert output.shape == x.shape, "AdaptiveInstanceNorm1d output shape mismatch!"

def test_activation_functions():
    print("Testing Activation Functions...")
    x = jt.array([-1.0, 0.0, 1.0])
    relu = nn.relu
    assert (relu(x) == jt.array([0.0, 0.0, 1.0])).all(), "ReLU failed!"

    lrelu = nn.leaky_relu
    assert (lrelu(x) == jt.array([-0.2, 0.0, 1.0])).all(), "Leaky ReLU failed!"

    tanh = nn.Tanh()
    assert (tanh(x) == jt.tanh(x)).all(), "Tanh failed!"

    print("All activation functions passed.")

def run_all_tests():
    test_upsample()
    test_adaptive_instance_norm()
    test_activation_functions()
    print("All tests passed successfully!")

if __name__ == "__main__":
    jt.flags.use_cuda = 1  # Enable CUDA if available
    run_all_tests()
