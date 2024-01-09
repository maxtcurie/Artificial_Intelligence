import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt

# Making sure we're using TensorFlow 2.x
if not tf.__version__.startswith('2'):
    raise ValueError('This code requires TensorFlow V2.x')

tfd = tfp.distributions
tfb = tfp.bijectors
tfk = tfp.math.psd_kernels

# Generate sample data
np.random.seed(41)
x = np.sort(np.random.uniform(-1., 1., 30))[:, np.newaxis]
y = np.sin(3 * np.pi * x[:, 0]) + 0.3 * np.cos(9 * np.pi * x[:, 0]) + 0.5 * np.random.normal(size=30)

# Visualize the sample data
plt.scatter(x, y, marker='o')
plt.title("Sample Data")
plt.show()

# Define the Gaussian Process kernel (RBF Kernel in this case)
amplitude = tfp.util.TransformedVariable(
    1., bijector=tfb.Softplus(), dtype=tf.float64)
length_scale = tfp.util.TransformedVariable(
    1., bijector=tfb.Softplus(), dtype=tf.float64)
kernel = tfk.ExponentiatedQuadratic(amplitude, length_scale)

# Define the Gaussian Process
observation_noise_variance = tfp.util.TransformedVariable(
    np.exp(-1), bijector=tfb.Softplus(), dtype=tf.float64)

gp = tfd.GaussianProcess(
    kernel=kernel,
    index_points=x,
    observation_noise_variance=observation_noise_variance)

# Define the log likelihood
def log_prob():
    return gp.log_prob(y)

# Optimize log likelihood (train the model)
num_iters = 1000
optimizer = tf.optimizers.Adam(learning_rate=.05, beta_1=.5, beta_2=.99)
losses = np.zeros(num_iters)
for i in range(num_iters):
    with tf.GradientTape() as tape:
        loss = -log_prob()
    grads = tape.gradient(loss, gp.trainable_variables)
    optimizer.apply_gradients(zip(grads, gp.trainable_variables))
    losses[i] = loss.numpy()

# Visualize training progress
plt.figure(figsize=(12, 4))
plt.plot(losses)
plt.title("Training Loss")
plt.show()

# Generate predictive values
gprm = tfd.GaussianProcessRegressionModel(
    kernel=kernel,
    index_points=np.linspace(-1.1, 1.1, 200)[:, np.newaxis],
    observation_index_points=x,
    observations=y,
    observation_noise_variance=observation_noise_variance)

# Visualize predictions
upper, lower = gprm.mean() + [2 * gprm.stddev(), -2 * gprm.stddev()]
plt.scatter(x, y, marker='o')
plt.plot(np.linspace(-1.1, 1.1, 200), gprm.mean())
plt.fill_between(np.linspace(-1.1, 1.1, 200), upper, lower, color='k', alpha=.1)
plt.title("Predictions with 95% Credible Interval")
plt.show()
