# from the web page https://fluxml.ai/Flux.jl/stable/models/quickstart/

# With Julia 1.7+, this will prompt if neccessary to install everything, including CUDA:
using Flux, Statistics, ProgressMeter

# Generate some data for the XOR problem: vector of length 2, as columns of a matrix:
noisy = rand(Float32, 2, 1000)
truth = [xor(col[1]>0.5, col[2]>0.5) for col in eachcol(noisy)]
# Define our model, a multi-layer perceptron with one hidden layer of size 3.
model = Chain(
    Dense(2 => 3, tanh), # activation funtion inside the layer
    BatchNorm(3),
    Dense(3 => 2),
    softmax) |> gpu # move model to GPU, if available

# The model encapsulates parameters, ramdomly initialised. Its initial output is:
out1 = model(noisy |> gpu) |> cpu

# To train the model, we use batches of 64 samples, and one-hot encoding:
target = Flux.onehotbatch(truth, [true, false])
loader = Flux.DataLoader((noisy, target) |> gpu, batchsize=64, shuffle=true);

optim = Flux.setup(Flux.Adam(0.01), model)

# Training loop, using the whole dataset 1000 times:
# losses = []
# @showprogress for epoch in 1:1_000
#     for (x, y) in loader
#         loss, grad = Flux.withgradient(model) do m
#             # Evaluate model and loss inside gradient context
