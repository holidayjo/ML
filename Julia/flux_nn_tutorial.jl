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
    Dense(3 => 6, tanh),
    Dense(6 => 2),
    softmax) |> gpu # move model to GPU, if available

# The model encapsulates parameters, ramdomly initialised. Its initial output is:
out1 = model(noisy |> gpu) |> cpu

# To train the model, we use batches of 64 samples, and one-hot encoding:
target = Flux.onehotbatch(truth, [true, false])
loader = Flux.DataLoader((noisy, target) |> gpu, batchsize=64, shuffle=true);

optim = Flux.setup(Flux.Adam(0.01), model)

# Training loop, using the whole dataset 1000 times:
losses = []
@showprogress for epoch in 1:100
    for (x, y) in loader
        loss, grads = Flux.withgradient(model) do m
            # Evaluate model and loss inside gradient context
            y_hat = m(x)
            Flux.crossentropy(y_hat, y)
        end
        Flux.update!(optim, model, grads[1])
        push!(losses, loss)
    end
end

optim
out2 = model(noisy |> gpu) |> cpu # first row is prob of true, second row p(false)

mean((out2[1,:] .> 0.5) .==truth) # accuracy 97.4% so far!

using Plots # to draw the above figure

p_true = scatter(noisy[1,:], noisy[2,:], zcolor=truth, title="True classification", legend=false)
p_raw  = scatter(noisy[1,:], noisy[2,:], zcolor=out1[1,:], title="Untrained Network", label="", clim=(0,1))
p_raw  = scatter(noisy[1,:], noisy[2,:], zcolor=out1[1,:], title="Untrained Network", label="", clim=(0,1))
p_done = scatter(noisy[1,:], noisy[2,:], zcolor=out2[1,:], title="trained Network", legend=false)

plot(p_true, p_raw, p_done, layout=(1,3), size=(1000,330))


# Here is the loss during training:
plot(losses; xaxis=(:log10, "iteration"), yaxis="loss", label="per batch")
n = length(loader)
plot!(n:n:length(losses), mean.(Iterators.partition(losses, n)), label="epoch mean", dpi=200)
