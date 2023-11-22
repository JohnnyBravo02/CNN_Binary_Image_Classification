# Dependencies
using CUDA
using Flux, JLD2
using FileIO, DelimitedFiles, Images, ImageView
using Plots

# Configuration
CUDA.allowscalar(false)
newModel = false;
saveModel = true;
metrics = true;
useGpu = false;
clearVariables = true;

# Import Data
imgDimension = (146, 146);
channels = 3;

## Dogs
dogs_numImages = 2500;
dogs_rawData = Float32.(zeros(imgDimension[1], imgDimension[2], channels, dogs_numImages));
for element in 1:dogs_numImages
    cImg = load("data/dogs/$(element).jpg");
    for channel in 1:channels
        for xp in 1:imgDimension[1]
            for yp in 1:imgDimension[2]
                if channel == 1
                    dogs_rawData[xp, yp, channel, element] = Float32.(red(cImg[xp, yp]));
                elseif channel == 2
                    dogs_rawData[xp, yp, channel, element] = Float32.(green(cImg[xp, yp]));
                else
                    dogs_rawData[xp, yp, channel, element] = Float32.(blue(cImg[xp, yp]));
                end
            end
        end
    end
    println("Loaded image $(element)");
end

## Cats
cats_numImages = 2500;
cats_rawData = Float32.(zeros(imgDimension[1], imgDimension[2], channels, cats_numImages));
for element in 1:cats_numImages
    cImg = load("data/cats/$(element).jpg");
    for channel in 1:channels
        for xp in 1:imgDimension[1]
            for yp in 1:imgDimension[2]
                if channel == 1
                    cats_rawData[xp, yp, channel, element] = Float32.(red(cImg[xp, yp]));
                elseif channel == 2
                    cats_rawData[xp, yp, channel, element] = Float32.(green(cImg[xp, yp]));
                else
                    cats_rawData[xp, yp, channel, element] = Float32.(blue(cImg[xp, yp]));
                end
            end
        end
    end
    println("Loaded image $(element)");
end

# Complete Raw Data
rawData = Float32.(zeros(imgDimension[1], imgDimension[2], channels, dogs_numImages + cats_numImages));
rawData[:, :, :, 1:dogs_numImages] = dogs_rawData;
rawData[:, :, :, dogs_numImages+1:dogs_numImages+cats_numImages] = cats_rawData;

# Data Normalization
rawData = Flux.normalise(rawData);

# labels
dog_labels = ones(dogs_numImages, 1);
cat_labels = zeros(cats_numImages, 1);
labels = vcat(dog_labels, cat_labels);

# Data Formatting
train, test, val = Flux.MLUtils.splitobs((rawData, labels[:, 1]'), at=(0.6, 0.2), shuffle=true);

# Model Properties
classes = 1;
convFilter = (3, 3);

# CNN Model
model = Chain(
        Conv(convFilter, 3=>16, relu), # 1st Conv Layer
        MaxPool((2, 2)),
        Conv(convFilter, 16=>32, relu), # 2nd Conv Layer
        MaxPool((2, 2)),
        Conv(convFilter, 32=>64, relu), # 3rd Conv Layer
        MaxPool((2, 2)),
        x -> reshape(x, :, size(x, 4)),
        Dense(16384=>5250, relu, init=Flux.glorot_uniform),
        Dense(5250=>classes, sigmoid, init=Flux.glorot_uniform)
    );
if !newModel
    Flux.loadmodel!(model, JLD2.load("Model.jld2")["model"])
end
if useGpu
    model = model |> gpu;
end

# Hyperparameters
epochs = 10;
α = 0.01;
ψ = 0.0001;

# Loss
loss(x, y) = Flux.binarycrossentropy(model(x), y);

# Optimizer
opt = Momentum(α, ψ);

# Metrics
if metrics
    lossLog = zeros(epochs, 2);
    accuracyLog = zeros(epochs, 2);
end

# Batching
batchSize = 64;
batches = Flux.DataLoader((train[1], train[2]), batchsize = batchSize, shuffle = true);

# Training Setup
if useGpu
    train = (train[1] |> gpu, train[2] |> gpu);
    val = (val[1] |> gpu, val[2] |> gpu);
    test = (test[1] |> gpu, test[2] |> gpu);
end

trainData = Flux.MLUtils.getobs(train[1]);
trainLabels = Flux.MLUtils.getobs(train[2]);
formattedData = [(trainData, trainLabels)];
valData = Flux.MLUtils.getobs(val[1]);
valLabels = Flux.MLUtils.getobs(val[2]);

# Clear Variables
if clearVariables
    rawData = nothing;
    dogs_rawData = nothing;
    cats_rawData = nothing;
    dog_labels = nothing;
    cat_labels = nothing;
    labels = nothing;
    GC.gc();
end

# Training Loop
for epoch in 1:epochs
    for miniBatch in batches
        if useGpu
            miniBatch = (miniBatch[1] |> gpu, miniBatch[2] |> gpu);
        end
        Flux.train!(loss, Flux.params(model), [miniBatch], opt);
    end
    
    if (metrics)
        lossLog[epoch, 1] = loss(trainData, trainLabels);
        lossLog[epoch, 2] = loss(valData, valLabels);
        ŷ_train = round.(model(trainData));
        accuracyLog[epoch, 1] = sum(ŷ_train .== trainLabels) / length(trainLabels) * 100;
        ŷ_val = round.(model(valData));
        accuracyLog[epoch, 2] = sum(ŷ_val .== valLabels) / length(valLabels) * 100;
        
        if epoch % 1 == 0
            println("Epoch: ", epoch, " | Training Loss: ", lossLog[epoch, 1], " | Training Accuracy: ", accuracyLog[epoch, 1], "%", " | Validation Loss: ", lossLog[epoch, 2], " | Validation Accuracy: ", accuracyLog[epoch, 2], "%");
        end
    else
        if epoch % 1 == 0
            println("Epoch: ", epoch);
        end
    end
end

# Testing
testData = Flux.MLUtils.getobs(test[1]);
testLabels = Flux.MLUtils.getobs(test[2]);
ŷ = round.(model(testData));
testAccuracy = sum(ŷ .== testLabels) / length(testLabels) * 100;
println("Test Accuracy: ", testAccuracy, "%");

# Save Model
if saveModel
    JLD2.jldsave("Model.jld2"; model);
    println("Model Saved");
end

# Loss Visualization
if metrics
    plot(1:epochs, lossLog[:, 1], label="Training Loss", xlabel="Epochs", ylabel="Loss", title="Loss");
    display(plot!(1:epochs, lossLog[:, 2], label="Validation Loss", xlabel="Epochs", ylabel="Loss"));
    plot(1:epochs, accuracyLog[:, 1], label="Training Accuracy", xlabel="Epochs", ylabel="Accuracy", title="Accuracy");
    display(plot!(1:epochs, accuracyLog[:, 2], label="Validation Accuracy", xlabel="Epochs", ylabel="Accuracy"));
end