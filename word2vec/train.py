import torch.optim as optim
import torch.nn as nn

def train_w2c_model(model, 
                    inputs, 
                    train_outputs,
                    epochs=10,
                    optimizer=optim.Adam,
                    lr=0.01,
                    loss=nn.CrossEntropyLoss(),
                    ):
    
    optimizer = optimizer(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss_value = loss(outputs, train_outputs)
        
        loss_value.backward()
        optimizer.step()

        print (f"Epoch {epoch+1}/{epochs}, Loss: {loss_value.item():.4f}")

    return model