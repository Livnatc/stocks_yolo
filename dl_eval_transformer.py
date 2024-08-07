import torch
from dl_training_transformer import TransAm, PositionalEncoding, get_batch
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # Load the entire model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransAm().to(device)

    max_value = 81
    min_value = 35

    model = torch.load('model_C.pt')
    model.eval()  # Set the model to evaluation mode

    # Load the data
    test_data = torch.load('test_data_C.pt')

    # Predict the next price
    with torch.no_grad():
        data, targets = get_batch(test_data, 0, len(test_data))
        output = model(data)
        print(output)
        # Save the output
        torch.save(output, 'output.pt')

        # Mean Squared Error:
        criterion = torch.nn.MSELoss()
        loss = criterion(output, targets)
        print(f'Mean Squared Error: {loss.item()}')

        # Plot the output
        # ...
        for i in range(len(test_data) - 1):
            true_price = test_data[i].numpy()[1]
            predicted_price = output[:, i].numpy()
            mse = criterion(torch.tensor(true_price[-10:]), torch.tensor(predicted_price[-10:]))

            # get back the original prices:
            original_tensor_true = (true_price[-10:] + 1) / 2 * (max_value - min_value) + min_value
            original_tensor_pred = (predicted_price[-10:] + 1) / 2 * (max_value - min_value) + min_value

            plt.figure()
            plt.plot(original_tensor_true, label='True Price', color='blue')
            plt.plot(original_tensor_pred, label='Predicted Price', color='red')
            plt.legend()
            plt.savefig(f'test_graphs_C/output_{i}.png')
        # ...
        # Save the plot
        print('.')