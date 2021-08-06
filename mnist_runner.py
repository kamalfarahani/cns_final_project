import os
import torch
import numpy as np

from torchvision import transforms
from tqdm import tqdm

from bindsnet.datasets import MNIST
from bindsnet.network.monitors import Monitor
from bindsnet.encoding import PoissonEncoder
from bindsnet.evaluation import all_activity, proportion_weighting, assign_labels

from model import DigitNet

seed = 0
n_neurons = 100
n_epochs = 1
n_test = 10000
n_train = 60000
n_workers = -1
exc = 22.5
inh = 120
theta_plus = 0.05
time = 250
dt = 1.0
intensity = 128
progress_interval = 10
update_interval = 250
train = 'store_true'
plot = 'store_true'
gpu = 'store_true'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if gpu and torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
else:
    torch.manual_seed(seed)
    device = "cpu"
    if gpu:
        gpu = False

torch.set_num_threads(os.cpu_count() - 1)
print("Running on Device = ", device)

# Determines number of workers to use
if n_workers == -1:
    n_workers = 0  # gpu * 4 * torch.cuda.device_count()


n_sqrt = int(np.ceil(np.sqrt(n_neurons)))
start_intensity = intensity

network = DigitNet(
    n_inpt=784,
    n_neurons=n_neurons,
    exc=exc,
    inh=inh,
    dt=dt,
    norm=78.4,
    theta_plus=theta_plus,
    inpt_shape=(1, 28, 28),
)

if gpu:
    network.to("cuda")

train_dataset = MNIST(
    PoissonEncoder(time=time, dt=dt),
    None,
    root='./data',
    download=True,
    train=True,
    transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Lambda(lambda x: x * intensity)]
    ),
)

# Record spikes during the simulation.
spike_record = torch.zeros((update_interval, int(time / dt), n_neurons), device=device)

# Neuron assignments
n_classes = 10
assignments = -torch.ones(n_neurons, device=device)

# Sequence of accuracy estimates.
accuracy = { "all": [] }

spikes = {}
for layer in set(network.layers):
    spikes[layer] = Monitor(
        network.layers[layer], state_vars=["s"], time=int(time / dt), device=device
    )
    network.add_monitor(spikes[layer], name="{}_spikes".format(layer))

print("Start training...\n")
for epoch in range(n_epochs):
    labels = []

    # Create a dataloader to iterate and batch data
    dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1, shuffle=True, num_workers=n_workers, pin_memory=gpu
    )

    for step, batch in enumerate(tqdm(dataloader)):
        if step > n_train:
            break
        # Get next input sample.
        inputs = {"X": batch["encoded_image"].view(int(time / dt), 1, 1, 28, 28)}
        if gpu:
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        if step % update_interval == 0 and step > 0:
            # Convert the array of labels into a tensor
            label_tensor = torch.tensor(labels, device=device)

            # Get network predictions.
            all_activity_pred = all_activity(
                spikes=spike_record, assignments=assignments, n_labels=n_classes
            )

            # Compute network accuracy according to available classification strategies.
            accuracy["all"].append(
                100
                * torch.sum(label_tensor.long() == all_activity_pred).item()
                / len(label_tensor)
            )

            print(
                "All activity accuracy: {} (last), {} (average), {} (best)\n".format(
                    accuracy["all"][-1],
                    np.mean(accuracy["all"]),
                    np.max(accuracy["all"]),
                )
            )

            # Assign labels to excitatory layer neurons.
            assignments, _, _ = assign_labels(
                spikes=spike_record,
                labels=label_tensor,
                n_labels=n_classes,
            )

            labels = []
        
        labels.append(batch["label"])

        # Run the network on the input.
        network.run(inputs=inputs, time=time, input_time_dim=1)

        # Add to spikes recording.
        spike_record[step % update_interval] = spikes["Ae"].get("s").squeeze()

        # Reset state variables.
        network.reset_state_variables()

print("Training completed.\n")

# Test Phase

test_dataset = MNIST(
    PoissonEncoder(time=time, dt=dt),
    None,
    root='./data',
    download=True,
    train=False,
    transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Lambda(lambda x: x * intensity)]
    ),
)

accuracy = { "all": 0 }

spike_record = torch.zeros((1, int(time / dt), n_neurons), device=device)

network.train(mode=False)

pbar = tqdm(total=n_test)
for step, batch in enumerate(test_dataset):
    if step >= n_test:
        break
    # Get next input sample.
    inputs = {"X": batch["encoded_image"].view(int(time / dt), 1, 1, 28, 28)}
    if gpu:
        inputs = {k: v.cuda() for k, v in inputs.items()}

    # Run the network on the input.
    network.run(inputs=inputs, time=time, input_time_dim=1)

    # Add to spikes recording.
    spike_record[0] = spikes["Ae"].get("s").squeeze()

    # Convert the array of labels into a tensor
    label_tensor = torch.tensor(batch["label"], device=device)

    # Get network predictions.
    all_activity_pred = all_activity(
        spikes=spike_record, assignments=assignments, n_labels=n_classes
    )

    # Compute network accuracy according to available classification strategies.
    accuracy["all"] += float(torch.sum(label_tensor.long() == all_activity_pred).item())

    network.reset_state_variables()
    pbar.set_description_str("Test progress: ")
    pbar.update()

print("All activity accuracy: {}".format((accuracy["all"] / n_test)))

print("Testing complete.\n")
