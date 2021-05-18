from training.data import DataImporter, SequenceClassificationDataModule
from sentence_transformers import SentenceTransformer
from model import SentenceClassificationModule
from training.trainer import Trainer
from torch.optim import Adam
from torch.nn import BCELoss
from matplotlib import pyplot as plt
import numpy as np

path_to_parameters = "parameters.json"

parameters = DataImporter.load_parameters(path_to_parameters)

data_importer = DataImporter(parameters)

model = SentenceClassificationModule(parameters["embedding_size"], 64, dropout=parameters["dropout"])

sentence_encoder = SentenceTransformer(parameters["sentence_transformer"])

if parameters["cuda"]:
    sentence_encoder = sentence_encoder.cuda()

data_module = SequenceClassificationDataModule(data_importer, sentence_encoder,
                                               batch_size=parameters["batch_size"],
                                               num_workers=parameters["num_workers"])

optimizer = Adam(model.parameters(), lr=parameters["learning_rate"], amsgrad=True,
                 weight_decay=parameters["weight_decay"])

train_data_loader = data_module.get_train_data_loader()

test_data_loader = data_module.get_test_data_loader()

loss_function = BCELoss()

trainer = Trainer(model, loss_function, optimizer, train_data_loader, test_data_loader, parameters["cuda"])

# trainer.restore_checkpoint(100)  # .25 88% .28

losses = trainer.fit(parameters["num_iterations"])

plt.plot(np.arange(len(losses[0])), losses[0], label="train loss")

plt.plot(np.arange(len(losses[1])), losses[1], label="validation loss")

plt.yscale("log")

plt.legend()

plt.show()
