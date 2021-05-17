from training.data import DataImporter, SequenceClassificationDataModule
from sentence_transformers import SentenceTransformer
from model import SentenceClassificationModule
from training.trainer import Trainer
from torch.optim import Adam
from torch.nn import BCELoss
from matplotlib import pyplot as plt
import numpy as np

path_to_parameters = "parameters.json"

data_importer = DataImporter(path_to_parameters)

model = SentenceClassificationModule(512, 64, dropout=0.25)

sentence_encoder = SentenceTransformer('distiluse-base-multilingual-cased-v2')

sentence_encoder = sentence_encoder.cuda()

data_module = SequenceClassificationDataModule(data_importer, sentence_encoder)

optimizer = Adam(model.parameters(), lr=0.001, amsgrad=True)

train_data_loader = data_module.get_train_data_loader()

test_data_loader = data_module.get_test_data_loader()

loss_function = BCELoss()

trainer = Trainer(model, loss_function, optimizer, train_data_loader, test_data_loader, sentence_encoder)

# trainer.restore_checkpoint(100)  # .25 88% .28

losses = trainer.fit(10)

plt.plot(np.arange(len(losses[0])), losses[0], label="train loss")

plt.plot(np.arange(len(losses[1])), losses[1], label="validation loss")

plt.yscale("log")

plt.legend()

plt.show()