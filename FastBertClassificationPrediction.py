from pathlib import Path
from venv import logger
from fast_bert.prediction import BertClassificationPredictor
from fast_bert.data import BertDataBunch, BertTokenizer, BertConfig
from fast_bert.learner_cls import BertLearner
from fast_bert.metrics import accuracy
from tokenizers import BertWordPieceTokenizer
import os


#initial Configuration / Hyper parameters
from torch.cuda import device


DATA_PATH = Path('../data/')     # path for data files (train and val)
LABEL_PATH = Path('../labels/')  # path for labels file
MODEL_PATH=Path('../models/')    # path for model artifacts to be stored
LOG_PATH=Path('../logs/')       # path for log files to be stored
# location for the pretrained BERT models
BERT_PRETRAINED_PATH = Path('../../bert_models/pretrained-weights/uncased_L-12_H-768_A-12/')

#BertConfig
# BERTCONFIG = BertConfig()
# print(BERTCONFIG)
#Notice that the BertConfig is also predefined.
BERT_PARAMETERS = {
    "max_seq_length": 512,
    "do_lower_case": True,
    "train_batch_size": 32,
    "learning_rate": 6e-5,
    "num_train_epochs": 12.0,
    "warmup_proportion": 0.002,
    "local_rank": -1,
    "gradient_accumulation_steps": 1,
    "fp16": True,
    "loss_scale": 128
}
"""
     I recommend to either use a different path for the tokenizers and the model or to keep the 
     config.json of your model because some modifications you apply to your model will be stored in the config.json which is created during model.save_pretrained() and will be overwritten when you save the tokenizer as described above after
     your model (i.e. you won't be able to load your modified model with tokenizer config.json).
"""
tokenizer = BertTokenizer.from_pretrained(BERT_PRETRAINED_PATH, do_lower_case=BERT_PARAMETERS['do_lower_case'])
# GPU & Device

# Training a BERT model does require a single or more preferably multiple GPUs. In this step we can setup GPU parameters for our training.

devices = torch.device('cuda')

# check if multiple GPUs are available
if torch.cuda.device_count() > 1:
    multi_gpu = True
else:
    multi_gpu = False

# BertDataBunch

# This is an excellent idea borrowed from fast.ai library. The databunch object takes training,
# validation and test csv files and converts the data into internal representation for BERT.
# The object also instantiates the correct data-loaders based on device profile and batchsize and maxsequence_length.


bertdatabunch = BertDataBunch(DATA_PATH, LABEL_PATH, tokenizer,
                          train_file='train.csv', val_file='val.csv', label_file='labels.csv',
                          bs=BERT_PARAMETERS['train_batch_size'], maxlen=BERT_PARAMETERS['max_seq_length'],
                          multi_gpu=multi_gpu, multi_label=False)

# BertLearner
metrics = []
metrics.append({'name': 'accuracy', 'function': accuracy})

# BertLearner
learner = BertLearner.from_pretrained_model(bertdatabunch, BERT_PRETRAINED_PATH, metrics, device, logger,                                             finetuned_wgts_path=None,
                                            is_fp16=BERT_PARAMETERS['fp16'], loss_scale=BERT_PARAMETERS['loss_scale'],
                                            multi_gpu=multi_gpu,  multi_label=False)

# Train Model
learner.fit(6, lr=BERT_PARAMETERS['learning_rate'],schedule_type="warmup_cosine_hard_restarts")

data = "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json"
learner.predict_batch(data)

predictor = BertClassificationPredictor(model_path=MODEL_PATH, pretrained_path=BERT_PRETRAINED_PATH,
                                        label_path=LABEL_PATH, multi_label=False)

# Single prediction
single_prediction = predictor.predict("The Quick Brown fox jumps over the Lazy Dog")

multiple_predictions = predictor.predict(data)

