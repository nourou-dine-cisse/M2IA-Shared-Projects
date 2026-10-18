import re
import time
import argparse
from tqdm import tqdm
from datasets import Dataset
from sklearn.metrics import classification_report
from seqeval.metrics import precision_score, recall_score, f1_score, accuracy_score
from transformers import AutoTokenizer, AutoModelForTokenClassification, DataCollatorForTokenClassification, Trainer, TrainingArguments





parser = argparse.ArgumentParser(description='Finetuning BERT kind models for multi-class classification')
parser.add_argument('--model', type=str, help='Huggingface BERT model to be called', default="almanach/camembert-bio-base")
parser.add_argument('--epochs',  type=int, help='Numbers of epochs (default 5)', default=5)
parser.add_argument('--max_len',  type=int, help='maximum text length (default 128)', default=128)
parser.add_argument('--batch',  type=int, help='batch size (default 8)', default=8)
parser.add_argument('--lr',  type=float, help='learning rate (default 1e-05)', default=1e-05)

args = parser.parse_args()


bert_model = args.model
TRAIN_BATCH_SIZE = args.batch
VALID_BATCH_SIZE = 4
EPOCHS = args.epochs
LEARNING_RATE = args.lr

# Data paths
MED_TRAIN_DATA_PATH = "./TP_ISD2020/QUAERO_FrenchMed/MEDLINE/MEDLINEtrain_layer1_ID.conll"
MED_VALID_DATA_PATH = "./TP_ISD2020/QUAERO_FrenchMed/MEDLINE/MEDLINEdev_layer1_ID.conll"
MED_TEST_DATA_PATH = "./TP_ISD2020/QUAERO_FrenchMed/MEDLINE/MEDLINEtest_layer1_ID.conll"

PRESS_TRAIN_DATA_PATH = "./TP_ISD2020/QUAERO_FrenchPress/fra4_ID.train"
PRESS_VALID_DATA_PATH = "./TP_ISD2020/QUAERO_FrenchPress/fra4_ID.dev"
PRESS_TEST_DATA_PATH = "./TP_ISD2020/QUAERO_FrenchPress/fra4_ID.test"

def tokens_labels(data_path):
    """Read a token-label file into aligned sentence-level lists."""
    with open(data_path, "r", encoding="utf-8") as f:
        data = f.readlines()

    pattern = re.compile(r"^\d+\s+(\S+)\s+\S+\s+\S+\s+(\S+)")
    tokens = []
    labels = []
    phrase = []
    phrase_labels = []

    for line in data:
        match = pattern.search(line)
        if match:
            phrase.append(match.group(1))
            phrase_labels.append(match.group(2))
        else:
            if phrase:
                tokens.append(phrase)
                labels.append(phrase_labels)
            phrase = []
            phrase_labels = []

    return tokens, labels


def prepare_ner_dataset(tokens, labels, model_name, label2id={}, id2label={}, train=True):
    """Build and tokenize a Hugging Face dataset for token-level NER."""

    # Check sentence alignment
    assert len(tokens) == len(labels), (
        f"Number of sentences differs: "
        f"{len(tokens)} tokens vs {len(labels)} labels"
    )

    # Check token/label alignment
    for i, (sentence_tokens, sentence_labels) in enumerate(
        zip(tokens, labels)
    ):
        assert len(sentence_tokens) == len(sentence_labels), (
            f"Sentence {i}: "
            f"{len(sentence_tokens)} tokens vs {len(sentence_labels)} labels"
        )

    # Create label mappings only for the training dataset
    if train:
        label_names = sorted(
            {
                label
                for sentence_labels in labels
                for label in sentence_labels
            }
        )

        label2id = {
            label: i
            for i, label in enumerate(label_names)
        }

        id2label = {
            i: label
            for label, i in label2id.items()
        }

    # Create Hugging Face dataset
    dataset = Dataset.from_list([
        {
            "tokens": sentence_tokens,
            "labels": sentence_labels
        }
        for sentence_tokens, sentence_labels
        in zip(tokens, labels)
    ])

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Tokenize and align NER labels with subtokens
    def tokenize_and_align_labels(examples):
        encoding = tokenizer(
            examples["tokens"],
            is_split_into_words=True,
            truncation=True
        )

        aligned_labels = []

        for batch_index in range(len(examples["tokens"])):
            word_ids = encoding.word_ids(
                batch_index=batch_index
            )

            sentence_labels = examples["labels"][batch_index]
            label_ids = []
            previous_word_id = None

            for word_id in word_ids:

                # Special tokens
                if word_id is None:
                    label_ids.append(-100)

                # First subtoken of a word
                elif word_id != previous_word_id:
                    label_ids.append(
                        label2id[sentence_labels[word_id]]
                    )

                # Other subtokens
                else:
                    label_ids.append(-100)

                previous_word_id = word_id

            aligned_labels.append(label_ids)

        encoding["labels"] = aligned_labels
        return encoding

    # Apply tokenization to the dataset
    dataset = dataset.map(
        tokenize_and_align_labels,
        batched=True
    )

    return dataset.remove_columns("tokens"), label2id, id2label, tokenizer

# Prepare datasets and tokenizers for both MEDLINE and Press corpora
med_train, med_label2id, med_id2label, med_tokenizer = prepare_ner_dataset(
    *tokens_labels(MED_TRAIN_DATA_PATH),
    model_name=bert_model
)
med_test, _, _, _ = prepare_ner_dataset(
    *tokens_labels(MED_TEST_DATA_PATH),
    model_name=bert_model, label2id=med_label2id, id2label=med_id2label, train=False
)

med_valid, _, _, _ = prepare_ner_dataset(
	*tokens_labels(MED_VALID_DATA_PATH),
	model_name=bert_model, label2id=med_label2id, id2label=med_id2label, train=False
)

press_train, press_label2id, press_id2label, press_tokenizer = prepare_ner_dataset(
    *tokens_labels(PRESS_TRAIN_DATA_PATH),
    model_name=bert_model
)
press_test, _, _, _ = prepare_ner_dataset(
    *tokens_labels(PRESS_TEST_DATA_PATH),
    model_name=bert_model, label2id=press_label2id, id2label=press_id2label, train=False
)

press_valid, _, _, _ = prepare_ner_dataset(
	*tokens_labels(PRESS_VALID_DATA_PATH),
	model_name=bert_model, label2id=press_label2id, id2label=press_id2label, train=False
)

#  NER training for medical data

# Medical data collator and medical model
med_data_collator = DataCollatorForTokenClassification(tokenizer=med_tokenizer)
med_model =AutoModelForTokenClassification.from_pretrained(bert_model, num_labels=len(med_label2id), id2label=med_id2label, label2id=med_label2id)

# Press data collator and press model
press_data_collator = DataCollatorForTokenClassification(tokenizer=press_tokenizer)
press_model = AutoModelForTokenClassification.from_pretrained(bert_model, num_labels=len(press_label2id), id2label=press_id2label, label2id=press_label2id)


def make_compute_metrics(id2label):

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = predictions.argmax(axis=-1)

        true_predictions = []
        true_labels = []

        for prediction, label in zip(predictions, labels):
            sentence_predictions = []
            sentence_labels = []

            for pred, true in zip(prediction, label):
                if true != -100:
                    sentence_predictions.append(id2label[pred])
                    sentence_labels.append(id2label[true])

            true_predictions.append(sentence_predictions)
            true_labels.append(sentence_labels)

        return {
            "precision": precision_score(
                true_labels,
                true_predictions
            ),
            "recall": recall_score(
                true_labels,
                true_predictions
            ),
            "f1": f1_score(
                true_labels,
                true_predictions
            ),
            "accuracy": accuracy_score(
                true_labels,
                true_predictions
            ),
        }

    return compute_metrics

def train_ner_model(model, train_dataset, valid_dataset, data_collator, epochs=EPOCHS, batch_size=TRAIN_BATCH_SIZE):
	"""Train a Hugging Face NER model."""
	# Training configuration
	training_args = TrainingArguments(
        output_dir="./ner_results",
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        report_to="none"
    )

    # Hugging Face training loop
	trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=data_collator,
        compute_metrics=make_compute_metrics(model.config.id2label)
    )

    # Start training
	trainer.train()

	return trainer

print("Training NER model on medical data...")

med_trainer = train_ner_model(
    med_model,
    med_train,
    med_valid,
    med_data_collator
)

med_trainer.save_model("models/med_model")

print("Training NER model on press data...")

press_model.to("mps")
press_trainer = train_ner_model(
	press_model,
	press_train,
	press_valid,
	press_data_collator
)

press_trainer.save_model("models/press_model")


### Evaluation of the medical model on the test set
print("Evaluation of the medical model on the test set...")

med_model = AutoModelForTokenClassification.from_pretrained(
    "models/med_model"
)

med_model.eval()

med_trainer = Trainer(
    model=med_model,
    eval_dataset=med_test,
    data_collator=med_data_collator,
    compute_metrics=make_compute_metrics(med_id2label)
)

metrics = med_trainer.evaluate()

print(f"Metrics of the medical model on {len(med_test)} test data points: {metrics}")

print("Evaluation of the press model on the test set...")

### Evaluation of the press model on the test set

press_model = AutoModelForTokenClassification.from_pretrained(
    "models/press_model"
)

press_model.eval()

press_trainer = Trainer(
    model=press_model,
    eval_dataset=press_test,
    data_collator=press_data_collator,
    compute_metrics=make_compute_metrics(press_id2label)
)

metrics = press_trainer.evaluate()

print(f"Metrics of the press model on {len(press_test)} test data points: {metrics}")
