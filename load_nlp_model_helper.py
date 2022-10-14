import torch
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
import os
import numpy as np
import pandas as pd

class InputExample(object):
    def __init__(self, guid, text_a, text_b=None, label=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        if isinstance(label, list):
            self.label = label
        elif label:
            self.label = str(label)
        else:
            self.label = None
            
class InputFeatures(object):
    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        
        
def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def convert_examples_to_features(
    examples,
    label_list,
    max_seq_length,
    tokenizer,
    output_mode="classification",
    cls_token_at_end=False,
    pad_on_left=False,
    cls_token="[CLS]",
    sep_token="[SEP]",
    pad_token=0,
    sequence_a_segment_id=0,
    sequence_b_segment_id=1,
    cls_token_segment_id=1,
    pad_token_segment_id=0,
    mask_padding_with_zero=True,
    logger=None,
):
    label_map = {label: i for i, label in enumerate(label_list)}
    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(str(example.text_a))
        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(str(example.text_b))
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[: (max_seq_length - 2)]
        tokens = tokens_a + [sep_token]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if tokens_b:
            tokens += tokens_b + [sep_token]
            segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)

        if cls_token_at_end:
            tokens = tokens + [cls_token]
            segment_ids = segment_ids + [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = (
                [0 if mask_padding_with_zero else 1] * padding_length
            ) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            input_mask = input_mask + (
                [0 if mask_padding_with_zero else 1] * padding_length
            )
            segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if isinstance(example.label, list):
            label_id = []
            for label in example.label:
                label_id.append(float(label))
        else:
            if example.label is not None:
                label_id = label_map[example.label]
            else:
                label_id = ""

        features.append(
            InputFeatures(
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                label_id=label_id,
            )
        )
    return features


def get_dataset_from_examples(tokenizer, examples, labels, set_type="test", is_test=False, no_cache=False
):
    features = convert_examples_to_features(
        examples,
        label_list=labels,
        max_seq_length=512,
        tokenizer=tokenizer,
        output_mode="classification",
        cls_token_at_end=False,
        cls_token=tokenizer.cls_token,
        sep_token=tokenizer.sep_token,
        cls_token_segment_id=0,
        pad_on_left=False,
        pad_token_segment_id=0,
    )
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor(
        [f.input_mask for f in features], dtype=torch.long
    )
    all_segment_ids = torch.tensor(
        [f.segment_ids for f in features], dtype=torch.long
    )
    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids)
    return dataset

def get_dl_from_texts(tokenizer, texts, label_dir, batch_size=None):
    test_examples = []
    input_data = []
    for index, text in enumerate(texts):
        test_examples.append(InputExample(index, text, label=None))
        input_data.append({"id": index, "text": text})
    test_dataset = get_dataset_from_examples(
        tokenizer, test_examples, label_dir, "test", is_test=True, no_cache=True
    )
    test_sampler = SequentialSampler(test_dataset)
    if batch_size is None:
        batch_size = 16
    return DataLoader(
        test_dataset, sampler=test_sampler, batch_size=batch_size
    )

def predict_batch(model, tokenizer, text, labels, batch_size=None):
    dl = get_dl_from_texts(tokenizer, text, labels, batch_size)
    all_logits = None
    model.eval()
    for step, batch in enumerate(dl):
        batch = tuple(t.to("cpu") for t in batch)
        inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": None}
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs[0]
            if False:
                logits = logits.sigmoid()
            else:
                logits = logits.softmax(dim=1)

        if all_logits is None:
            all_logits = logits.detach().cpu().numpy()
        else:
            all_logits = np.concatenate(
                (all_logits, logits.detach().cpu().numpy()), axis=0
            )
    result_df = pd.DataFrame(all_logits, columns=labels)
    results = result_df.to_dict("record")
    results = [sorted(x.items(), key=lambda kv: kv[1], reverse=True) for x in results]
    prediction = []
    for i, o in enumerate(results):
        r = dict(o)
        r["text"] = text[i]
        prediction.append(r)
    return prediction
