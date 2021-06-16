import numpy as np


def data_preprocessing(rdr, code):
    texts = []

    labels = []

    for idx, line in enumerate(rdr):
        if idx == 0:
            continue
        text = ""
        label = [0 for _ in range(len(code))]
        for i in line[2].split('\n'):
            text += i + ' '
        for i in line[-1].split(';'):
            if i in code:
                label[code.index(i)] = 1
        text = text.split()

        if len(text) > 2500:
            text = text[:2500]

        etc = []
        etc2 = []

        for i in range(len(text)):
            etc.append(text[i])
            if len(etc) % 100 == 0:
                etc2.append(' '.join(etc))
                etc = []
        if len(etc) % 100 != 0:
            etc2.append(' '.join(etc))
        if len(etc2) != 25:
            for i in range(25 - len(etc2)):
                etc2.append(' ')

        texts.append(etc2)
        labels.append(label)

    return texts, labels


def bert_tokenizer(tokenizer, sent, max_len):
    encoded_dict = tokenizer.encode_plus(text=sent, add_special_tokens=True, max_length=max_len,
                                         pad_to_max_length=True, return_attention_mask=True)
    input_id = encoded_dict['input_ids']
    attention_mask = encoded_dict['attention_mask']
    token_type_id = encoded_dict['token_type_ids']

    return input_id, attention_mask, token_type_id


def model_input(X, y, tokenizer, max_len):
    X_input = []
    X_attention = []
    X_token_type = []

    X_data_labels = []

    for sent, label in zip(X, y):
        input_ = []
        attention_ = []
        token_type = []

        for X_sent in sent:
            input_id, attention_mask, token_type_id = bert_tokenizer(tokenizer, X_sent, max_len)
            input_.append(input_id)
            attention_.append(attention_mask)
            token_type.append(token_type_id)

        X_input.append(input_)
        X_attention.append(attention_)
        X_token_type.append(token_type)

        X_data_labels.append(label)

    X_data_labels = np.asarray(X_data_labels, dtype=np.int32)

    X_input = np.array(X_input, dtype=int)
    X_input = np.reshape(X_input, (X_input.shape[0], X_input.shape[1] * X_input.shape[2]))

    X_attention = np.array(X_attention, dtype=int)
    X_attention = np.reshape(X_attention, (X_attention.shape[0], X_attention.shape[1] * X_attention.shape[2]))

    X_token_type = np.array(X_token_type, dtype=int)
    X_token_type = np.reshape(X_token_type, (X_token_type.shape[0], X_token_type.shape[1] * X_token_type.shape[2]))

    X_inputs = (X_input, X_attention, X_token_type)

    return X_inputs, X_data_labels
