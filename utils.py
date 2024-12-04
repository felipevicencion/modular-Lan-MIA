def recall_process_prefix(max_length, prefix, avg_length, pass_window, total_shots, tokenizer):
    if pass_window:
        return prefix
    token_counts = [len(tokenizer.encode(shot)) for shot in prefix]
    target_token_count = avg_length
    total_tokens = sum(token_counts) + target_token_count
    if total_tokens <= max_length:
        return prefix
    # Determine the maximum number of shots that can fit within the max_length
    max_shots = 0
    cumulative_tokens = target_token_count
    for count in token_counts:
        if cumulative_tokens + count <= max_length:
            max_shots += 1
            cumulative_tokens += count
        else:
            break
    # Truncate the prefix to include only the maximum number of shots
    truncated_prefix = prefix[-max_shots:]
    total_shots = max_shots
    return truncated_prefix

def recall_get_all_prob(input_ids, loss, logits):
    probabilities = torch.nn.functional.log_softmax(logits, dim=-1)
    all_prob = []
    input_ids_processed = input_ids[0][1:]
    for i, token_id in enumerate(input_ids_processed):
        probability = probabilities[0, i, token_id].item()
        all_prob.append(probability)
    ll = -loss.item()  # log-likelihood
    ppl = torch.exp(loss).item()
    prob = torch.exp(-loss).item()
    return prob, ll , ppl, all_prob, loss.item()

def recall_get_ll(sentence, model, tokenizer, device):
    print(sentence)
    print(tokenizer.encode(sentence))
    input_ids = torch.tensor(tokenizer.encode(sentence)).unsqueeze(0)
    input_ids = input_ids.to(device)
    print(input_ids)
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    loss, logits = outputs[:2]
    return recall_get_all_prob(input_ids, loss, logits)

def recall_get_conditional_ll(input_text, target_text, model, tokenizer, device):
    input_encodings = tokenizer(input_text, return_tensors="pt")
    # target_encodings = tokenizer(target_text, return_tensors="pt")
    concat_ids = torch.cat((input_encodings.input_ids.to(device), target_encodings.input_ids.to(device)), dim=1)
    labels = concat_ids.clone()
    labels[:, : input_encodings.input_ids.size(1)] = -100
    with torch.no_grad():
        outputs = model(concat_ids, labels=labels)
    loss, logits = outputs[:2]
    return recall_get_all_prob(labels, loss, logits)


def tokenize_data(tokenizer, train_data, test_data):
    tokenize_data = lambda batch: tokenizer(batch['text'], padding=True, truncation=True)
    train_data = train_data.map(tokenize_data, batched=True, batch_size=len(train_data))
    test_data = test_data.map(tokenize_data, batched=True, batch_size=len(test_data))
    train_data.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
    test_data.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
    return train_data, test_data