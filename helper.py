import torch
from transformers import BertTokenizer, BertModel,BertForMaskedLM

def multiple_mask_tokens(input_text, n = 5):
    """
    :param input_text: string with MASK tokens
    :param n: the top number of tokens to return
    :return: list of n tokens for every mask token. Returns a blank list if no mask token is found
    """
    model = BertForMaskedLM.from_pretrained('bert-base-uncased')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    inputs = tokenizer(input_text, return_tensors='pt')
    outputs = model(**inputs)

    # predicitons is the probability distribution over the vocabulary for each token
    predictions = outputs[0]

    # get index of masked tokens
    masked_indices = (inputs['input_ids'] == tokenizer.mask_token_id).nonzero(as_tuple=True)

    if masked_indices[0].shape[0] == 0:
        print("No masked tokens found")
        return []

    # get the probability distribution over the vocabulary for each masked token
    masked_predictions = predictions[masked_indices]

    # get the top 5 predictions for each masked token
    top_n_values = torch.topk(masked_predictions, n, dim=1,sorted=True)
    top_n_probability = top_n_values.values
    top_n_token_numbers = top_n_values.indices

    # get the token words for the top n predictions
    answers = []
    for i in range(len(masked_indices[0]-1)):
        proabilities = []
        top_n_tokens = tokenizer.convert_ids_to_tokens(top_n_token_numbers[i])
        for j in range(len(top_n_tokens)):
            proabilities.append((top_n_tokens[j],top_n_probability[i][j].item()))
        answers.append(proabilities)
    return answers

print("Masked tokens in the sentence are:", multiple_mask_tokens("[MASK] [MASK] [MASK] of the US is public service"))