from copy import deepcopy
import torch
import numpy as np

def splitBertInputSequences (tokens, tokenizer):
    joined_tokenized_and_cleaned_doc = []

    for doc in tokens:
        joined_tokenized_and_cleaned_doc.extend(doc)

    bert_sequences = []
    embeddings_length = []
    cls_token = tokenizer.tokenize("CLS")
    sep_token = tokenizer.tokenize("SEP")
    sequence = deepcopy(cls_token)
    total_token_len = 0

    for doc in joined_tokenized_and_cleaned_doc:
        token = tokenizer.tokenize(doc)
        sequence.extend(token)
        total_token_len += len(token)
        if len(sequence) > 511:
            new_sequence = deepcopy(sequence[256:])
            if len(sequence) > 511:
                sequence = sequence[:511]
            sequence += deepcopy(sep_token)
            bert_sequences.append(sequence)

            sequence = deepcopy(cls_token)
            sequence.extend(new_sequence)

    sequence += deepcopy(sep_token)
    bert_sequences.append(sequence)

    return bert_sequences

def bertTextPreparation(sequence, tokenizer):
    """Preparing the input for BERT
    
    Takes a string argument and performs
    pre-processing like adding special tokens,
    tokenization, tokens to ids, and tokens to
    segment ids. All tokens are mapped to seg-
    ment id = 1.
    
    Args:
        text (str): Text to be converted
        tokenizer (obj): Tokenizer object
            to convert text into BERT-re-
            adable tokens and ids
        
    Returns:
        list: List of BERT-readable tokens
        obj: Torch tensor with token ids
        obj: Torch tensor segment ids
    
    
    """
    indexed_tokens = tokenizer.convert_tokens_to_ids(sequence)
    segments_ids = [1]*len(indexed_tokens)

    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])

    return tokens_tensor, segments_tensors

def getBertEmbeddings(tokens_tensor, segments_tensors, model):
    """Get embeddings from an embedding model
    
    Args:
        tokens_tensor (obj): Torch tensor size [n_tokens]
            with token ids for each token in text
        segments_tensors (obj): Torch tensor size [n_tokens]
            with segment ids for each token in text
        model (obj): Embedding model to generate embeddings
            from token and segment ids
    
    Returns:
        list: List of list of floats of size
            [n_tokens, n_embedding_dimensions]
            containing embeddings for each token
    
    """
    
    # Gradient calculation id disabled
    # Model is in inference mode
    with torch.no_grad():
        outputs = model(tokens_tensor, segments_tensors)
        # Removing the first hidden state
        # The first state is the input state
        hidden_states = outputs[2][1:]
        token_embeddings = torch.stack(hidden_states, dim=0)
        token_embeddings = torch.squeeze(token_embeddings, dim=1)
        token_embeddings = token_embeddings.permute(1,0,2)
        print (token_embeddings.size())
        # Stores the token vectors, with shape [22 x 768]
        token_vecs_sum = []


        # For each token in the sentence...
        for token in token_embeddings:
              # `token` is a [12 x 768] tensor

                # Sum the vectors from the last four layers.
                sum_vec = torch.sum(token[-4:], dim=0)
                
                # Use `sum_vec` to represent `token`.
                token_vecs_sum.append(sum_vec)

        print ('Shape is: %d x %d' % (len(token_vecs_sum), len(token_vecs_sum[0])))

    return token_vecs_sum

def computeBertEmbeddings (bert_sequences, tokenizer, model):
    # Getting embeddings for the target
    # word in all given contexts
    word_embeddings = []

    for sequence in bert_sequences:
        tokens_tensor, segments_tensors = bertTextPreparation(sequence, tokenizer)
        list_token_embeddings = getBertEmbeddings(tokens_tensor, segments_tensors, model)
        word_embeddings.append(np.transpose(np.array(list_token_embeddings, dtype=object)))
    
    return word_embeddings
    

def combineBertOutput (word_embeddings):
    unified_embeddings = []
    unified_embeddings.extend(word_embeddings[0][2:257])


    for index, embedding in enumerate(word_embeddings):
        if index == len(word_embeddings) - 1:
            unified_embeddings.extend(embedding[256:len(embedding)-1])
            break
        combined = np.add(embedding[256:len(embedding)-1], word_embeddings[index+1][2:257])
        unified_embeddings.extend(combined)

    return unified_embeddings