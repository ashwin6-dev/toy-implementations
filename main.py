from word2vec import preprocess, train, model


text = """
The king and the queen ruled the kingdom. 
The man and the woman lived in the village. 
A boy played with his sister, and a girl walked with her brother. 
The cat chased the dog, and the dog barked at the cat. 
The city was full of people, while the village was small and quiet. 
The queen and the king visited the city, but the man and the woman stayed in the village. 
The boy and the girl fed the cat and the dog. 
The kingdom grew as the city and the village became friends.
"""


inputs, outputs, word_ids, vocab_size = preprocess.build_dataset(text)

w2v_model = model.Word2VecModel(vocab_size=vocab_size)
trained_model = train.train_w2c_model(w2v_model, inputs, outputs)

embeddings = trained_model.embeddings.weight.detach()

king_idx = word_ids['king']
queen_idx = word_ids['queen']
woman_idx = word_ids['woman']
man_idx = word_ids['man']

print (embeddings[queen_idx])
print (embeddings[king_idx] - embeddings[man_idx] + embeddings[woman_idx])