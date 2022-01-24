import json
from sentence_transformers import SentenceTransformer, util
import numpy as np

model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

mapqa2_file = "./mapqa2.json"
mapqa2 = None

def normalization(vector):
	return vector / np.linalg.norm(vector)


def find_answer(context):
	global mapqa2

	res = "no answer found!"
	sim_score = -3

	for candidate in mapqa2:
		a = candidate["answer"]
		c = np.array(candidate["context"])
		
		score = np.inner(c, context)
		score = score * score

		if score > sim_score:
			sim_score = score
			res = a

	print(sim_score)
	return res


def answer_question():
	global model, mapqa2, mapqa2_file

	try:
		with open(mapqa2_file, 'r') as infile:
			mapqa2 = json.load(infile)
	except Exception as e:
		mapqa2 = []

	while True:
		print("ask me anything!")

		question = input()
		if question == "x":
			break

		context = model.encode([question])[0]
		context = normalization(context)

		answer = find_answer(context)

		print(answer)

	print("goodbye!")


def add_new_qa():
	global model, mapqa2, mapqa2_file

	try:
		with open(mapqa2_file, 'r') as infile:
			mapqa2 =	json.load(infile)
	except Exception as e:
		mapqa2 = []

	print(mapqa2)

	print("add new qa pairs!")

	question = "empty"

	while True:
		question = input()

		if question == "x":
			break

		answer = input()

		context = model.encode([question])
		context = normalization(context)

		mapqa2.append({"question": question,
						"answer": answer,
						"context": context[0].tolist()})

	print("program terminate! saving new mapqa file!")

	with open(mapqa2_file, 'w') as outfile:
		outfile.write(json.dumps(mapqa2))

# add_new_qa()

answer_question()
