from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import pprint
import modeling
import tokenization
import tensorflow.compat.v1 as tf1
import tensorflow as tf
import numpy as np

tf1.disable_eager_execution()
tf1.disable_v2_behavior()

# ------------------------------------------------------------------------------------

vocab_file = "./model/vocab.txt"
config_file = "./model/bert_config2.json"
checkpoint_file = "./model/bert_model.ckpt"
mapqa_file = "./mapqa.json"

tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case=True)
bert_config = modeling.BertConfig.from_json_file(config_file)
max_seq_length = 128

# ------------------------------------------------------------------------------------

model = None
predictions = None
mapqa = None

ph_input_ids = tf1.placeholder(dtype=tf.int32, shape=(None, max_seq_length), name="ph_input_ids")
ph_input_mask = tf1.placeholder(dtype=tf.int32, shape=(None, max_seq_length), name="ph_input_mask")
ph_input_type_ids = tf1.placeholder(dtype=tf.int32, shape=(None, max_seq_length), name="ph_input_type_ids")

# ================================================================================================


class InputFeatures(object):
	"""A single set of features of data."""

	def __init__(self, tokens, input_ids, input_mask, input_type_ids):
		self.tokens = tokens
		self.input_ids = input_ids
		self.input_mask = input_mask
		self.input_type_ids = input_type_ids


def get_feed_dict(features):
	input_ids = []
	input_mask = []
	input_type_ids = []

	for feature in features:
		input_ids.append(feature.input_ids)
		input_mask.append(feature.input_mask)
		input_type_ids.append(feature.input_type_ids)

	return {ph_input_ids: input_ids,
			ph_input_mask: input_mask,
			ph_input_type_ids: input_type_ids}

def convert_to_feature(question):
	global tokenizer, max_seq_length

	input_type_ids = []
	tokens = ["[CLS]"] + tokenizer.tokenize(question) + ["[SEP]"]

	# some stupid way of ensuring that the sentences don't get too long
	if len(tokens) > 128:
		tokens = tokens[0:127]
		tokens.append("[SEP]")

	input_type_ids = [0] * len(tokens)
	input_ids = tokenizer.convert_tokens_to_ids(tokens)
	input_mask = [1] * len(input_ids)

	# Zero-pad up to the sequence length.
	while len(input_ids) < max_seq_length:
		input_ids.append(0)
		input_mask.append(0)
		input_type_ids.append(0)

	return InputFeatures(
					tokens=tokens,
					input_ids=input_ids,
					input_mask=input_mask,
					input_type_ids=input_type_ids)


def load_model(session):
	global checkpoint_file

	if checkpoint_file == "./model/bert_model.ckpt":
		tvars = tf1.trainable_variables()
		(assignment_map, _
		) = modeling.get_assignment_map_from_checkpoint(tvars, checkpoint_file)
		tf1.train.init_from_checkpoint(checkpoint_file, assignment_map)
	else:
		new_saver = tf1.train.Saver()
		new_saver.restore(session, checkpoint_file)


def model_builder(session):
	global model, ph_input_ids, ph_input_mask, ph_input_type_ids, predictions, checkpoint_file

	model = modeling.BertModel(
		config=bert_config,
		is_training=False,
		input_ids=ph_input_ids,
		input_mask=ph_input_mask,
		token_type_ids=ph_input_type_ids)

	predictions = model.get_pooled_output()

	if checkpoint_file == "./model/bert_model.ckpt":
		load_model(session)

	init_op = tf1.global_variables_initializer()
	session.run(init_op)

	if checkpoint_file != "./model/bert_model.ckpt":
		load_model(session)


def find_answer(context):
	global mapqa

	res = "no answer found!"
	sim_score = 10000000

	for candidate in mapqa:
		a = candidate["answer"]
		c = np.array(candidate["context"])
		
		score = np.inner(c - context, c - context)

		if score < sim_score:
			sim_score = score
			res = a

	print(sim_score)
	return res


def answer_question():
	global model, mapqa, predictions, mapqa_file

	try:
		with open(mapqa_file, 'r') as infile:
			mapqa =	json.load(infile)
	except Exception as e:
		mapqa = []

	with tf1.Session() as session:
		model_builder(session)

		while True:
			print("ask me anything!")

			question = input()
			if question == "x":
				break

			features = [convert_to_feature(question)]
			context = session.run(predictions, feed_dict=get_feed_dict(features)).tolist()[0]

			answer = find_answer(context)

			print(answer)

		print("goodbye!")


def add_new_qa():
	global model, mapqa, predictions, mapqa_file

	try:
		with open(mapqa_file, 'r') as infile:
			mapqa =	json.load(infile)
	except Exception as e:
		mapqa = []

	print(mapqa)

	with tf1.Session() as session:
		model_builder(session)

		print("add new qa pairs!")

		question = "empty"
		while True:
			question = input()

			if question == "x":
				break

			answer = input()

			features = [convert_to_feature(question)]
			context = session.run(predictions, feed_dict=get_feed_dict(features)).tolist()

			mapqa.append({"question": question,
						  "answer": answer,
						  "context": context[0]})

	print("program terminate! saving new mapqa file!")

	with open(mapqa_file, 'w') as outfile:
		outfile.write(json.dumps(mapqa))

# add_new_qa()

answer_question()
