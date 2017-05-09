#coding:utf-8
import numpy as np
import sys
import math
import random
reload(sys)
sys.setdefaultencoding('utf-8')
tripleTotal = 1
entityTotal = 1
tagTotal = 1
def min(a, b):
	if a > b:
		return b
	return a
#get global values: tripleTotal, entityTotal, tagTotal
def getGlobalValues():
	return tripleTotal, entityTotal, tagTotal
#load triples from file
def getTriples(path):
	headList = []
	tailList = []
	relationList = []
	headSet = []
	tailSet = []
	f = open(path, "r")
	content = f.readline()
	global tripleTotal, entityTotal, tagTotal
	tripleTotal, entityTotal, tagTotal = [int(i) for i in content.strip().split()]
	for i in range(entityTotal):
		headSet.append(set())
		tailSet.append(set())
	while(True):
		content = f.readline()
		if content == "":
			break
		values = content.strip().split()
		values = [(int)(i) for i in values]
		headList.append(values[0])
		tailList.append(values[1])
		headSet[values[0]].add(values[1])
		tailSet[values[1]].add(values[0])
		relationList.append(values[2:])
	f.close()
	return headList, tailList, relationList, headSet, tailSet
#generate transNet training batches
def batch_iter(headList, tailList, relationList, headSet, tailSet, batch_size, aeBeta):
	data_size = len(headList)
	entity_size = entityTotal
 	# Shuffle the data at each epoch
	tag_size = tagTotal
	shuffle_indices = np.random.permutation(np.arange(data_size))
	start_index = 0
	end_index = min(start_index+batch_size, data_size)
	batch_id = 0
	while start_index < data_size:
		pos_h = []
		pos_t = []
		pos_r = []
		pos_b = []
		neg_h = []
		neg_t = []
		neg_r = []
		neg_b = []
		for i in range(start_index, end_index):
			cur_h = headList[shuffle_indices[i]]
			cur_t = tailList[shuffle_indices[i]]
			cur_r = relationList[shuffle_indices[i]]
			set_r = set(cur_r)
			r_one_hot = np.zeros(tag_size, dtype=float)
			b = np.ones(tag_size, dtype=float)
			r_one_hot[cur_r] = 1.0
			b[cur_r] = aeBeta
			#for index in cur_r:
			#	r_one_hot[index] = 1.0
			#replace head
			pos_h.append(cur_h)
			pos_t.append(cur_t)
			pos_r.append(r_one_hot)
			pos_b.append(b)
			rand_h = random.randint(0, entity_size-1)
			while(rand_h in tailSet[cur_t]):
				rand_h = random.randint(0, entity_size-1)
			neg_h.append(rand_h)
			neg_t.append(cur_t)
			neg_r.append(r_one_hot)
			neg_b.append(b)
			#repalce tail
			pos_h.append(cur_h)
			pos_t.append(cur_t)
			pos_r.append(r_one_hot)
			pos_b.append(b)
			rand_t = random.randint(0, entity_size-1)
			while(rand_t in headSet[cur_h]):
				rand_t = random.randint(0, entity_size-1)
			neg_h.append(cur_h)
			neg_t.append(rand_t)
			neg_r.append(r_one_hot)
			neg_b.append(b)
			#replace relation
			pos_h.append(cur_h)
			pos_t.append(cur_t)
			pos_r.append(r_one_hot)
			pos_b.append(b)
			rand_set_r = set([])
			rand_r = random.randint(0, tag_size-1)
			len_r = len(cur_r)
			while(len(rand_set_r) < len_r and len(rand_set_r) + len_r < tag_size):
				if (rand_r not in set_r) and (rand_r not in rand_set_r):
					rand_set_r.add(rand_r)
				rand_r = random.randint(0, tag_size-1)
			rand_cur_r = [r for r in rand_set_r]
			rand_r_one_hot = np.zeros(tag_size, dtype=float)
			rand_r_one_hot[rand_cur_r] = 1.0
			rand_b = np.ones(tag_size, dtype=float)
			rand_b[rand_cur_r] = aeBeta
			neg_h.append(cur_h)
			neg_t.append(cur_t)
			neg_r.append(rand_r_one_hot)
			neg_b.append(rand_b)
		batch_id += 1
		yield np.array(pos_h), np.array(pos_t), np.array(pos_r), np.array(pos_b), np.array(neg_h), np.array(neg_t), np.array(neg_r), np.array(neg_b)
		start_index = end_index
		end_index = min(start_index+batch_size, data_size)
#generate evaluation batches
def batch_test(headList, tailList, relationList, batch_size):
	data_size = len(headList)
	entity_size = entityTotal
	tag_size = tagTotal
	start_index = 0
	end_index = min(start_index+batch_size, data_size)
	batch_id = 0
	while start_index < data_size:
		pos_h = []
		pos_t = []
		pos_r = []
		for i in range(start_index, end_index):
			cur_h = headList[i]
			cur_t = tailList[i]
			cur_r = relationList[i]
			r_one_hot = np.zeros(tag_size, dtype=float)
			b = np.ones(tag_size, dtype=float)
			r_one_hot[cur_r] = 1.0
			pos_h.append(cur_h)
			pos_t.append(cur_t)
			pos_r.append(r_one_hot)
		#print 'batch ', batch_id
		batch_id += 1
		'''
		print pos_h
		print pos_t
		print pos_r
		print neg_h
		print neg_t
		print neg_r
		'''
		yield pos_h, pos_t, pos_r
		start_index = end_index
		end_index = min(start_index+batch_size, data_size)
#generate relation autoencoder warm-up batches
def batch_autoencoder(vecList, vec_size, batch_size, aeBeta):
	data_size = len(vecList)
	shuffle_indices = np.random.permutation(np.arange(data_size))
	start_index = 0
	end_index = min(start_index+batch_size, data_size)
	batch_id = 0
	while start_index < data_size:
		vecs = []
		bs = []
		for i in range(start_index, end_index):
			vec_index = vecList[shuffle_indices[i]]
			vec = np.zeros(vec_size, dtype=float)
			b = np.ones(vec_size, dtype=float)
			vec[vec_index] = 1.0
			b[vec_index] = aeBeta
			vecs.append(vec)
			bs.append(b)
		batch_id += 1
		yield np.array(vecs), np.array(bs)
		start_index = end_index
		end_index = min(start_index+batch_size, data_size)