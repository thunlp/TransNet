#coding:utf-8
import numpy as np
import tensorflow as tf
import os
import time
import datetime
import random
from init import *

def variable_summaries(var, name):
		with tf.name_scope('summaries'):
			mean = tf.reduce_mean(var)
			tf.summary.scalar('mean/' + name, mean)
class TransNet(object):
	def __init__(self, config):
		relation_layers = config.relation_layers
		relation_layer_length = len(relation_layers)
		node_size = config.entity_size
		relation_size = relation_layers[0]
		mid_layer = (relation_layer_length-1)/2
		rep_size = relation_layers[mid_layer]
		l2_lambda = config.l2_lambda
		keep_prob = config.keep_prob
		hits_k = config.hits_k
		gamma = config.gamma
		alpha = config.alpha

		self.pos_h = tf.placeholder(tf.int32, [None])
		self.pos_t = tf.placeholder(tf.int32, [None])
		self.pos_r = tf.placeholder(tf.float32, [None, relation_size])
		self.pos_br = tf.placeholder(tf.float32, [None, relation_size])

		self.neg_h = tf.placeholder(tf.int32, [None])
		self.neg_t = tf.placeholder(tf.int32, [None])
		self.neg_r = tf.placeholder(tf.float32, [None, relation_size])
		self.neg_br = tf.placeholder(tf.float32, [None, relation_size])
		with tf.name_scope("node_lookup"):
			cur_seed = random.getrandbits(32)
			self.int_embeddings = tf.get_variable(name = "int_embeddings", shape = [node_size, rep_size], initializer = tf.contrib.layers.xavier_initializer(uniform = False, seed=cur_seed))
			cur_seed = random.getrandbits(32)
			self.adv_embeddings = tf.get_variable(name = "adv_embeddings", shape = [node_size, rep_size], initializer = tf.contrib.layers.xavier_initializer(uniform = False, seed=cur_seed))

			pos_h_e = tf.nn.l2_normalize(tf.nn.embedding_lookup(self.int_embeddings, self.pos_h), 1)
			pos_t_e = tf.nn.l2_normalize(tf.nn.embedding_lookup(self.adv_embeddings, self.pos_t), 1)

			neg_h_e = tf.nn.l2_normalize(tf.nn.embedding_lookup(self.int_embeddings, self.neg_h), 1)
			neg_t_e = tf.nn.l2_normalize(tf.nn.embedding_lookup(self.adv_embeddings, self.neg_t), 1)

		with tf.name_scope("relation_autoencoder"):
			self.relation_W = []
			self.relation_b = []
			self.pos_r_hidden = []
			self.pos_r_hidden_test = []
			self.neg_r_hidden = []
			self.relation_ae_l2_loss = 0.0
			for i in range(relation_layer_length-1):
				cur_seed = random.getrandbits(32)
				self.relation_W.append(tf.get_variable(name = "relation_W"+str(i), shape = [relation_layers[i], relation_layers[i+1]], initializer = tf.contrib.layers.xavier_initializer(uniform = False, seed=cur_seed)))
				self.relation_b.append(tf.Variable(name="relation_b"+str(i), initial_value = tf.zeros([relation_layers[i+1]])))
				self.relation_ae_l2_loss += tf.nn.l2_loss(self.relation_W[i])+tf.nn.l2_loss(self.relation_b[i])
				#feed pos_h, pos_t, neg_h, neg_t into node autoencoder
				if i == 0:
					relation_pos_r = tf.nn.tanh(tf.matmul(self.pos_r, self.relation_W[i])+self.relation_b[i])
					relation_neg_r = tf.nn.tanh(tf.matmul(self.neg_r, self.relation_W[i])+self.relation_b[i])
					relation_pos_r_test = tf.nn.tanh(tf.matmul(self.pos_r, self.relation_W[i])+self.relation_b[i])
				elif i == relation_layer_length-2:
					relation_pos_r = tf.nn.sigmoid(tf.matmul(self.pos_r_hidden[i-1], self.relation_W[i])+self.relation_b[i])
					relation_neg_r = tf.nn.sigmoid(tf.matmul(self.neg_r_hidden[i-1], self.relation_W[i])+self.relation_b[i])
					relation_pos_r_test = tf.nn.sigmoid(tf.matmul(self.pos_r_hidden_test[i-1], self.relation_W[i])+self.relation_b[i])
				else:
					relation_pos_r = tf.nn.tanh(tf.matmul(self.pos_r_hidden[i-1], self.relation_W[i])+self.relation_b[i])
					relation_neg_r = tf.nn.tanh(tf.matmul(self.neg_r_hidden[i-1], self.relation_W[i])+self.relation_b[i])
					relation_pos_r_test = tf.nn.tanh(tf.matmul(self.pos_r_hidden_test[i-1], self.relation_W[i])+self.relation_b[i])
				self.pos_r_hidden_test.append(relation_pos_r_test)
				if i == (relation_layer_length-3)/2:
					cur_seed = random.getrandbits(32)
					self.pos_r_rep = tf.nn.dropout(relation_pos_r, keep_prob, seed=cur_seed)
					cur_seed = random.getrandbits(32)
					self.neg_r_rep = tf.nn.dropout(relation_neg_r, keep_prob, seed=cur_seed)
					self.pos_r_hidden.append(self.pos_r_rep)
					self.neg_r_hidden.append(self.neg_r_rep)
				else:
					self.pos_r_hidden.append(relation_pos_r)
					self.neg_r_hidden.append(relation_neg_r)

			#calculate node ae loss
			self.relation_loss = tf.reduce_sum(abs(tf.mul(self.pos_r_hidden[-1] - self.pos_r, self.pos_br)))
			self.relation_loss += tf.reduce_sum(abs(tf.mul(self.neg_r_hidden[-1] - self.neg_r, self.neg_br)))
			variable_summaries(self.relation_loss, 'relation_loss')
			#for relation ae warm-up
			self.relation_pos_r_loss = tf.reduce_sum(abs(tf.mul(self.pos_r_hidden[-1] - self.pos_r, self.pos_br)))+l2_lambda*self.relation_ae_l2_loss
			self.relation_sum = tf.reduce_sum(self.pos_r)
			self.relation_hits = []
			for k in hits_k:
				relation_topk = tf.nn.top_k(self.pos_r_hidden_test[-1], k=k).indices
				relation_pred = tf.reduce_sum(tf.one_hot(relation_topk, relation_size), 1)
				self.relation_hits.append(tf.reduce_sum(tf.mul(relation_pred, self.pos_r)))
		with tf.name_scope("trans"):
			pos = tf.reduce_sum(abs(pos_h_e + self.pos_r_rep - pos_t_e), 1, keep_dims = True)
			neg = tf.reduce_sum(abs(neg_h_e + self.neg_r_rep - neg_t_e), 1, keep_dims = True)
			self.trans_loss = tf.reduce_sum(tf.maximum(pos - neg + gamma, 0))
			self.l2_loss = self.relation_ae_l2_loss
			self.loss = self.trans_loss+alpha*self.relation_loss+l2_lambda*self.l2_loss
			variable_summaries(self.trans_loss, 'trans_loss')
			variable_summaries(self.loss, 'loss')
		with tf.name_scope("evaluation"):
			self.pos_r_minus = pos_t_e - pos_h_e
			self.pos_r_dec = self.pos_r_minus
			for i in range(mid_layer, relation_layer_length-1):
				if i == relation_layer_length-2:
					self.pos_r_dec = tf.nn.sigmoid(tf.matmul(self.pos_r_dec, self.relation_W[i])+self.relation_b[i])
				else:
					self.pos_r_dec = tf.nn.tanh(tf.matmul(self.pos_r_dec, self.relation_W[i])+self.relation_b[i])
			self.sum = tf.reduce_sum(self.pos_r)
			self.hits = []
			for k in hits_k:
				topk_indices = tf.nn.top_k(self.pos_r_dec, k=k).indices
				pred = tf.reduce_sum(tf.one_hot(topk_indices, relation_size), 1)
				self.hits.append(tf.reduce_sum(tf.mul(pred, self.pos_r)))
			self.sorted_indices = tf.nn.top_k(self.pos_r_dec, k=relation_size).indices
			self.topk_sorted = tf.nn.top_k(self.pos_r_dec, k=hits_k[-1]).indices
			self.merged = tf.summary.merge_all()
class Config(object):
	def __init__(self):
		self.warm_up_epochs_relation = 40
		self.epochs = 200
		self.batch_size = 200
		self.eval_batch_size = 2000
		self.entity_size = 0
		self.tag_size = 0
		self.gamma = 1
		self.alpha = 0.01
		self.l2_lambda = 0.001
		self.hits_k = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
		self.keep_prob = 0.5
		self.beta = 100.0
		self.relation_layers = []
def makedir(path):
	if not os.path.exists(path):
		os.mkdir(path)
if __name__ == "__main__":
	'''
	sys.argv[1]: name of dataset
	sys.argv[2]: alpha, the weight of autoencoder loss
	sys.argv[3]: beta, the weight of non-zero element in autoencoder
	sys.argv[4]: if >=0, reload saved autoencoder parameters and skip warm-up process
	sys.argv[5]: if >=0, reload saved TransNet parameters
	'''
	#fix random seeds
	np.random.seed(0)
	random.seed(0)
	#set working dirs
	dataDir = "./data/"+sys.argv[1]
	modelDir = dataDir+"models/"
	resultDir = dataDir+"results/"
	summaryDir = dataDir+"summaries/"
	makedir(modelDir)
	makedir(resultDir)
	makedir(summaryDir)
	makedir(modelDir+"relation/")
	#load train, valid, test data
	headList, tailList, relationList, headSet, tailSet = getTriples(dataDir+"train.txt")
	headList_test, tailList_test, relationList_test, headSet_test, tailSet_test = getTriples(dataDir+"test.txt")
	headList_valid, tailList_valid, relationList_valid, headSet_valid, tailSet_valid = getTriples(dataDir+"valid.txt")
	#set tagTotal, entityTotal
	tripleTotal, entityTotal, tagTotal = getGlobalValues()
	for i in range(entityTotal):
		headSet[i] = headSet[i]|headSet_test[i]|headSet_valid[i]
		tailSet[i] = tailSet[i]|tailSet_test[i]|tailSet_valid[i]
	tripleTotal = len(headList)
	config = Config()
	config.tag_size = tagTotal
	config.entity_size = entityTotal
	config.relation_layers = [tagTotal, 100, tagTotal]
	config.alpha = float(sys.argv[2])
	config.beta = float(sys.argv[3])
	with tf.Graph().as_default():
		sess = tf.Session()
		with sess.as_default():
			cur_seed = random.getrandbits(32)
			initializer = tf.contrib.layers.xavier_initializer(uniform = False, seed=cur_seed)
			with tf.variable_scope("model", reuse=None, initializer = initializer):
				model = TransNet(config = config)
				optimizer = tf.train.AdamOptimizer(0.001)
				train_op = optimizer.minimize(model.loss)
				save_variables = model.relation_W+model.relation_b
				save_variables.append(model.adv_embeddings)
				save_variables.append(model.int_embeddings)
				saver = tf.train.Saver(save_variables)
				saver_relation = tf.train.Saver(model.relation_W+model.relation_b)
			sess.run(tf.global_variables_initializer())
			def hits_relation():
				hits = [0]*len(config.hits_k)
				all_count = 0.0
				batches = batch_autoencoder(relationList_valid[:min(len(relationList_valid), 50000)], config.tag_size, config.eval_batch_size, config.beta)
				instance_id = 0
				for batch in batches:
					inputs, bs = batch
					feed_dict = {
						model.pos_r: inputs,
						model.pos_br: bs
						}
					cur_hits, cur_sum = sess.run([model.relation_hits, model.relation_sum], feed_dict=feed_dict)
					hits = list(map(lambda x: x[0]+x[1], zip(hits, cur_hits)))
					all_count += cur_sum
				hits_k = [hit/all_count for hit in hits]
				return hits_k
			def evaluation_transNet(hList, tList, rList, write=False):
				hits = [0]*len(config.hits_k)
				all_count = 0.0
				sum_rank = 0.0
				batches = batch_test(hList, tList, rList, config.eval_batch_size)
				instance_id = 0
				if write:
					f = open(resultDir+"prediction_"+str(config.beta)+"_"+str(config.alpha)+".txt", 'w')
				for batch in batches:
					pos_h, pos_t, pos_r = batch
					feed_dict = {
						model.pos_h: pos_h,
						model.pos_t: pos_t,
						model.pos_r: pos_r
						}
					cur_hits, cur_sum, cur_sorted = sess.run([model.hits, model.sum, model.sorted_indices], feed_dict=feed_dict)
					hits = list(map(lambda x: x[0]+x[1], zip(hits, cur_hits)))
					for i in range(len(pos_r)):
						if write:
							f.write(str(instance_id)+" "+str(rList[instance_id])+" ")
							f.write(str(cur_sorted[i][0:10])+"\n")
						for j in range(config.tag_size):
							if pos_r[i][cur_sorted[i][j]] == 1.0:
								sum_rank += j+1
						instance_id += 1
					all_count += cur_sum
				if write:
					f.close()
				hits_k = [hit/all_count for hit in hits]
				mean_rank = sum_rank/all_count
				return hits_k, mean_rank
			def evaluation_transNet_noMR(hList, tList, rList, write=False):
				hits = [0]*len(config.hits_k)
				p = [0]*len(config.hits_k)
				r = [0]*len(config.hits_k)
				p_indice = [float(i+1) for i in range(len(hits))]
				all_count = 0.0
				sum_rank = 0.0
				batches = batch_test(hList, tList, rList, config.eval_batch_size)
				instance_id = 0
				if write:
					f = open(resultDir+"prediction_"+str(config.beta)+"_"+str(config.alpha)+".txt", 'w')
				for batch in batches:
					pos_h, pos_t, pos_r = batch
					feed_dict = {
						model.pos_h: pos_h,
						model.pos_t: pos_t,
						model.pos_r: pos_r
						}
					cur_hits, cur_sum, topk_sorted = sess.run([model.hits, model.sum, model.topk_sorted], feed_dict=feed_dict)
					hits = list(map(lambda x: x[0]+x[1], zip(hits, cur_hits)))
					p_value = [len(pos_r)*indice for indice in p_indice]
					p = list(map(lambda x: x[0]+x[1], zip(p, p_value)))
					for i in range(len(pos_r)):
						if write:
							f.write(str(instance_id)+" "+str(rList[instance_id])+" ")
							f.write(str(topk_sorted[i])+"\n")
						instance_id += 1
					all_count += cur_sum
				if write:
					f.close()
				r = [hit/all_count for hit in hits]
				p_new = [hits[i]/p[i] for i in range(len(hits))]
				return p_new, r
			def train_step(pos_h_batch, pos_t_batch, pos_r_batch, pos_br_batch, neg_h_batch, neg_t_batch, neg_r_batch, neg_br_batch, cur_train_op):
				feed_dict = {
					model.pos_h: pos_h_batch,
					model.pos_t: pos_t_batch,
					model.pos_r: pos_r_batch,
					model.pos_br: pos_br_batch,
					model.neg_h: neg_h_batch,
					model.neg_t: neg_t_batch,
					model.neg_r: neg_r_batch,
					model.neg_br: neg_br_batch,
				}
				_, loss, relation_loss, summary = sess.run(
					[cur_train_op, model.loss, model.relation_loss, model.merged], feed_dict)
	 			return loss, relation_loss, summary
			def initRelation():
				#relationAE variebles
				#warm-up stage: initialize the auto-encoder
				print "Starting warm-up relation"
				train_op = optimizer.minimize(model.relation_pos_r_loss)
				init_relation_file = open(resultDir+"init_beta"+str(config.beta)+".txt", 'w')
				max_hits_k = 0.0
		 		for epoch in range(config.warm_up_epochs_relation):
		 			time_str = datetime.datetime.now().isoformat()
		 			print 'Warm-up relation epoch: ', epoch, ' ', time_str
		 			sum_loss = 0.0
		 			batches = batch_autoencoder(relationList, config.tag_size, config.batch_size, config.beta)
		 			batch_id = 0
					for batch in batches:
						vecs, bs = batch
						feed_dict = {
							model.pos_r: vecs,
							model.pos_br: bs
						}
						_, cur_loss = sess.run([train_op, model.relation_pos_r_loss], feed_dict)
						sum_loss += cur_loss
						batch_id += 1
						if batch_id % 5000 == 0:
							time_str = datetime.datetime.now().isoformat()
							print 'batch ', batch_id, ' loss = ', cur_loss, ' ', time_str
					print sum_loss
					init_relation_file.write(str(epoch)+" "+time_str+" "+str(sum_loss)+"\n")
					if epoch % 5 == 0:
						hits_k = hits_relation()
						for i in range(len(hits_k)):
							print 'Hits' + str(config.hits_k[i]), hits_k[i]
							init_relation_file.write('Hits'+str(config.hits_k[i])+' '+str(hits_k[i])+"\n")
						init_relation_file.flush()
						if hits_k[0] > max_hits_k:
							max_hits_k = hits_k[0]
							saver_relation.save(sess, modelDir+'relation/transNet-relation-beta'+str(config.beta)+'-', global_step=epoch)
					if epoch == config.warm_up_epochs_relation-1:
						saver_relation.save(sess, modelDir+'relation/transNet-relation-beta'+str(config.beta)+'-', global_step=epoch)
				init_relation_file.close()
			def train():
				#reload parameters
				start = 0
				if int(sys.argv[4]) >= 0:
					saver_relation.restore(sess, modelDir+'relation/transNet-relation-beta'+str(config.beta)+'--'+sys.argv[4])
				else:
					initRelation()
				if int(sys.argv[5]) >= 0:
					saver.restore(sess, modelDir+'transNet-beta'+str(config.beta)+"-lambda"+str(config.alpha)+"-0")
					train_transNet_file = open(resultDir+"train_transNet_"+str(config.beta)+"_"+str(config.alpha)+".txt", 'a')
					start = int(sys.argv[5])
				else:
					train_transNet_file = open(resultDir+"train_transNet_"+str(config.beta)+"_"+str(config.alpha)+".txt", 'w')
				# train TransNet
				print "Starting train transNet"
				if not os.path.exists(dataDir + '/summaries/'+str(config.beta)+"_"+str(config.alpha)):
					os.mkdir(dataDir + '/summaries/'+str(config.beta)+"_"+str(config.alpha))
				train_writer = tf.summary.FileWriter(dataDir + '/summaries/'+str(config.beta)+"_"+str(config.alpha), sess.graph)
				var_list = model.relation_W+model.relation_b
				var_list.append(model.adv_embeddings)
				var_list.append(model.int_embeddings)
				train_op = optimizer.minimize(model.loss, var_list=var_list)
				max_hits_k = 0.0
				global_batch_id = 0
		 		for epoch in range(start, config.epochs):
		 			time_str = datetime.datetime.now().isoformat()
		 			print 'Train TransNet epoch: ', epoch, ' ', time_str
		 			sum_loss = 0.0
		 			batches = batch_iter(headList, tailList, relationList, headSet, tailSet, config.batch_size, config.beta)
	 				batch_id = 0
					for batch in batches:
						pos_h, pos_t, pos_r, pos_b, neg_h, neg_t, neg_r, neg_b = batch
						cur_loss, relation_loss, summary = train_step(pos_h, pos_t, pos_r, pos_b, neg_h, neg_t, neg_r, neg_b, train_op)
						train_writer.add_summary(summary, global_batch_id)
						sum_loss += cur_loss
						batch_id += 1
						global_batch_id += 1
						if batch_id % 2000 == 0:
							time_str = datetime.datetime.now().isoformat()
							print 'batch ', batch_id, ' loss = ', cur_loss, ' ', time_str
					print sum_loss
					train_transNet_file.write(str(epoch)+" "+time_str+" "+str(sum_loss)+"\n")
					#evaluation part
					indices_count = {}
					indices_sum = 0
					for i in range(config.tag_size):
						indices_count[i] = 0
					if epoch % 1 == 0:
						print 'Evaluating...'
						if epoch % 10 == 0 and epoch > 5:
							p_test, r_test = evaluation_transNet_noMR(headList_test, tailList_test, relationList_test, True)
						else:
							p_test, r_test = evaluation_transNet_noMR(headList_test, tailList_test, relationList_test)
						p_valid, r_valid = evaluation_transNet_noMR(headList_valid, tailList_valid, relationList_valid)
						for i in range(len(p_test)):
							print 'Test Precision ' + str(config.hits_k[i]), p_test[i], 'Valid Precision ' + str(config.hits_k[i]), p_valid[i]
							print 'Test Recall ' + str(config.hits_k[i]), r_test[i], 'Valid Recall ' + str(config.hits_k[i]), r_valid[i]
							train_transNet_file.write('Test Precision ' + str(config.hits_k[i])+' '+str(p_test[i])+' Valid Precision ' + str(config.hits_k[i])+' '+str(p_valid[i]))
							train_transNet_file.write('Test Recall ' + str(config.hits_k[i])+' '+str(r_test[i])+' Valid Recall ' + str(config.hits_k[i])+' '+str(r_valid[i]))
						train_transNet_file.write("\n")
						#print 'Test meanrank', mean_rank_test, 'Valid meanrank', mean_rank_valid
						#train_transNet_file.write('Test meanrank '+str(mean_rank_test)+' Valid meanrank '+str(mean_rank_valid)+"\n")
						train_transNet_file.flush()
						if r_valid[-1] > max_hits_k and epoch >= 50:
							max_hits_k = r_valid[-1]
							saver.save(sess, modelDir+'transNet-beta'+str(config.beta)+"-lambda"+str(config.alpha), global_step=0)
							f = open(resultDir+"int_embeddings_beta"+str(config.beta)+"_lambda"+str(config.alpha)+".txt", "w")
							embeddings = sess.run(tf.nn.l2_normalize(model.int_embeddings.eval(), 1))
							for i in embeddings:
								for j in i:
									f.write(str(j)+' ')
								f.write("\n")
							f.close()
							f = open(resultDir+"adv_embeddings_beta"+str(config.beta)+"_lambda"+str(config.alpha)+".txt", "w")
							embeddings = sess.run(tf.nn.l2_normalize(model.adv_embeddings.eval(), 1))
							for i in embeddings:
								for j in i:
									f.write(str(j)+' ')
								f.write("\n")
							f.close()
				train_transNet_file.close()
				train_writer.close()
			train()
