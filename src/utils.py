from matplotlib import ticker
import matplotlib.dates as mdates
import os
import matplotlib.pyplot as plt
import json
from blockchain import blockexplorer as bx
import requests
import csv
import datetime
import numpy as np
import pandas as pd
from config import APITOKEN


currency = 'btc'
api_path = 'https://api.graphsense.info/' + currency + '/'
headers = {'accept': 'application/json','Authorization': APITOKEN}
base_path = '../data/payment_flows/'
##############################
# helper functions
##############################


def list_to_file(l, path):
	with open(path, 'w') as f:
		for item in l:
			f.write("%s\n" % item)


def file_to_list(path):
	with open(path, 'r') as f:
		l = f.read().splitlines()
	return l


def read_json(path):
	with open(path, 'r') as fp:
		js = json.load(fp)
	return js


def write_json(js, path):
	with open(path, 'w') as fp:
		json.dump(js, fp)


def ts2date(ts):
	return datetime.datetime.fromtimestamp(int(ts)).strftime('%Y-%m-%d %H:%M:%S')


def url2dict(url):
	# print(url)
	# try:
	r = requests.get(url, headers=headers)
	d = json.loads(r.text)
	return d
	# except Exception as e:
	# 	print(url)
	# 	print(e)
	# 	return None


##############################
# api calls
##############################


def get_address_info(address):
	url = api_path + 'address/' + address 
	return url2dict(url)


def get_address_clusterID(address):
	address = str(address)
	url = api_path + 'address/' + address + '/cluster' 
	d = url2dict(url)
	if d and 'cluster' in d:
		clusterID = d['cluster']
		return clusterID
	return None


def get_address_tags(address):
	address = str(address)
	url = api_path + 'address/' + address + '/tags' 
	d = url2dict(url)
	if d:
		tags = list(set([el['label'] for el in d]))
		return tags
	return []

def get_address_tag(address):
	tag = 'unknown'
	tags = get_address_tags(address)
	if tags:
		tag = tags[0]
	return tag


def get_address_txs(address, limit=0):
	address = str(address)
	if not limit:
		limit = get_address_n_transactions(address) 
		if not limit:
			return {}
	url = api_path + 'address/' + address + '/transactions?limit=' + str(limit)
	d = url2dict(url)
	return [el['txHash'] for el in d['transactions']]


def get_address_txs_out(address, limit=0):
	## get hashes of txs where address is among inputs
	if not limit:
		limit = get_address_n_transactions(address)
	txs = get_address_txs(address,limit)
	tx_hashes_out = set()
	for tx_hash in txs:
		addresses_in = get_tx_addresses_in(tx_hash)
		if address in addresses_in:
			tx_hashes_out.add(tx_hash)	
	return tx_hashes_out


def get_address_txs_in(address, limit=0):
	## get hashes of txs where address is among outputs
	if not limit:
		limit = get_address_n_transactions(address)
	txs = get_address_txs(address, limit)
	tx_hashes_in = set()
	for tx_hash in txs:
		addresses_out = get_tx_addresses_out(tx_hash)
		if address in addresses_out:
			tx_hashes_in.add(tx_hash)	
	return tx_hashes_in


def get_address_neighbors_out(address, limit=0):
	## get all addresses receiving money from this address which means:
	## scan all edges and if source==address, select the target
	address = str(address)
	if not limit:
		limit = get_address_n_transactions_out(address)
		if not limit:
			return {}
	url = api_path + 'address/' + address + '/neighbors?direction=out&limit=' + str(limit) 
	d = url2dict(url)
	neighbors_out = set()
	if d:
		for e in d['edges']:
			if e['source'] == address:
				neighbors_out.add(e['target'])
	return neighbors_out


def get_address_neighbors_in(address, limit=0):
	## get all addresses sending money to this address which means:
	## scan all edges and if target==address, select the source
	address = str(address)
	if not limit:
		limit = get_address_n_transactions_in(address)
		if not limit:
			return {}
	url = api_path + 'address/' + address + '/neighbors?direction=in&limit=' + str(limit) 
	d = url2dict(url)
	neighbors_in = set()
	if d:
		for e in d['edges']:
			if e['target'] == address:
				neighbors_in.add(e['source'])
	return neighbors_in


def get_address_neighbors(address, limit):
	limit = int(limit/2) ## because we do a union later
	neighbors_in = get_address_neighbors_in(address, limit)
	neighbors_out = get_address_neighbors_out(address, limit)
	return neighbors_out.union(neighbors_in)


def get_address_money_in(address, coin='satoshi'):
	money = 0
	info = get_address_info(address)
	if 'totalReceived' in info.keys():
		money = info['totalReceived'][coin]
	return money


def get_address_money_out(address, coin='satoshi'):
	money = 0
	info = get_address_info(address)
	if 'totalSpent' in info.keys():
		money = info['totalSpent'][coin]
	return money


def get_address_n_transactions(address):
	return get_address_n_transactions_out(address) + get_address_n_transactions_in(address)


def get_address_n_transactions_out(address):
	n = 0
	info = get_address_info(address)
	if 'noOutgoingTxs' in info.keys():
		n = info['noOutgoingTxs']
	return n


def get_address_n_transactions_in(address):
	n = 0
	info = get_address_info(address)
	if 'noIncomingTxs' in info.keys():
		n = info['noIncomingTxs']
	return n


def get_address_balance(address, currency='eur'):
	balance = 0
	info = get_address_info(address)
	if info:
		balance = info['balance'][currency]
	return balance


def get_address_received(address, currency='eur'):
	received = 0
	info = get_address_info(address)
	if info:
		received = info['totalReceived'][currency]
	return received


def get_cluster_info(cluster):
	url = api_path + 'cluster/' + str(cluster)
	return url2dict(url)


def get_cluster_addresses(cluster, limit=0):
	if not limit:
		limit = get_cluster_n_addresses(cluster)
		if not limit:
			return {}
	url = api_path + 'cluster/' + str(cluster) + '/addresses?limit=' + str(limit)
	res = url2dict(url)
	if 'addresses' in res:
		return [el['address'] for el in res['addresses']]
	else:
		return res

def get_cluster_tags(cluster):
	url = api_path + 'cluster/' + str(cluster) + '/tags'
	d =  url2dict(url)
	if d:
		tags = list(set([el['label'] for el in d]))
		return tags
	return 'Unknown'


def get_cluster_tag(cluster):
	tag = 'Unknown'
	tags = get_cluster_tags(cluster)
	if tags:
		tag = tags[0]
	return tag


def get_cluster_n_neighbors_out(cluster):
	n = 0
	info = get_cluster_info(cluster)
	if 'outDegree' in info.keys():
		n = info['outDegree']
	return n


def get_cluster_n_neighbors_in(cluster):
	n = 0
	info = get_cluster_info(cluster)
	if 'inDegree' in info.keys():
		n = info['inDegree']
	return n


def get_cluster_neighbors_out(cluster, limit=0):
	## get all clusters receiving money from this cluster which means:
	## scan all edges and if source==cluster, select the target
	neighbors_out = set()
	if not limit:
		limit = get_cluster_n_transactions_out(cluster)
		if not limit:
			return neighbors_out
	url = api_path + 'cluster/' + str(cluster) + '/neighbors?direction=out&limit=' + str(limit)
	d = url2dict(url)
	if d:
		for e in d['edges']:
			if e['source'] == cluster:
				neighbors_out.add(e['target'])
	return neighbors_out


def get_cluster_neighbors_in(cluster, limit=0):
	## get all clusters sending money to this cluster which means:
	## scan all edges and if target==cluster, select the source
	if not limit:
		limit = get_cluster_n_transactions_out(cluster)
		if not limit:
			return neighbors_in
	url = api_path + 'cluster/' + str(cluster) + '/neighbors?direction=in&limit=' + str(limit)
	d = url2dict(url)
	neighbors_in = set()
	if d:
		for e in d['edges']:
			if e['target'] == cluster:
				neighbors_in.add(e['source'])
	return neighbors_in


def get_cluster_neighbors(cluster, limit):
	limit = int(limit/2)
	neighbors_in = get_cluster_neighbors_in(cluster, limit)
	neighbors_out = get_cluster_neighbors_out(cluster, limit)
	return neighbors_out.union(neighbors_in)


def get_cluster_n_transactions(cluster):
	return get_cluster_n_transactions_out(cluster) + get_cluster_n_transactions_out(cluster)


def get_cluster_n_transactions_out(cluster):
	n = 0
	info = get_cluster_info(cluster)
	if 'noOutgoingTxs' in info.keys():
		n = info['noOutgoingTxs']
	return n


def get_cluster_n_transactions_in(cluster):
	n = 0
	info = get_cluster_info(cluster)
	if 'noIncomingTxs' in info.keys():
		n = info['noIncomingTxs']
	return n


def get_cluster_balance(cluster, currency='eur'):
	balance = 0
	info = get_cluster_info(cluster)
	if info:
		balance = info['balance'][currency]
	return balance

def get_cluster_received(cluster, currency='eur'):
	received = 0
	info = get_cluster_info(cluster)
	if info:
		received = info['totalReceived'][currency]
	return received

def get_cluster_n_addresses(cluster):
	n_addresses = 0
	info = get_cluster_info(cluster)
	if info:
		n_addresses = info['noAddresses']
	return n_addresses

def get_cluster_txs_timestamps(cluster, limit=0):
	## returns a list of transaction timestamps for a cluster (or address)
	tss = []
	## get all cluster addresses txs
	addresses = get_cluster_addresses(cluster, limit=get_cluster_info(cluster)['noAddresses'])
	## for each address get txs_ts and append them
	for address in addresses:
		txs = get_address_txs(address, limit=get_address_n_transactions(address)+1)
		tmp = [el['timestamp'] for el in txs]
		for t in tmp:
			tss.append(t)
	return tss

## TODO:
# def get_cluster_n_neighbors_in()
# def get_cluster_n_neighbors_out()
# def get_cluster_n_neighbors()
# def get_cluster_txs()
# def get_cluster_txs_out()
# def get_cluster_txs_in()

def get_tx(tx_hash):
	url = api_path + 'tx/' + tx_hash
	return url2dict(url)


def get_tx_addresses_in(tx_hash):
	tx = get_tx(tx_hash)
	addresses_in = [el['address'] for el in tx['inputs']]
	return addresses_in


def get_tx_addresses_out(tx_hash):
	tx = get_tx(tx_hash)
	addresses_out = [el['address'] for el in tx['outputs']]
	return addresses_out


def get_tx_clusters_in(tx_hash):
	clusters_in = [get_address_clusterID(addr) for addr in get_tx_addresses_in(tx_hash)]
	return clusters_in


def get_tx_clusters_out(tx_hash):
	clusters_out = [get_address_clusterID(addr) for addr in get_tx_addresses_out(tx_hash)]
	return clusters_out


def get_tx_tags_in(tx_hash):
	tags_in = [get_address_tag(addr) for addr in get_tx_addresses_in(tx_hash)]
	return tags_in


def get_tx_tags_out(tx_hash):
	tags_out = [get_address_tag(addr) for addr in get_tx_addresses_out(tx_hash)]
	return tags_out


def get_tx_values_out(tx_hash, currency='eur'):
	return [el['value'][currency] for el in get_tx(tx_hash)['outputs']]


def get_tx_values_in(tx_hash, currency='eur'):
	return [el['value'][currency] for el in get_tx(tx_hash)['inputs']]


def get_addresses_txs(addresses, direction='both'):
	addresses_txs = dict() # key: address, value: list of txs
	ln = len(addresses)
	addresses = list(addresses)
	addresses.sort()
	for i, addr in enumerate(addresses):
		print(ln, i, addr, end='\r')
		if direction == 'in':
			addresses_txs[addr] = [get_tx(hsh) for hsh in get_address_txs_in(addr)]
		if direction == 'out':
			addresses_txs[addr] = [get_tx(hsh) for hsh in get_address_txs_out(addr)]
		if direction == 'both':
			addresses_txs[addr] = [get_tx(hsh) for hsh in get_address_txs(addr)]
	return addresses_txs


##############################
# notebook functions
##############################


def plot_addresses_txs_in_wrapper(addresses, address_type, min_n_txs=5):
	plot_addresses_txs_in(get_addresses_txs(addresses, direction='in'), address_type, min_n_txs=min_n_txs)
#	 return addresses_txs


def plot_addresses_txs_in(address_txs, address_type, min_n_txs=5):
	## https://stackoverflow.com/questions/52697594/python-matplotlib-how-to-change-color-of-each-lollipop-in-a-lollipop-plot-ax
	## address_txs is a dict with address as key and list of txs as value
	## get all tx timestamps and the net received BTC of an address
	fs = 18
	currency = 'usd'
	for i, address in enumerate(list(address_txs.keys())):
		x = []
		y = []
		x_pos = []
		x_neg = []
		x_ = []
		y_pos = []
		y_neg = []
		y_ = []
		for tx in address_txs[address][::-1]:
#			 x.append(tx['timestamp'])
#			 y.append(out['value'][currency])
			for out in tx['outputs']:
				if out['address'] == address:
					if out['value'][currency] > 0:
						if len(tx['outputs']) == 2:
							x_pos.append(tx['timestamp'])
							y_pos.append(out['value'][currency])
						elif len(tx['outputs']) == 1:
							x_.append(tx['timestamp'])
							y_.append(out['value'][currency])
						else:
							x_neg.append(tx['timestamp'])
							y_neg.append(out['value'][currency])

		if len(y_pos) > min_n_txs: # avoid addresses with few txs
			plt.figure(figsize=(15,8))
#			 y_cumsum = np.array(y).cumsum().tolist()
#			 x_cumsum = x[:len(y_cumsum)]

#			 np_dates = [np.datetime64(datetime.datetime.utcfromtimestamp(x_cumsum[i])) for i in range(len(x_cumsum))]
			np_dates_pos = [np.datetime64(datetime.datetime.utcfromtimestamp(x_pos[i])) for i in range(len(x_pos))]
			np_dates_neg = [np.datetime64(datetime.datetime.utcfromtimestamp(x_neg[i])) for i in range(len(x_neg))]
			np_dates_ = [np.datetime64(datetime.datetime.utcfromtimestamp(x_[i])) for i in range(len(x_))]
			if np_dates_:
				datemin = np.datetime64(min(np_dates_pos[0], np_dates_neg[0], np_dates_[0]), 'Y')
				datemax = np.datetime64(max(np_dates_pos[-1], np_dates_neg[-1], np_dates_[-1]), 'Y') + np.timedelta64(1, 'Y')
			elif np_dates_neg:
				datemin = np.datetime64(min(np_dates_pos[0], np_dates_neg[0]), 'Y')
				datemax = np.datetime64(max(np_dates_pos[-1], np_dates_neg[-1]), 'Y') + np.timedelta64(1, 'Y')
			else:
				datemin = np.datetime64(np_dates_pos[0], 'Y')
				datemax = np.datetime64(np_dates_pos[-1], 'Y') + np.timedelta64(1, 'Y')
			years = mdates.YearLocator()   # every year
			months = mdates.MonthLocator()  # every month
			yearsFmt = mdates.DateFormatter('%Y')

	#		 plt.step(np_dates, y_cumsum, color='b', where='post', label='balance')
			plt.stem(np_dates_pos,y_pos, 'g', markerfmt='go', label='2 ouputs')
			if np_dates_neg:
				plt.stem(np_dates_neg,y_neg, 'r', markerfmt='ro', label='not 2 outputs')
			if np_dates_:
				plt.stem(np_dates_,y_, 'k', markerfmt='ko', label='1 output')
	#		 title = address + ' ransom range: ' + str(address_ransom_range[address][0]) + '-' + str(address_ransom_range[address][1])
			title = str(i) + ' ' + address
			plt.title(title)
			plt.yscale(value='log')
			plt.ylabel('USD', fontsize=fs)
			plt.xlabel('Time', fontsize=fs)
			plt.legend()
			plt.xticks(fontsize=fs, rotation=45)
			plt.grid(which='both', axis='y')
			# plt.savefig('../images/txs_in_' + address_type + '_' + address + '.pdf')
			plt.show()


##############################
# flow functions
##############################


def tree2step_dict(step_dict, tree, step, origin, use_cluster_ID=False, use_address_clusterID_tag=True, exchanges_flag_needed=True):
	exchange_found = False
	tag = 'None'
	for child in tree['children']:
		redeemed_tx = ''
		if 'redeemed_tx' in child.keys():
			redeemed_tx = child['redeemed_tx'][0]
		if 'name' in child.keys():
			node_name = child['name']
			if use_address_clusterID_tag:
				cluster_ID = get_address_clusterID(node_name)
				tag = get_cluster_tag(cluster_ID)
				node_name = node_name + '_' + str(cluster_ID) + '_' + tag  
			if node_name not in step_dict.keys():
				step_dict[node_name] = []
			tx_info = dict()
			tx_info['received_BTC'] = child['value']
			tx_info['redeemed_tx'] = redeemed_tx
			tx_info['when'] = tree['time']
			tx_info['from'] = origin
			step_dict[node_name].append(tx_info)
			node_name = node_name + '_' + str(step + 1)
			prev_node_name = tx_info['from'] + '_' + str(step)
	if exchanges_flag_needed:
		return step_dict, exchange_found
	return step_dict

def do_steps(steps_list, step, max_steps, use_cluster_ID=True, nodes_limit=100, exchange_found=False):
	# nodes_limit is to avoid scraping tons of transactions when max_steps is too large
	if step <= max_steps and len(steps_list[step-1].keys()) < nodes_limit and not exchange_found:
		step_dict = dict() # addresses as keys, list of [BTC, redeemed_tx] as value
		for node_name in steps_list[step-1].keys():
			# see how it spent BTC
			for tx in steps_list[step-1][node_name]: # an addr or cluster can receive BTC multiple times from different addresses of the previous level
				tx_index = tx['redeemed_tx']
				if tx_index:
					tree_url = 'https://blockchain.info/tree/' + str(tx_index) + '?format=json'
					tree = url2dict(tree_url)
					# for each spent output get indeces of next level
					step_dict, exchange_flag = tree2step_dict(step_dict, tree, step, node_name, use_cluster_ID=use_cluster_ID, exchanges_flag_needed=True)
					if exchange_flag or exchange_found:
						exchange_found = True
		# at least one redeemed_tx is needed to do another step
		if step_dict:
			steps_list.append(step_dict)
			success = False
			while not success:
				try:
					return do_steps(steps_list, step+1, max_steps, use_cluster_ID, exchange_found=exchange_found)
					success = True
				except Exception as e:
					print(e, 'do_steps')
		else:
			return steps_list
	else:
		return steps_list


def flow2known_clusters(flow):
	# flow is a list of dicts
	# returns:
	# a set of clusterID_tag, where tag is not None
	# a set of tags
	# a dict with clusterID_tag as key and a dict as value with 
	# steps as key and a dict as value 
	# with ints (representing the steps at which tag appears) as key and received BTC at that step as value
	tag_steps = dict()
	known_clusters_tags = set()
	tags = set()
	for step in range(len(flow)):
		for k in flow[step].keys():
			a, c, tag = k.split('_')
			if tag != 'unknown':
				known_clusters_tags.add(c + '_' + tag)
				tags.add(tag)
				cluster_tag = c + '_' + tag
				if cluster_tag not in tag_steps.keys():
					tag_steps[cluster_tag] = dict()
				if step not in tag_steps[cluster_tag]:
					tag_steps[cluster_tag][step] = 0
				btc = 0
				for i in flow[step][k]:
					btc += i['received_BTC']
					tag_steps[cluster_tag][step] = btc
	return known_clusters_tags, tags, tag_steps


def save_payment_flow(steps_list, h, tx_hash, use_cluster_ID=False):
	file_path = base_path + h + '_' + tx_hash + '.json'
	with open(file_path, 'w') as fp:
		json.dump(steps_list, fp)


def continue_payment_flow(address, tx_hash, use_cluster_ID, max_steps):
	# get the json
	file_path = base_path + address + '_' + tx_hash + '.json'
	with open(file_path, 'r') as fp:
		steps_list = json.load(fp)
	initial_len = len(steps_list)
	if initial_len-1 < max_steps:
		success = False
		while not success:
			try:
				steps_list = do_steps(steps_list, initial_len, max_steps, use_cluster_ID)
				success = True
			except Exception as e:
				print(e, 'continue_payment_flow')
		save_payment_flow(steps_list, address, tx_hash, use_cluster_ID)
		return steps_list
	else:
		return steps_list[:max_steps]


def get_payment_flow(address, tx_hash, max_steps=0, use_cluster_ID=False, follow_address=''):
	# max_steps: int specifying steps from the payment, if 0 we try to use all the already available steps
	# use_cluster_ID: False to see payment flowing through addresses, True for clusters
	# return a list with a tree-like structure, each element is a dict representing a step	 
	# if max_steps is not null, we get all the steps till max_steps, resuming from file or starting from scratch
	if max_steps:
		try:
			# check if the flow is already available
			return continue_payment_flow(address, tx_hash, use_cluster_ID=use_cluster_ID, max_steps=max_steps)
		except:
			# if it is not already available, start from scratch
			steps_list = [] # index i for step i, each element is a step_dict
			success = False
			while not success:
				block_hash = get_block_hash(get_tx(tx_hash)['height'])
				block = bx.get_block(block_hash)
				tx_index = [tx.tx_index for tx in block.transactions if tx.hash == tx_hash][0] # payment
				if follow_address:
					url = 'https://blockchain.info/tree/' + str(tx_index) + '?format=json'
					d = url2dict(url)
					follow_found = False
					for child in d['children']:
						if 'name' in child.keys() and child['name'] == follow_address and 'redeemed_tx' in child.keys():
							tx_index = child['redeemed_tx'][0]
							follow_found = True
					if not follow_found:
						return steps_list
				success = True
			tree_url = 'https://blockchain.info/tree/' + str(tx_index) + '?format=json'
			tree = url2dict(tree_url)
			step_dict = dict() # addresses as keys, list of [BTC, redeemed_tx] as value

			node_name = address[:7] + '_0'
			# for each spent output get indeces of next level
			step_dict, exchange_found = tree2step_dict(step_dict, tree, 0, address, use_cluster_ID) # step from payment to output addresses
			steps_list.append(step_dict)

			success = False
			while not success:
				try:
					steps_list = do_steps(steps_list, 1, max_steps, use_cluster_ID, exchange_found=exchange_found)
					success = True
				except Exception as e:
					print(e, 'get_payment_flow')
			save_payment_flow(steps_list, address, tx_hash, use_cluster_ID)
			return steps_list
	# we just get the steps we have
	# if we fail, is because the file does not exist and we must specify max_steps 
	else:
		file_path = base_path + '_' +  str(address) + '.json'
		with open(file_path, 'r') as fp:
			steps_list = json.load(fp)
		return steps_list


def get_block_hash(height):
	url = api_path + 'block/' + str(height)
	return url2dict(url)['blockHash']

