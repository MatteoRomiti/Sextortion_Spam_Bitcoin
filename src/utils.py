import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import json
from blockchain import blockexplorer as bx
import datetime
import numpy as np
from api_calls import *

base_path = '../data/payment_flows/'


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


##############################
# notebook functions
##############################


def plot_addresses_txs_in_wrapper(addresses, address_type, min_n_txs=5):
	plot_addresses_txs_in(get_addresses_txs(addresses, direction='in'), address_type, min_n_txs=min_n_txs)


def plot_addresses_txs_in(address_txs, address_type, min_n_txs=5):
	# https://stackoverflow.com/questions/52697594/python-matplotlib-how-to-change-color-of-each-lollipop-in-a-lollipop-plot-ax
	# address_txs is a dict with address as key and list of txs as value
	# get all tx timestamps and the net received BTC of an address
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


def tree2step_dict(step_dict, tree, origin, use_address_clusterID_tag=True, exchanges_flag_needed=True):
	exchange_found = False
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
	if exchanges_flag_needed:
		return step_dict, exchange_found
	return step_dict


def do_steps(steps_list, step, max_steps, nodes_limit=100, exchange_found=False):
	# nodes_limit is to avoid scraping tons of transactions when max_steps is too large
	if step <= max_steps and len(steps_list[step-1].keys()) < nodes_limit and not exchange_found:
		step_dict = dict()  # addresses as keys, list of [BTC, redeemed_tx] as value
		for node_name in steps_list[step-1].keys():
			# see how it spent BTC
			for tx in steps_list[step-1][node_name]:  # an addr or cluster can receive BTC multiple times from different addresses of the previous level
				tx_index = tx['redeemed_tx']
				if tx_index:
					tree_url = 'https://blockchain.info/tree/' + str(tx_index) + '?format=json'
					tree = url2dict(tree_url)
					# for each spent output get indices of next level
					step_dict, exchange_flag = tree2step_dict(step_dict, tree, step, node_name, exchanges_flag_needed=True)
					if exchange_flag or exchange_found:
						exchange_found = True
		# at least one redeemed_tx is needed to do another step
		if step_dict:
			steps_list.append(step_dict)
			success = False
			while not success:
				try:
					return do_steps(steps_list, step+1, max_steps, exchange_found=exchange_found)
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


def save_payment_flow(steps_list, h, tx_hash):
	file_path = base_path + h + '_' + tx_hash + '.json'
	with open(file_path, 'w') as fp:
		json.dump(steps_list, fp)


def continue_payment_flow(address, tx_hash, max_steps):
	# get the json
	file_path = base_path + address + '_' + tx_hash + '.json'
	with open(file_path, 'r') as fp:
		steps_list = json.load(fp)
	initial_len = len(steps_list)
	if initial_len-1 < max_steps:
		success = False
		while not success:
			try:
				steps_list = do_steps(steps_list, initial_len, max_steps)
				success = True
			except Exception as e:
				print(e, 'continue_payment_flow')
		save_payment_flow(steps_list, address, tx_hash)
		return steps_list
	else:
		return steps_list[:max_steps]


def get_payment_flow(address, tx_hash, max_steps=0, follow_address=''):
	# max_steps: int specifying steps from the payment, if 0 we try to use all the already available steps
	# return a list with a tree-like structure, each element is a dict representing a step
	# if max_steps is not null, we get all the steps till max_steps, resuming from file or starting from scratch
	if max_steps:
		try:
			# check if the flow is already available
			return continue_payment_flow(address, tx_hash, max_steps=max_steps)
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
			# for each spent output get indices of next level
			step_dict, exchange_found = tree2step_dict(step_dict, tree, 0, address) # step from payment to output addresses
			steps_list.append(step_dict)

			success = False
			while not success:
				try:
					steps_list = do_steps(steps_list, 1, max_steps, exchange_found=exchange_found)
					success = True
				except Exception as e:
					print(e, 'get_payment_flow')
			save_payment_flow(steps_list, address, tx_hash)
			return steps_list
	# we just get the steps we have
	# if we fail, is because the file does not exist and we must specify max_steps 
	else:
		file_path = base_path + '_' +  str(address) + '.json'
		with open(file_path, 'r') as fp:
			steps_list = json.load(fp)
		return steps_list
