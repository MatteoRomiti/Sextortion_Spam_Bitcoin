import requests
import json
from config import APITOKEN

api_path = 'https://api.graphsense.info/'
headers = {'accept': 'application/json', 'Authorization': APITOKEN}


def url2dict(url):
    r = requests.get(url, headers=headers)
    d = json.loads(r.text)
    return d


##############################
# api calls
##############################


def get_address_info(address, currency='btc'):
    url = api_path + currency + '/' + 'address/' + address
    return url2dict(url)


def get_address_clusterID(address, currency='btc'):
    address = str(address)
    url = api_path + currency + '/' + 'address/' + address + '/cluster'
    d = url2dict(url)
    if d and 'cluster' in d:
        clusterID = d['cluster']
        return clusterID
    return None


def get_address_tags(address, currency='btc'):
    address = str(address)
    url = api_path + currency + '/' + 'address/' + address + '/tags'
    d = url2dict(url)
    if d:
        tags = list(set([el['label'] for el in d]))
        return tags
    return []


def get_address_tag(address, currency='btc'):
    tag = 'unknown'
    tags = get_address_tags(address, currency=currency)
    if tags:
        tag = tags[0]
    return tag


def get_address_txs(address, currency='btc', limit=0):
    address = str(address)
    if not limit:
        limit = get_address_n_transactions(address, currency=currency)
        if not limit:
            return {}
    url = api_path + currency + '/' + 'address/' + address + '/transactions?limit=' + str(limit)
    d = url2dict(url)
    return [el['txHash'] for el in d['transactions']]


def get_address_txs_out(address, currency='btc', limit=0):
    # get hashes of txs where address is among inputs
    if not limit:
        limit = get_address_n_transactions(address, currency=currency)
    txs = get_address_txs(address, currency=currency, limit=limit)
    tx_hashes_out = set()
    for tx_hash in txs:
        addresses_in = get_tx_addresses_in(tx_hash, currency=currency)
        if address in addresses_in:
            tx_hashes_out.add(tx_hash)
    return tx_hashes_out


def get_address_txs_in(address, currency='btc', limit=0):
    # get hashes of txs where address is among outputs
    if not limit:
        limit = get_address_n_transactions(address, currency=currency)
    txs = get_address_txs(address, currency=currency, limit=limit)
    tx_hashes_in = set()
    for tx_hash in txs:
        addresses_out = get_tx_addresses_out(tx_hash, currency=currency)
        if address in addresses_out:
            tx_hashes_in.add(tx_hash)
    return tx_hashes_in


def get_address_neighbors_out(address, currency='btc', limit=0):
    # get all addresses receiving money from this address which means:
    # scan all edges and if source==address, select the target
    address = str(address)
    if not limit:
        limit = get_address_n_transactions_out(address, currency=currency)
        if not limit:
            return {}
    url = api_path + currency + '/' + 'address/' + address + '/neighbors?direction=out&limit=' + str(limit)
    d = url2dict(url)
    neighbors_out = set()
    if d:
        for e in d['edges']:
            if e['source'] == address:
                neighbors_out.add(e['target'])
    return neighbors_out


def get_address_neighbors_in(address, currency='btc', limit=0):
    # get all addresses sending money to this address which means:
    # scan all edges and if target==address, select the source
    address = str(address)
    if not limit:
        limit = get_address_n_transactions_in(address, currency=currency)
        if not limit:
            return {}
    url = api_path + currency + '/' + 'address/' + address + '/neighbors?direction=in&limit=' + str(limit)
    d = url2dict(url)
    neighbors_in = set()
    if d:
        for e in d['edges']:
            if e['target'] == address:
                neighbors_in.add(e['source'])
    return neighbors_in


def get_address_neighbors(address, currency='btc', limit=0):
    limit = int(limit/2)  # because we do a union later
    neighbors_in = get_address_neighbors_in(address, currency=currency, limit=limit)
    neighbors_out = get_address_neighbors_out(address, currency=currency, limit=limit)
    return neighbors_out.union(neighbors_in)


def get_address_money_in(address, currency='btc', coin='satoshi'):
    money = 0
    info = get_address_info(address, currency=currency)
    if 'totalReceived' in info.keys():
        money = info['totalReceived'][coin]
    return money


def get_address_money_out(address, currency='btc', coin='satoshi'):
    money = 0
    info = get_address_info(address, currency=currency)
    if 'totalSpent' in info.keys():
        money = info['totalSpent'][coin]
    return money


def get_address_n_transactions(address, currency='btc'):
    return get_address_n_transactions_out(address, currency=currency) + get_address_n_transactions_in(address, currency=currency)


def get_address_n_transactions_out(address, currency='btc'):
    n = 0
    info = get_address_info(address, currency=currency)
    if 'noOutgoingTxs' in info.keys():
        n = info['noOutgoingTxs']
    return n


def get_address_n_transactions_in(address, currency='btc'):
    n = 0
    info = get_address_info(address, currency=currency)
    if 'noIncomingTxs' in info.keys():
        n = info['noIncomingTxs']
    return n


def get_address_balance(address, currency='btc', coin='eur'):
    balance = 0
    info = get_address_info(address, currency=currency)
    if info:
        balance = info['balance'][coin]
    return balance


def get_address_received(address, currency='btc', coin='eur'):
    received = 0
    info = get_address_info(address, currency=currency)
    if info:
        received = info['totalReceived'][coin]
    return received


def get_cluster_info(cluster, currency='btc'):
    url = api_path + currency + '/' + 'cluster/' + str(cluster)
    return url2dict(url)


def get_cluster_addresses(cluster, currency='btc', limit=0):
    if not limit:
        limit = get_cluster_n_addresses(cluster, currency=currency)
        if not limit:
            return {}
    url = api_path + currency + '/' + 'cluster/' + str(cluster) + '/addresses?limit=' + str(limit)
    res = url2dict(url)
    if 'addresses' in res:
        return [el['address'] for el in res['addresses']]
    else:
        return res


def get_cluster_tags(cluster, currency='btc'):
    url = api_path + currency + '/' + 'cluster/' + str(cluster) + '/tags'
    d = url2dict(url)
    if d:
        tags = list(set([el['label'] for el in d]))
        return tags
    return 'Unknown'


def get_cluster_tag(cluster, currency='btc'):
    tag = 'Unknown'
    tags = get_cluster_tags(cluster, currency=currency)
    if tags:
        tag = tags[0]
    return tag


def get_cluster_n_neighbors_out(cluster, currency='btc'):
    n = 0
    info = get_cluster_info(cluster, currency=currency)
    if 'outDegree' in info.keys():
        n = info['outDegree']
    return n


def get_cluster_n_neighbors_in(cluster, currency='btc'):
    n = 0
    info = get_cluster_info(cluster, currency=currency)
    if 'inDegree' in info.keys():
        n = info['inDegree']
    return n


def get_cluster_neighbors_out(cluster, currency='btc', limit=0):
    # get all clusters receiving money from this cluster which means:
    # scan all edges and if source==cluster, select the target
    neighbors_out = set()
    if not limit:
        limit = get_cluster_n_transactions_out(cluster, currency=currency)
        if not limit:
            return neighbors_out
    url = api_path + currency + '/' + 'cluster/' + str(cluster) + '/neighbors?direction=out&limit=' + str(limit)
    d = url2dict(url)
    if d:
        for e in d['edges']:
            if e['source'] == cluster:
                neighbors_out.add(e['target'])
    return neighbors_out


def get_cluster_neighbors_in(cluster, currency='btc', limit=0):
    # get all clusters sending money to this cluster which means:
    # scan all edges and if target==cluster, select the source
    neighbors_in = set()
    if not limit:
        limit = get_cluster_n_transactions_out(cluster, currency=currency)
        if not limit:
            return neighbors_in
    url = api_path + currency + '/' + 'cluster/' + str(cluster) + '/neighbors?direction=in&limit=' + str(limit)
    d = url2dict(url)
    neighbors_in = set()
    if d:
        for e in d['edges']:
            if e['target'] == cluster:
                neighbors_in.add(e['source'])
    return neighbors_in


def get_cluster_neighbors(cluster, currency='btc', limit=0):
    limit = int(limit/2)
    neighbors_in = get_cluster_neighbors_in(cluster, currency=currency, limit=limit)
    neighbors_out = get_cluster_neighbors_out(cluster, currency=currency, limit=limit)
    return neighbors_out.union(neighbors_in)


def get_cluster_n_transactions(cluster, currency='btc'):
    return get_cluster_n_transactions_out(cluster, currency=currency) + get_cluster_n_transactions_out(cluster)


def get_cluster_n_transactions_out(cluster, currency='btc'):
    n = 0
    info = get_cluster_info(cluster, currency=currency)
    if 'noOutgoingTxs' in info.keys():
        n = info['noOutgoingTxs']
    return n


def get_cluster_n_transactions_in(cluster, currency='btc'):
    n = 0
    info = get_cluster_info(cluster, currency=currency)
    if 'noIncomingTxs' in info.keys():
        n = info['noIncomingTxs']
    return n


def get_cluster_balance(cluster, currency='btc', coin='eur'):
    balance = 0
    info = get_cluster_info(cluster, currency=currency)
    if info:
        balance = info['balance'][coin]
    return balance


def get_cluster_received(cluster, currency='btc', coin='eur'):
    received = 0
    info = get_cluster_info(cluster, currency=currency)
    if info:
        received = info['totalReceived'][coin]
    return received


def get_cluster_n_addresses(cluster, currency='btc'):
    n_addresses = 0
    info = get_cluster_info(cluster, currency=currency)
    if info:
        n_addresses = info['noAddresses']
    return n_addresses


def get_cluster_txs_timestamps(cluster, currency='btc'):
    # returns a list of transaction timestamps for a cluster (or address)
    tss = []
    # get all cluster addresses txs
    addresses = get_cluster_addresses(cluster , currency=currency, limit=get_cluster_info(cluster, currency=currency)['noAddresses'])
    # for each address get txs_ts and append them
    for address in addresses:
        txs = get_address_txs(address, currency=currency, limit=get_address_n_transactions(address, currency=currency)+1)
        tmp = [el['timestamp'] for el in txs]
        for t in tmp:
            tss.append(t)
    return tss

# TODO:
# def get_cluster_n_neighbors_in()
# def get_cluster_n_neighbors_out()
# def get_cluster_n_neighbors()
# def get_cluster_txs()
# def get_cluster_txs_out()
# def get_cluster_txs_in()


def get_tx(tx_hash, currency='btc'):
    url = api_path + currency + '/' + 'tx/' + tx_hash
    return url2dict(url)


def get_tx_addresses_in(tx_hash, currency='btc'):
    tx = get_tx(tx_hash, currency=currency)
    addresses_in = [el['address'] for el in tx['inputs']]
    return addresses_in


def get_tx_addresses_out(tx_hash, currency='btc'):
    tx = get_tx(tx_hash, currency=currency)
    addresses_out = [el['address'] for el in tx['outputs']]
    return addresses_out


def get_tx_clusters_in(tx_hash, currency='btc'):
    clusters_in = [get_address_clusterID(addr, currency=currency) for addr in get_tx_addresses_in(tx_hash, currency=currency)]
    return clusters_in


def get_tx_clusters_out(tx_hash, currency='btc'):
    clusters_out = [get_address_clusterID(addr, currency=currency) for addr in get_tx_addresses_out(tx_hash, currency=currency)]
    return clusters_out


def get_tx_tags_in(tx_hash, currency='btc'):
    tags_in = [get_address_tag(addr, currency=currency) for addr in get_tx_addresses_in(tx_hash, currency=currency)]
    return tags_in


def get_tx_tags_out(tx_hash, currency='btc'):
    tags_out = [get_address_tag(addr, currency=currency) for addr in get_tx_addresses_out(tx_hash, currency=currency)]
    return tags_out


def get_tx_values_out(tx_hash, currency='btc', coin='eur'):
    return [el['value'][coin] for el in get_tx(tx_hash, currency=currency)['outputs']]


def get_tx_values_in(tx_hash, currency='btc', coin='eur'):
    return [el['value'][coin] for el in get_tx(tx_hash, currency=currency)['inputs']]


def get_addresses_txs(addresses, currency='btc', direction='both'):
    addresses_txs = dict()  # key: address, value: list of txs
    ln = len(addresses)
    addresses = list(addresses)
    addresses.sort()
    for i, addr in enumerate(addresses):
        print(ln, i, addr, end='\r')
        if direction == 'in':
            addresses_txs[addr] = [get_tx(hsh, currency=currency) for hsh in get_address_txs_in(addr, currency=currency)]
        if direction == 'out':
            addresses_txs[addr] = [get_tx(hsh, currency=currency) for hsh in get_address_txs_out(addr, currency=currency)]
        if direction == 'both':
            addresses_txs[addr] = [get_tx(hsh, currency=currency) for hsh in get_address_txs(addr, currency=currency)]
    return addresses_txs


def get_block_hash(height, currency='btc'):
    url = api_path + currency + '/' + 'block/' + str(height)
    return url2dict(url)['blockHash']
