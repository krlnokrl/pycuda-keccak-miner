# Cuda keccak256 miner for siricoin
# Author: krlnokrl 
# keccak.cu from Matt Zweil & The Mochimo Core Contributor Team https://github.com/mochimodev/cuda-hashing-algos

# Benchmark:	
# GTX1070 - 145MH/s

import time, json, sha3
from web3.auto import w3
from eth_account.account import Account
from eth_account.messages import encode_defunct
import requests

import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import numpy


# helper functions
def refreshAccountInfo(acc):
	accountInfo_url = NodeAddr + "accounts/accountInfo/" + acc
	temp_txs = requests.get(accountInfo_url).json().get("result")
	_txs = temp_txs.get("transactions")
	lastSentTx = _txs[len(_txs)-1]
	balance = temp_txs.get("balance")
	return(lastSentTx)

def signTransaction(private_key, transaction):
	message = encode_defunct(text=transaction["data"])
	transaction["hash"] = w3.soliditySha3(["string"], [transaction["data"]]).hex()
	_signature = w3.eth.account.sign_message(message, private_key=private_key).signature.hex()
	signer = w3.eth.account.recover_message(message, signature=_signature)
	sender = w3.toChecksumAddress(json.loads(transaction["data"])["from"])
	if (signer == sender):
		transaction["sig"] = _signature
	return transaction

def submit_block(nonce, proof):
	txid = "None"
	priv_key = w3.solidityKeccak(["string", "address"], ["SiriCoin Will go to MOON - Just a disposable key", rewardsRecipient])
	
	acct = w3.eth.account.from_key(priv_key)
		
	blockData={"miningData" : {"miner": rewardsRecipient,"nonce": nonce,"difficulty": difficulty,"miningTarget": target,"proof": proof}, "parent": lastBlock,"messages": messages.hex(), "timestamp": timestamp, "son": "0000000000000000000000000000000000000000000000000000000000000000"}
	
	lastSentTx = refreshAccountInfo(acct.address)
	data = json.dumps({"from": acct.address, "to": acct.address, "tokens": 0, "parent": lastSentTx, "blockData": blockData, "epoch": lastBlock, "type": 1})
	tx = {"data": data}
	tx = signTransaction(priv_key, tx)
	
	tmp_get = requests.get(f"{send_url}{json.dumps(tx).encode().hex()}")
	#print(f"{send_url}{json.dumps(tx).encode().hex()}")
	# notify all nodes so we don't miss on time
	for node in nodes_notify:
		requests.get(f"{node}/send/rawtransaction/?tx={json.dumps(tx).encode().hex()}")



NodeAddr = "https://node-1.siricoin.tech:5006/"
nodes_notify = ["https://node-1.siricoin.tech:5006/"]


#NodeAddr = "http://127.0.0.1:5005/"
#nodes_notify=[]


# initialize the network data
rewardsRecipient = '0x2a5c2fA213b2ba385C0614248B1C705e77480f3A'
send_url = NodeAddr + "send/rawtransaction/?tx="
block_url = NodeAddr + "chain/miningInfo"
target = 0
last_block = 0

info = requests.get(block_url).json().get("result")
target = info["target"]
difficulty = info["difficulty"]
lastBlock = info["lastBlockHash"]
timestamp = int(time.time())
messages = b"null"
messagesHash = w3.keccak(messages)

bRoot = w3.soliditySha3(["bytes32", "uint256", "bytes32","address"], [lastBlock, timestamp, messagesHash, rewardsRecipient])



# create cuda kernel

f = open("keccak.cu", "r")
sha3_file = f.read()


code = '''

// cuda kernel for parallel mining
// in indata - the block root hash
// out outdata - a valid nonce
// in target - 8 byte prefix of target hash
// in seed - 2 byte part of nonce
// out found - 1 int (0 if no solution found, 1 if a solution was found)

__global__ void mine_keccak(BYTE* const indata, BYTE* outdata, BYTE* const target, int* const seed, int* found)
{
	//Thread id as seed
	const int Idx = threadIdx.x + blockDim.x * blockIdx.x;
	
	int i;
	
	BYTE hash[32]={0};
	BYTE nonce[32] = {0};
	BYTE* out = outdata;
	
	// build nonce
	nonce[26] = seed[0]>>8;
	nonce[27] = seed[0]%256;
	nonce[28] = Idx>>16;
	nonce[29] = Idx>>8;
	nonce[30] = Idx>>4;
	nonce[31] = Idx%256;
	
	//create and execute hash context
    CUDA_KECCAK_CTX ctx;
    cuda_keccak_init(&ctx, 256);
    cuda_keccak_update(&ctx, indata, 32);
	cuda_keccak_update(&ctx, nonce, 32);
    cuda_keccak_final(&ctx, hash);
	
	
	//if a solution exists do not override it
	if(found[0]==1){
		return;
	}
	
	//check if has is lower than nonce fist 8 bytes
	#pragma unroll 7
	for(i=0; i<7;i++){
		if(hash[i]<target[i]){
			found[0] = 1;
			out[0] = 0;
			out[1] = 0;
			out[2] = nonce[26];
			out[3] = nonce[27];
			out[4] = nonce[28];
			out[5] = nonce[29];
			out[6] = nonce[30];
			out[7] = nonce[31];
			return;
		}
		if(hash[i]>target[i]){
			return;
		}
	}
	
}

'''
mod = SourceModule(sha3_file+code, options=['-O3'])


# allocate memory
a= numpy.frombuffer(bRoot)
b=numpy.array(b'a'*8)

seed_a = numpy.array([0])
found = numpy.array([0])

a_gpu = cuda.mem_alloc(a.nbytes)
b_gpu = cuda.mem_alloc(b.nbytes)
target_gpu = cuda.mem_alloc(b.nbytes)
seed_gpu = cuda.mem_alloc(seed_a.nbytes)
found_gpu = cuda.mem_alloc(found.nbytes)

# pycuda function
func = mod.get_function('mine_keccak')
func.prepare("PPPPP")

#main mining loop
while True:

	#request latest block data
	
	info = requests.get(block_url).json().get("result")
	target = info["target"]
	difficulty = info["difficulty"]
	lastBlock = info["lastBlockHash"]
	timestamp = int(time.time())
	messages = b"null"
	messagesHash = w3.keccak(messages)
	
	bRoot = w3.soliditySha3(["bytes32", "uint256", "bytes32","address"], [lastBlock, timestamp, messagesHash, rewardsRecipient])

	print(f"Root: {bRoot.hex()}")
	
	
	a= numpy.frombuffer(bRoot)
	
	target_bin = (int(target, 16)).to_bytes(32, byteorder='big')
	target_arr = numpy.array([target_bin[:8]])
	
	print(f"GPU Target: {target_arr.data.hex()}")

	#copy root and target to the device
	cuda.memcpy_htod(a_gpu, a)
	cuda.memcpy_htod(target_gpu, target_arr)
	
	seed = 0 
	start = time.time()
	
	# perform cuda loops for 15s before updating block data
	while(time.time()<start+15):
		seed += 1
		
		# empty solution arrays
		found = numpy.array([0])
		b=numpy.array(b'a'*8)
		
		#copy seed to device and init found 0
		cuda.memcpy_htod(seed_gpu, numpy.array([seed]))
		cuda.memcpy_htod(found_gpu, found)
		
		# execute kernel on device; type 1D array, size must be tuned for each device type
		func.prepared_call((1024*512,1,1), (32,1,1), a_gpu, b_gpu, target_gpu, seed_gpu, found_gpu)
		
		# check if device has found asolution
		cuda.memcpy_dtoh(found, found_gpu)
		if(found[0] > 0):
			# copy solution nonce form device and validate it on the cpu
			cuda.memcpy_dtoh(b, b_gpu)
			print(f"NONCE: {b.tobytes().hex()}")
			ctx_proof = sha3.keccak_256()
			ctx_proof.update(bRoot)
			ctx_proof.update((0).to_bytes(24, byteorder='big'))
			ctx_proof.update(b.tobytes())
			bProof = ctx_proof.digest()
			print(f"CPU: {bProof.hex()}")
			
			# if the solution is valid, submit it
			if (int.from_bytes(bProof, "big") < int(target, 16)):
				print("=======Valid Block=======")
				nonce = int.from_bytes(b.tobytes(), 'big')
				print (nonce)
				submit_block(nonce, "0x" + bProof.hex())
		# slow down the cuda device by waiting between loops 
		#time.sleep(0.5)
	# reporting kernels*kernelsize/time = hashrate
	print(f"{round(seed*32*1024*512/(time.time()-start)/1000000, 2)} MH/s  {round(time.time()-start,2)}s/round")




