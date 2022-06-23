# Cuda keccak256 miner for siricoin
# Author: krlnokrl 

# Donate to 0x2a5c2fA213b2ba385C0614248B1C705e77480f3A
# siri / eth / matic / etc.

# Benchmark:	
# GTX1070 - 250MH/s

import time, json, sha3,random
from web3.auto import w3
from eth_account.account import Account
from eth_account.messages import encode_defunct
import requests

import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import numpy


rewardsRecipient = '0x2a5c2fA213b2ba385C0614248B1C705e77480f3A'


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
		
	if (tmp_get.status_code != 500 ):
		txid = tmp_get.json().get("result")[0]
	print(f"Block accepted in tx: {txid}")

# best settings for GTX 1070 at 256 Threads/block - vhigh grid
CUDA_BLOCK_SIZE = 256	# should be fine in most cases
CUDA_GRID_SIZE = 2**22 # 2**19 for weaker gpu and up to 2**24

#node values
NodeAddr = "http://138.197.181.206:5005/"
nodes_notify = ["http://138.197.181.206:5005/", "https://node-1.siricoin.tech:5006"]
send_url = NodeAddr + "send/rawtransaction/?tx="
block_url = NodeAddr + "chain/miningInfo"

#global block variabless
info = requests.get(block_url).json().get("result")
target = info["target"]
difficulty = info["difficulty"]
lastBlock = info["lastBlockHash"]
messages = b"null"
messagesHash = w3.keccak(messages)
timestamp  = time.time()


def mine():
	with open("keccak.cu", "r") as f:
		sha3_file = f.read()
	#kernel code
	code = '''

	// cuda kernel for parallel mining
	// in indata - the block root hash
	// out outdata - a valid nonce
	// in target - 8 byte prefix of target hash
	// in seed - 2 byte part of nonce
	// out found - 1 int (0 if no solution found, 1 if a solution was found)

	//global context for precomputed hash
	__device__ static CUDA_KECCAK_CTX ctx;

	// precompute hash state
	__global__ void init_miner(BYTE* const indata){

		BYTE zero[24] = {0};
		cuda_keccak_init(&ctx);
		cuda_keccak_update(&ctx, indata, 32);
		cuda_keccak_update(&ctx, zero, 24);
	}

	__global__ void mine_keccak(BYTE* outdata, BYTE* const target, const uint16_t seed, uint16_t* found)
	{
		//Thread id as seed
		const uint32_t Idx = threadIdx.x + blockDim.x * blockIdx.x;
		
		int i;
		
		BYTE hash[8]={0};
		BYTE nonce[8] = {0};
		BYTE* out = outdata;
		
		// build nonce
		nonce[0] = seed>>16;
		nonce[1] = seed>>8;
		nonce[2] = seed%256;
		nonce[3] = Idx>>24;
		nonce[4] = Idx>>16;
		nonce[5] = Idx>>8;
		nonce[6] = Idx>>4;
		nonce[7] = Idx%256;
		
		//create and execute hash context
		//copy global context that contains precomputed state
		CUDA_KECCAK_CTX ctx_ = ctx;
		
		// append nonce and compute hash
		cuda_keccak_update(&ctx_, nonce, 8);
		cuda_keccak_final(&ctx_, hash);
		
		
		//if a solution exists do not override it
		if(found[0]==1){
			return;
		}
		
		//check if has is lower than nonce fist 8 bytes
		#pragma unroll 7
		for(i=0; i<7;i++){
			if(hash[i]<target[i]){
				break;
			}
			if(hash[i]>target[i]){
				return;
			}
		}
		
		found[0] = 1;
		out[0] = nonce[0];
		out[1] = nonce[1];
		out[2] = nonce[2];
		out[3] = nonce[3];
		out[4] = nonce[4];
		out[5] = nonce[5];
		out[6] = nonce[6];
		out[7] = nonce[7];
		
	}

	'''
	mod = SourceModule(sha3_file+code, options=['-O3'])
	
	#allocate memory on device
	inputBuffer = numpy.frombuffer(b'a'*32)
	nonceBuffer = numpy.array(b'a'*8)
	found = numpy.array([0])
	found.astype(numpy.uint32)

	input_gpu = cuda.mem_alloc(inputBuffer.nbytes)
	nonce_gpu = cuda.mem_alloc(nonceBuffer.nbytes)
	target_gpu = cuda.mem_alloc(nonceBuffer.nbytes)
	found_gpu = cuda.mem_alloc(found.nbytes)
	
	#prepare function calls
	gpu_mine_init = mod.get_function('init_miner')
	gpu_mine_init.prepare("P")

	gpu_mine = mod.get_function('mine_keccak')
	gpu_mine.prepare("PPPP")
	
	
	#import global block data
	global target
	global difficulty
	global lastBlock
	global messages
	global messagesHash
	global timestamp
	
	#main mining loop
	while True:
		#get data
		info = requests.get(block_url).json().get("result")
		target = info["target"]
		difficulty = info["difficulty"]
		lastBlock = info["lastBlockHash"]
		messages = b"null"
		messagesHash = w3.keccak(messages)
		timestamp = int(time.time())
		bRoot = w3.soliditySha3(["bytes32", "uint256", "bytes32","address"], [lastBlock, timestamp, messagesHash, rewardsRecipient])

		inputBuffer = numpy.frombuffer(bRoot)
		target_bin = int(target, 16).to_bytes(32, byteorder='big')
		target_arr = numpy.array([target_bin[:8]])
		
		print(f"GPU Target: {target_arr.data.hex()}")
		#copy root and target to the device
		cuda.memcpy_htod(input_gpu, inputBuffer)
		cuda.memcpy_htod(target_gpu, target_arr)
		
		gpu_mine_init.prepared_call((1,1,1), (1,1,1), input_gpu)
		
		start = time.time()
		seed = 0
		# perform cuda loops for 20s before updating block data
		while(time.time()<start+20):
			seed += 1
			
			# empty solution arrays
			found[0] = 0
			nonceBuffer=numpy.array(b'0'*8)
			
			#copy seed to device and init found 0
			#cuda.memcpy_htod(seed_gpu, numpy.array([seed]))
			cuda.memcpy_htod(found_gpu, found)
			
			# execute kernel on device; type 1D array, size must be tuned for each device type
			gpu_mine.prepared_call((CUDA_GRID_SIZE,1,1), (CUDA_BLOCK_SIZE,1,1), nonce_gpu, target_gpu, numpy.uint16(seed), found_gpu)
			
			# check if device has found asolution
			cuda.memcpy_dtoh(found, found_gpu)
			if(found[0] > 0):
				# copy solution nonce form device and validate it on the cpu
				cuda.memcpy_dtoh(nonceBuffer, nonce_gpu)
				
				ctx_proof = sha3.keccak_256()
				ctx_proof.update(bRoot)
				ctx_proof.update((0).to_bytes(24, byteorder='big'))
				ctx_proof.update(nonceBuffer.tobytes())
				bProof = ctx_proof.digest()
				print("====Solution====")
				print(f"FoundBlockHash: {bProof.hex()}")
				# if the solution is valid, submit it
				if (int.from_bytes(bProof, "big") < int(target, 16)):
					seed = 0
					print("=Valid Block=")
					print("Submitting ... ")
					nonce = int.from_bytes(nonceBuffer.tobytes(), 'big')
						
					submit_block(nonce, "0x" + bProof.hex())					

					print("Sleeping for 1s")
					print("====		====")
					time.sleep(1)
					break;
					
			# slow down the cuda device by waiting between loops if you are mining too fast
			#time.sleep(0.9)
		# reporting kernels*kernelsize/time = hashrate
		print(f"{round(seed*CUDA_BLOCK_SIZE*CUDA_GRID_SIZE/(time.time()-start)/1000000, 2)} MH/s  in the last {round(time.time()-start,2)}s")


if __name__ == '__main__':

	print(f"""
				______________________________
				||__________________________||
				||    Siricoin GPU Miner    || 
				||     	 by krlnokrl        ||
				||__________________________||
				|____________________________|
						""")
	print("			Donate to 0x2a5c2fA213b2ba385C0614248B1C705e77480f3A")
	print("")
	mine()