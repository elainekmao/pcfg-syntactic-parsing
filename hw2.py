import sys, json

words = {}
rare_words = list()
nonterminal_counts = {}
q_unary = {}
q_binary = {}

def find_rare (count_file):
	global rare_words
	words = {}
	#Dictionary to keep track of words and their counts
	counts_file = open(count_file, 'r')
	for line in counts_file:
		fields = line.split()
		#Only looks at UNARYRULE lines
		if fields[1] == 'UNARYRULE':
			count = int(fields[0])
			word = fields[3]
			#Checks to see if we already have the word in our dictionary
			#If so, increments the count
			if word in words:
				words[word] += count
			#If not, adds word to dictionary
			else:
				words[word] = count
	for key in words:
		#Checks to see if a word occurs less than 5 times
		#If so, adds to rare_words list
		if words[key] < 5:
			rare_words.append(key)
	counts_file.close()

def replace_rare(training_file):
	output_file = open('rare_replaced.dat', 'w')
	for line in open(training_file):
		tree = json.loads(line)
		#Calls helper function below on each tree
		replace_rare_words(tree)
		output_file.write(json.dumps(tree) + '\n')
	output_file.close()

#Helper function for replace_rare
def replace_rare_words(tree):
	tree[0] = str(tree[0])
	#If it is a binary rule
	if len(tree) == 3:
		#Recurses on next level of tree
		replace_rare_words(tree[1])
		replace_rare_words(tree[2])
	#If it is a unary rule
	elif len (tree) == 2:
		word = tree[1]
		#Checks to see if the word is in rare_words list
		#If so, replaces word in the tree with "_RARE_" symbol
		if word in rare_words:
			tree[1] = "_RARE_"
		#Otherwise does nothing with the word
		else:
			tree[1] = str(tree[1])
			tree[1] = tree[1]

#Helper function that gets called by q_calculator
#Calculates counts for nonterminals, stores in global dictionary
def nonterminal_calculator(counts_file):
	global nonterminal_counts
	f = open(counts_file, 'r')
	for line in f:
		l = line.split()
		count = l[0]
		tag = l[2]
		if l[1] == 'NONTERMINAL':
			nonterminal_counts[tag] = count

#Creates dictionary of q values for unary and binary rules
def q_calculator(counts_file):
	global q_unary, q_binary, words
	f = open(counts_file, 'r')
	nonterminal_calculator(counts_file)
	for line in f:
		l = line.split()
		count = l[0]
		root = l[2]
		rule = ' '.join(l[2:])
		if l[1] == 'UNARYRULE':
			word = l[3]
			if word in words:
				words[word] += count
			else:
				words[word] = count
			q_unary[rule] = float(count) / float(nonterminal_counts[root])
		elif l[1] == 'BINARYRULE':
			q_binary[rule] = float(count) / float(nonterminal_counts[root])
	f.close()

#Returns list of binary rules for a given nonterminal; used in CKY algorithm
def rules(nonterminal):
	rules = list()
	for key in q_binary.keys():
		rule = key.split()
		if rule[0] == nonterminal:
			rules.append(key)
	return rules

#Implementation of the CKY algorithm
def cky(counts_file, input_file):
	count_file = open(counts_file, 'r')
	f = open(input_file, 'r')
	output_file = open('prediction_file', 'w')
	for line in f:
		#Splits each sentence into a list of words
		sentence = line.split()
		#Calculates length of sentence
		n = len(sentence)
		#Defines argmax accessory function which is used to calculate backpointers
		def argmax(l):
			if not l:
				return None, 0
			else:
				return max(l, key = lambda x: x[1])
		#Sets start of sentence as index 1
		x = [''] + list(sentence)
		#Initializes pi and bp dictionaries
		pi = {}
		bp = {}
		#Initializes values for pi[i,i,X]
		for i in range(1, n+1):
			#For every nonterminal X, calculate
			for X in nonterminal_counts.keys():
				#If we have a q parameter for X -> x[i], use that
				if X + ' ' + x[i] in q_unary.keys():
					pi[i,i,X] = q_unary[X + ' ' + x[i]]
					bp[i,i,X] = (X, x[i])
				#If we have never seen this word before, treat it as a rare word
				elif x[i] not in words:
					#If we have a q parameter for X -> '_RARE_', use that
					if X + ' ' + '_RARE_' in q_unary.keys():
						pi[i,i,X] = q_unary[X + ' ' + '_RARE_']
						bp[i,i,X] = (X, '_RARE_')
					#If not, set pi[i,i,X] to 0
					else:
						pi[i,i,X] = 0
						bp[i,i,X] = None
				#For any other case, set pi[i,i,X] to 0
				else:
					pi[i,i,X] = 0
					bp[i,i,X] = None
		#Dynamic part of CKY algorithm
		for l in range(1, n):
			for i in range(1, n - l + 1):
				j = i + l
				for X in nonterminal_counts.keys():
					#Checks if there are rules of the form X -> Y Z
					if rules(X):
						backpointer, score = argmax([((rule, i, s, j), q_binary[rule]*lookup((i,s,rule.split()[1]), X, i, j)*lookup((s+1, j, rule.split()[2]), X, i, j)) for s in range(i,j) for rule in rules(X) if pi.get])
						#If so, finds max score over all rules for X and all possible values of s
						backpointer, score = argmax([((rule, i, s, j), q_binary[rule]*pi[i, s, rule.split()[1]]*pi[s+1, j, rule.split()[2]]) for s in range(i, j) for rule in rules(X) if pi.get((i,s, rule.split()[1]), 0) > 0 if pi.get((s+1, j, rule.split()[2]), 0) > 0])
						if score > 0: 
							bp[i,j,X], pi[i,j,X] = backpointer, score
		#Backtrace algorithm which is used to build the tree from the backpointers
		def backtrace(backpointer):
			#If bp value is empty
			if not backpointer:
				return None
			#If bp is a binary rule of form X -> Y Z, sets X as root and calls backtrace on Y and Z
			elif len(backpointer[0].split()) == 3:
				return [backpointer[0].split()[0], backtrace(bp[backpointer[1],backpointer[2],backpointer[0].split()[1]]), backtrace(bp[backpointer[2]+1, backpointer[3], backpointer[0].split()[2]])]
			#If bp is a unary rule, returns rule
			else:
				return backpointer
		#If there is a tree with S as the root, return that tree
		if pi.get((1, n, 'S'), 0) != 0:
			tree = backtrace(bp[1, n, 'S'])
			output_file.write(json.dumps(tree) + '\n')
		#If not, return the highest scoring tree 
		else: 
			max_X, score = argmax([(X, (pi[1, n, X])) for X in nonterminal_counts.keys() if pi.get((1,n,X), 0) > 0])
			tree = backtrace(bp[1, n, max_X])
			output_file.write(json.dumps(tree) + '\n')
	count_file.close()
	f.close()
	output_file.close()

#Implementation of the CKY algorithm with attempted memoization
def cky_memo(counts_file, input_file):
	count_file = open(counts_file, 'r')
	f = open(input_file, 'r')
	output_file = open('prediction_file_vert', 'w')
	for line in f:
		#Splits each sentence into a list of words
		sentence = line.split()
		#Calculates length of sentence
		n = len(sentence)
		#Defines argmax accessory function which is used to calculate backpointers
		def argmax(l):
			if not l:
				return None, 0
			else:
				return max(l, key = lambda x: x[1])
		#Sets start of sentence as index 1
		x = [''] + list(sentence)
		#Initializes pi and bp dictionaries
		pi = {}
		bp = {}
		#Initializes values for pi[i,i,X]
		for i in range(1, n+1):
			#For every nonterminal X, calculate
			for X in nonterminal_counts.keys():
				#If we have a q parameter for X -> x[i], use that
				if X + ' ' + x[i] in q_unary.keys():
					pi[i,i,X] = q_unary[X + ' ' + x[i]]
					bp[i,i,X] = (X, x[i])
				#If we have never seen this word before, treat it as a rare word
				elif x[i] not in words:
					#If we have a q parameter for X -> '_RARE_', use that
					if X + ' ' + '_RARE_' in q_unary.keys():
						pi[i,i,X] = q_unary[X + ' ' + '_RARE_']
						bp[i,i,X] = (X, '_RARE_')
					#If not, set pi[i,i,X] to 0
					else:
						pi[i,i,X] = 0
						bp[i,i,X] = None
				#For any other case, set pi[i,i,X] to 0
				else:
					pi[i,i,X] = 0
					bp[i,i,X] = None
		def lookup(pi_key):
			if pi_key in pi:
				return bp[pi_key], pi[pi_key]
			else:
				i = pi_key[0]
				j = pi_key[1]
				X = pi_key[2]
				#Checks if there are rules of the form X -> Y Z
				if rules(X):
					#If so, finds max score over all rules for X and all possible values of s
					backpointer, score = argmax([((rule, i, s, j), q_binary[rule]*lookup((i, s, rule.split()[1]))[1]*lookup((s+1, j, rule.split()[2]))[1]) for s in range(i, j) for rule in rules(X) if pi.get((i,s, rule.split()[1]), 0) > 0 if pi.get((s+1, j, rule.split()[2]), 0) > 0])
					if score > 0: 
						bp[i,j,X], pi[i,j,X] = backpointer, score				
		#Dynamic part of CKY algorithm
		for l in range(1, n):
			for i in range(1, n - l + 1):
				j = i + l
				for X in nonterminal_counts.keys():
					lookup((i,j,X))					
		#Backtrace algorithm which is used to build the tree from the backpointers
		def backtrace(backpointer):
			#If bp value is empty
			if not backpointer:
				return None
			#If bp is a binary rule of form X -> Y Z, sets X as root and calls backtrace on Y and Z
			elif len(backpointer[0].split()) == 3:
				return [backpointer[0].split()[0], backtrace(bp[backpointer[1],backpointer[2],backpointer[0].split()[1]]), backtrace(bp[backpointer[2]+1, backpointer[3], backpointer[0].split()[2]])]
			#If bp is a unary rule, returns rule
			else:
				return backpointer
		#If there is a tree with S as the root, return that tree
		if pi.get((1, n, 'S'), 0) != 0:
			tree = backtrace(bp[1, n, 'S'])
			output_file.write(json.dumps(tree) + '\n')
		#If not, return the highest scoring tree 
		else: 
			max_X, score = argmax([(X, (pi[1, n, X])) for X in nonterminal_counts.keys() if pi.get((1,n,X), 0) > 0])
			tree = backtrace(bp[1, n, max_X])
			output_file.write(json.dumps(tree) + '\n')
	count_file.close()
	f.close()
	output_file.close()