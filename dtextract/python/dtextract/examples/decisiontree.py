from runCompare import *
from ..network import *

DT_DATA_TYPES = [NUM, NUM, NUM, NUM, CAT_RES]

DT_OUTPUT = 'dt.log'

from sklearn.tree import _tree

def tree_to_code(tree, feature_names, savepath):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    f = open (savepath, "w")
    f.write ("def tree({}):\n".format(", ".join(feature_names)))

    def recurse(node, depth):
        indent = "  " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            f.write ("{}if {} <= {}:\n".format(indent, name, threshold))
            recurse(tree_.children_left[node], depth + 1)
            f.write ("{}else:  # if {} > {}\n".format(indent, name, threshold))
            recurse(tree_.children_right[node], depth + 1)
        else:
            #print "{}return {}".format(indent, tree_.value[node])
            f.write ("{}return {}\n".format(indent, np.argmax(tree_.value[node][0])))

    recurse(0, 1)
    f.close()

class DecisionTree:
	def __init__(self):
		self.dtExtract = None
		self.greedyDT = None

	def interprete(self, savepath):
		tree_to_code(self.greedyDT, ["x", "dx", "theta", "dtheta"], savepath)

	def getFunc(self, rf):
		return lambda xs: rf.predict(xs)

	# Learning a decision tree abstraction of a neural network
	def learn(self, datapath, netpath):
		hasHeader = False
		isClassify = True
		nDataMatrixCols = None
		distType = "CategoricalGaussianMixture"
		setCurOutput(DT_OUTPUT)

		# Step 1: Load and Set up
		log('Parsing CSV...', INFO)
		(df, res, resMap, catFeats) = readCsv(datapath, hasHeader, DT_DATA_TYPES)
		log('Done!', INFO)

		log('Splitting into training and test...', INFO)
		(trainDf, testDf) = split(df, trainingProp)
		log('Done!', INFO)

		log('Constructing data matrices...', INFO)
		(XTrain, yTrain, catFeatIndsTrain, numericFeatIndsTrain) = constructDataMatrix(trainDf, res, catFeats)
		(XTest, yTest, catFeatIndsTest, numericFeatIndsTest) = constructDataMatrix(testDf, res, catFeats)
		log('Done!', INFO)

		log('Loading neural net...', INFO)
		#rfConstructor = MLPClassifier if isClassify else MLPRegressor
		#rf = rfConstructor(solver='lbfgs', hidden_layer_sizes=(hiddenSize,))
		#rf.fit(XTrain, yTrain)
		rf = load(netpath)
		log('Done!', INFO)

		rfFunc = self.getFunc(rf)

		rfScoreFunc = f1Vec if isClassify else mseVec

		rfTrainScore = rfScoreFunc(rfFunc, XTrain, yTrain)
		rfTestScore = rfScoreFunc(rfFunc, XTest, yTest)

		log('Training score: ' + str(rfTrainScore), INFO)
		log('Test score: ' + str(rfTestScore), INFO)


		# Step 2: Distribution
		#if distType == 'CategoricalGaussianMixture':
		#	dist = CategoricalGaussianMixtureDist(XTrain, catFeatIndsTrain, numericFeatIndsTrain, nComponents)
		#else:
		#	raise Exception('Invalid distType: ' + distType)


		# Step 3: Sampling
		#log('Sampling ' + str(100000) + ' points', INFO)
		#xs = dist.sample([], 100000)
		#ys = rfFunc(xs)
		#log('Done! Sampled ' + str(len(xs)) + ' points', INFO)

		# Step 3: If no points sampled, return a dummy leaf
		#if len(xs) == 0:
		#	log('No points!', INFO)
		#	return (None, 0.0, LeafNode(0.0), 0.0)	

	    # Step 4: Train a (greedy) decision tree
		scoreFunc = f1 if isClassify else mse
		log('Training greedy decision tree', INFO)
		maxLeaves = (maxDtSize + 1)/2
		dtConstructor = DecisionTreeClassifier if isClassify else DecisionTreeRegressor
		self.greedyDT = dtConstructor(max_leaf_nodes=maxLeaves)
		self.greedyDT.fit(XTrain, yTrain)
		log('Done!', INFO)
		log('Node count: ' + str(self.greedyDT.tree_.node_count), INFO)

		#tree_to_code(self.greedyDT, ["x1", "x2", "x3", "x4"])

		#dtTrainRelTrainScore = scoreFunc(lambda x: self.greedyDT.predict(x.reshape(1, -1)), XTrain, rf.predict(XTrain))
		#dtTrainRelTestScore = scoreFunc(lambda x: self.greedyDT.predict(x.reshape(1, -1)), XTest, rf.predict(XTest))
		dtTrainRelTrainScore = scoreFunc(lambda x: self.greedyDT.predict(x.reshape(1, -1)), XTrain, yTrain)
		dtTrainRelTestScore = scoreFunc(lambda x: self.greedyDT.predict(x.reshape(1, -1)), XTest, yTest)

		log('Relative training score: ' + str(dtTrainRelTrainScore), INFO)
		log('Relative test score: ' + str(dtTrainRelTestScore), INFO)

		dtTrainTrainScore = scoreFunc(lambda x: self.greedyDT.predict(x.reshape(1, -1)), XTrain, yTrain)
		dtTrainTestScore = scoreFunc(lambda x: self.greedyDT.predict(x.reshape(1, -1)), XTest, yTest)

		log('Training score: ' + str(dtTrainTrainScore), INFO)
		log('Test score: ' + str(dtTrainTestScore), INFO)

		return [rfTrainScore, rfTestScore,
			dtTrainRelTrainScore, dtTrainRelTestScore, dtTrainTrainScore, dtTrainTestScore]


	def synthesize(self, datapath, netpath):
		hasHeader = False
		isClassify = True
		nDataMatrixCols = None
		distType = "CategoricalGaussianMixture"
		setCurOutput(DT_OUTPUT)

		# Step 1: Load and Set up
		log('Parsing CSV...', INFO)
		(df, res, resMap, catFeats) = readCsv(datapath, hasHeader, DT_DATA_TYPES)
		log('Done!', INFO)

		log('Splitting into training and test...', INFO)
		(trainDf, testDf) = split(df, trainingProp)
		log('Done!', INFO)

		log('Constructing data matrices...', INFO)
		(XTrain, yTrain, catFeatIndsTrain, numericFeatIndsTrain) = constructDataMatrix(trainDf, res, catFeats)
		(XTest, yTest, catFeatIndsTest, numericFeatIndsTest) = constructDataMatrix(testDf, res, catFeats)
		log('Done!', INFO)

		log('Loading neural net...', INFO)
		#rfConstructor = MLPClassifier if isClassify else MLPRegressor
		#rf = rfConstructor(solver='lbfgs', hidden_layer_sizes=(hiddenSize,))
		#rf.fit(XTrain, yTrain)
		rf = load(netpath)
		log('Done!', INFO)

		rfFunc = self.getFunc(rf)

		rfScoreFunc = f1Vec if isClassify else mseVec

		rfTrainScore = rfScoreFunc(rfFunc, XTrain, yTrain)
		rfTestScore = rfScoreFunc(rfFunc, XTest, yTest)

		log('Training score: ' + str(rfTrainScore), INFO)
		log('Test score: ' + str(rfTestScore), INFO)

		# Step 2: Set up decision tree extraction inputs
		paramsLearn = ParamsLearn(tgtScore, minGain, maxSize)
		paramsSimp = ParamsSimp(nPts, nTestPts, isClassify)

		# Step 3: Distribution
		if distType == 'CategoricalGaussianMixture':
			dist = CategoricalGaussianMixtureDist(XTrain, catFeatIndsTrain, numericFeatIndsTrain, nComponents)
		else:
			raise Exception('Invalid distType: ' + distType)

		# Step 4: Extract decision tree
		self.dtExtract = learnDTSimp(genAxisAligned, rfFunc, dist, paramsLearn, paramsSimp)

		log('Decision tree:', INFO)
		log(str(self.dtExtract), INFO)
		log('Node count: ' + str(self.dtExtract.nNodes()), INFO)

		scoreFunc = f1 if isClassify else mse

		dtExtractRelTrainScore = scoreFunc(self.dtExtract.eval, XTrain, rf.predict(XTrain))
		dtExtractRelTestScore = scoreFunc(self.dtExtract.eval, XTest, rf.predict(XTest))

		log('Relative training score: ' + str(dtExtractRelTrainScore), INFO)
		log('Relative test score: ' + str(dtExtractRelTestScore), INFO)

		dtExtractTrainScore = scoreFunc(self.dtExtract.eval, XTrain, yTrain)
		dtExtractTestScore = scoreFunc(self.dtExtract.eval, XTest, yTest)

		log('Training score: ' + str(dtExtractTrainScore), INFO)
		log('Test score: ' + str(dtExtractTestScore), INFO)

		# Step 5: Train a (greedy) decision tree
		log('Training greedy decision tree', INFO)
		maxLeaves = (maxDtSize + 1)/2
		dtConstructor = DecisionTreeClassifier if isClassify else DecisionTreeRegressor
		self.greedyDT = dtConstructor(max_leaf_nodes=maxLeaves)
		#self.greedyDT.fit(XTrain, rfFunc(XTrain))
		self.greedyDT.fit(XTrain, yTrain)
		log('Done!', INFO)
		log('Node count: ' + str(self.greedyDT.tree_.node_count), INFO)

		#dtTrainRelTrainScore = scoreFunc(lambda x: self.greedyDT.predict(x.reshape(1, -1)), XTrain, rf.predict(XTrain))
		#dtTrainRelTestScore = scoreFunc(lambda x: self.greedyDT.predict(x.reshape(1, -1)), XTest, rf.predict(XTest))
		dtTrainRelTrainScore = scoreFunc(lambda x: self.greedyDT.predict(x.reshape(1, -1)), XTrain, yTrain)
		dtTrainRelTestScore = scoreFunc(lambda x: self.greedyDT.predict(x.reshape(1, -1)), XTest, yTest)

		log('Relative training score: ' + str(dtTrainRelTrainScore), INFO)
		log('Relative test score: ' + str(dtTrainRelTestScore), INFO)

		dtTrainTrainScore = scoreFunc(lambda x: self.greedyDT.predict(x.reshape(1, -1)), XTrain, yTrain)
		dtTrainTestScore = scoreFunc(lambda x: self.greedyDT.predict(x.reshape(1, -1)), XTest, yTest)

		log('Training score: ' + str(dtTrainTrainScore), INFO)
		log('Test score: ' + str(dtTrainTestScore), INFO)

		#return [dtTrainRelTrainScore, dtTrainRelTestScore, dtTrainTrainScore, dtTrainTestScore]
		return [rfTrainScore, rfTestScore,
			dtExtractRelTrainScore, dtExtractRelTestScore, dtExtractTrainScore, dtExtractTestScore,
			dtTrainRelTrainScore, dtTrainRelTestScore, dtTrainTrainScore, dtTrainTestScore]

	def predict(self, data):	
		if (self.greedyDT == None):
			return 0
		else:
			data = np.asarray(data)
			data = data.reshape(1,-1)
			value = self.greedyDT.predict(data)
			value = value.item(0)
			return value
		# if (self.dtExtract == None):
		# 	return 0
		# else:
		# 	return self.dtExtract.eval (np.array(np.asarray(data)))

if __name__ == "__main__":
	print ("Loaded!")
	#tree = DecisionTree()
	#tree.synthesize('agent330.network.data', 'agent330.network.json')
	#val = tree.predict((100.0, 100.0, 100.0, 100.0))
	#print ('prediction: %d' % val)