import N
from P import Parser

def buildTree(parser, depth, dir_name, filestart=None):
	if not filestart is None:
		tree = N.findFile(filestart, parser, dir_name=dir_name)
	else:
		tree = N.Node(directory=dir_name)
	old_index = parser.index
	while not parser.atEnd() and tree.utility.count < depth:
		inst = parser.getNext()
		N.addInc(tree, inst)
		if not filestart is None and parser.index >= old_index * 2:
			tree.saveObject(filestart+str(parser.index)+'.cbwb', remove=filestart+str(old_index)+'.cbwb')
			old_index = parser.index
	if not filestart is None:
		tree.saveObject(filestart+str(parser.index)+'.cbwb', remove=filestart+str(old_index)+'.cbwb')
	return tree

if __name__ == "__main__":
	namecols = [0]
	timecols = [2]
	valcols = range(3,14)
	parser = Parser("AllPhoneData_approx_cleaned.csv",namecols,timecols,valcols,normalize=True)
	
	tree = buildTree(parser, 100000, ".", "test2_")
	tree.viz.plotSurprise()
