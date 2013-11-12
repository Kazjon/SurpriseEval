import N
from P import Parser
from matplotlib import pyplot as pl
import numpy as np

surprise = []
names = []
graph = None
tracker = None

class surpriseObj:
	def __init__(self):
		self.last_tag = None

def buildTree(parser, depth, dir_name):
	tree = N.Node(directory=dir_name)
	counter = 0
	while not parser.atEnd() and tree.utility.count < depth:
		counter += 1
		inst = parser.getNext()
		N.addInc(tree, inst)
		surprise.append(tree.measure.depth_only(inst))
		names.append(inst.pretty_print(False))
		tree.toDot(str(counter)+".dot", withStrings=False, latest=inst)
	return tree

def closestElement(x, y):
	if not tracker.last_tag is None:
		tracker.last_tag.remove()
	range_x = len(surprise)
	range_y = max(surprise) - min(surprise)
	dist = [float(abs(x - i))/range_x + float(abs(y-surprise[i]))/range_y for i in range(len(surprise))]
	index = dist.index(min(dist))
	tracker.last_tag = graph.annotate(
		names[index], 
		xy = (index, surprise[index]), xytext = (-20, 20),
		textcoords = 'offset points', ha = 'right', va = 'bottom',
		bbox = dict(boxstyle = 'round,pad=0.5', fc = 'white', alpha = 1.0),
		arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))
	pl.draw()

def onclick(event):
	#print 'button=%d, x=%d, y=%d, xdata=%f, ydata=%f'%(event.button, event.x, event.y, event.xdata, event.ydata)
	closestElement(event.xdata, event.ydata)

if __name__ == "__main__":
	tracker = surpriseObj()
	namecols = [0]
	timecols = [2]
	valcols = range(3,14)
	parser = Parser("AllPhoneData_pruned.csv",namecols,timecols,valcols,normalize=True)
	
	tree = buildTree(parser, 10, "testing")
	#pl.scatter(range(len(surprise_list)), surprise_list)
	
	fig = pl.figure()
	ax = fig.add_subplot(111)
	graph = ax
	ax.scatter(range(len(surprise)), surprise)
	cid = fig.canvas.mpl_connect('button_press_event', onclick)

	pl.show()
