import N
from P import Parser
import argparse

if __name__ == "__main__":
	interpreter = argparse.ArgumentParser(description='Show the surpisingness of phones already added to the cluster hierarchy.')
	interpreter.add_argument('saveName', metavar='<begining_of_cbwb_file>', type=str, nargs=1, help='The part of the .cbwb file for this cluster which comes before the \'_\' (i.e. "test" if the cluster is saved as test_N.cbwb)') 
	interpreter.add_argument('-line', dest="showLine", action='store_true', help='Show the line of average surprise over time instead of regular output.')
	interpreter.add_argument('-csv', type=str, help='Save surprise values to csv file.')
	interpreter.set_defaults(showLine=False)
	args = interpreter.parse_args()
	
	tree = N.findFile(args.saveName[0]+"_", None)
	if args.csv:
		tree.viz.toCSV(args.csv)
	if args.showLine:
		tree.viz.surpriseLine()
	else:
		tree.viz.plotSurprise()
