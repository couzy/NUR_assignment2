import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
import h5py

data=h5py.File('colliding.hdf5','r')

star_coords=data['PartType4/Coordinates'][:]
class Quad_tree:
	def __init__(self,data,x_0,y_0,x_1,y_1,threshold_node):
		self.threshold = threshold_node
		self.points = [Point(data[j,0],data[j,1])for j in range(len(data))]
		self.root = Node(x_0, y_0, x_1, y_1, self.points)

	def add_point(self,x, y):
		self.points.append(Point(x, y))

	def get_points(self):
		return self.points

	def subdivide(self):
		recursive_subdivide(self.root, self.threshold)

	def leaf_finding(self,point):
		find_leaf(self.root,self.threshold,point)
		
	def graph(self):
		fig = plt.figure(figsize=(12, 8))
		plt.title("Quadtree")
		ax = fig.add_subplot(111)
		c = find_childs(self.root)
		areas = set()
		for el in c:
			areas.add(el.width*el.height)
		for n in c:
			ax.add_patch(patches.Rectangle((n.x0, n.y0), n.width, n.height, fill=False))
		x = [point.x for point in self.points]
		y = [point.y for point in self.points]
		plt.scatter(x, y)
		plt.xlabel('x-position')
		plt.ylabel('y-position')
		plt.title('Quad-tree implementation')
		plt.savefig('plots/Quadtree.png')
		plt.clf()
		return

'''
Making a node which is a sqaure box with the lower left corner as coordinate x_0,y_0, the other 4 corners are defined by the height and the width the n=0 moment is calculated by simply taking the number of points in the node, as we have G=1 and all the particles are unit mass.
''' 

class Node:
	def __init__(self, x0, y0, width, heigth, points):
		self.x0 = x0
		self.y0 = y0
		self.width = width
		self.height = heigth
		self.points = points
		self.childs = []
		self.moments=len(points)

	def get_width(self):
		return self.width

	def get_height(self):
		return self.height

	def get_points(self):
		return self.points
'''
Make the points as a class so it can have an x- and a y-coordinate
'''
class Point:
	def __init__(self,x,y):
		self.x=x
		self.y=y
'''
Look recursively into the tree to find all the childs of a node,
'''			
def find_childs(node):
	if not (node.childs):
		return [node]
	else: 
		childs = []
		for child in node.childs:
			childs += (find_childs(child))
	return childs

'''
function to check for containing, looks if the points is in the square box
'''

def find_point(x_0, y_0, width, height, points):
	new_points = []
	for point in points:
		if point.x >= x_0 and point.x <= x_0+width and point.y>=y_0 and point.y<=y_0+height:
			new_points.append(point)
	return new_points		
'''
Look recursively into the tree and subdivide the nodes untill the threshold is met for each node, if the threshold is not met, recursively split up your square into four sqaures and look where your points should go. At the end every node has at most the amount of particles of our threshold.
'''
def recursive_subdivide(node,threshold):
	if len(node.points)<=threshold:
		return

	new_width = float(node.width/2)
	new_heigth = float(node.height/2)
			
	lower_left = find_point(node.x0, node.y0,  new_width, new_heigth, node.points)
	x1 = Node(node.x0, node.y0,  new_width, new_heigth, lower_left)
	recursive_subdivide(x1, threshold)

	upper_left = find_point(node.x0, node.y0+new_heigth, new_width, new_heigth, node.points)
	x2 = Node(node.x0, node.y0+new_heigth, new_width, new_heigth, upper_left)
	recursive_subdivide(x2, threshold)

	lower_right = find_point(node.x0+new_width, node.y0, new_width, new_heigth, node.points)
	x3 = Node(node.x0 +new_width, node.y0, new_width, new_heigth,lower_right)
	recursive_subdivide(x3, threshold)

	upper_right = find_point(node.x0+new_width, node.y0+new_width, new_width, new_heigth, node.points)
	x4 = Node(node.x0+new_width, node.y0+new_heigth, new_width, new_heigth, upper_right)
	recursive_subdivide(x4, threshold)

	node.childs = [x1, x2, x3, x4]

'''
Go down in the tree search each time which path you need to take to go to the right child, stop when you reach the final leaf of the tree, then output the moments you have saved up untill then.
'''

def find_leaf(node,threshold,point,moments=[]):
	if len(node.points)<=threshold:
		moments.append(node.moments)
		print('moments of the final leaf and the parent nodes are',moments[::-1])		
		return moments
	if point.x >= node.x0 and point.x <= node.x0+node.width/2 and point.y>=node.y0 and point.y<=node.y0+node.height/2:
		i=0
	elif point.x >= node.x0 and point.x <= node.x0+node.width/2 and point.y>=node.y0+node.height/2 and point.y<=node.y0+node.height:
		i=1
	elif point.x >= node.x0 +node.width/2 and point.x <= node.x0+node.width and point.y>=node.y0 and point.y<=node.y0+node.height/2:
		i=2
	elif point.x >= node.x0 +node.width/2 and point.x <= node.x0+node.width and point.y>=node.y0+node.height/2 and point.y<=node.y0+node.height:
		i=3
	moments.append(node.moments)
	find_leaf(node.childs[i],threshold,point,moments)

xy_star=star_coords[:,:2]
k_tree=Quad_tree(xy_star,0,0,150,150,12)
k_tree.subdivide()
k_tree.graph()
k_tree.leaf_finding(Point(xy_star[100][0],xy_star[100][1]))

