# LeefSnap.py

""""
This project is the final project for ENGR 27 Computer Vision, at Swarthmore College
It was written by Evan Greene and Mollie Wild.
It attempts to imitate the LeafSnap project, (http://leafsnap.com/) the work of
Neeraj Kumar et al. Using the LeafSnap dataset, we copy some of LeafSnap's methods
for segmenting images of leaves, then use a method similar to eigenfaces to
identify leaves.

The semgentation returns a set of fifty points along the outline of the leaf.
Once the orientation and size are standardized, principal component analysis is run
on a large dataset to obtain a "fingerprint" for each species of leaf. 

In order to identify a novel leaf, the novel leaf is then segmented and is projected
onto the principal components of the library. The fingerprint of the new leaf is then
compared to that of each species in the database, and the one which it most closely
resembles is returned.

This program uses every other image in the LeafSnap database to create a libary of
fingerprints, and then tests every other image for accuracy. It was able to correctly
identify around 20 %% of the leaves.
"""

import cv2
import numpy
import math
# import
# from pandas import DataFrame

# a function that unlike cv2.imshow() allows the ctrl+c to interrupt
def showImage(string, image):
	"""
	calls the cv2.imshow() function, but allows the ctrl-c to close
	the image
	"""
	cv2.imshow(string, image)
	while cv2.waitKey(15) < 0:
		pass


def getcontourinfo(c):
    """compute moments and derived quantities such as mean, area, and
    basis vectors from a contour as returned by cv2.findContours.

    This function was lifted character for character from the cvk2
    module, written by Matt Zucker"""

    m = cv2.moments(c)

    s00 = m['m00']
    s10 = m['m10']
    s01 = m['m01']
    c20 = m['mu20']
    c11 = m['mu11']
    c02 = m['mu02']

    try:

        mx = s10 / s00
        my = s01 / s00

        A = numpy.array( [
                [ c20 / s00 , c11 / s00 ],
                [ c11 / s00 , c02 / s00 ]
                ] )

        W, U, Vt = cv2.SVDecomp(A)

        ul = math.sqrt(W[0,0])
        vl = math.sqrt(W[1,0])

        ux = ul * U[0, 0]
        uy = ul * U[1, 0]

        vx = vl * U[0, 1]
        vy = vl * U[1, 1]

        mean = numpy.array([mx, my])
        uvec = numpy.array([ux, uy])
        vvec = numpy.array([vx, vy])

    except:

        mean = c[0].astype('float')
        uvec = numpy.array([1.0, 0.0])
        vvec = numpy.array([0.0, 1.0])

    return {'moments': m,
            'area': s00,
            'mean': mean,
            'b1': uvec,
            'b2': vvec}

def segment(image):
	"""This fucntion takes an image, or a filename pointing to an image,
	imports it as an opencv array, then returns a set of points that
	form the contour of the image"""

	# if the inumpyut is a filename, import it.
	if type(image) is str:
		filename = image
		image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
		# if the importation fails, print an error message and exit the program.
		if image is None:
			print 'file ' + filename + ' not found'
			# like sys.exit() but without the need for an import statement.
			assert(0)

	# Convert the image to grayscale if it isn't already
	elif len(image) == 3:
		image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

	# Convert the image to a numpy.uint8 type, best for this kind of application
	image = image.astype('uint8')

	image = cv2.resize(image, (400, 400))

	# create a mask with a simple threshold. Use the mean of the image as a threshold
	mu = numpy.mean(image)

	_, mask = cv2.threshold(src = image, thresh = mu - 50, maxval = 255, type = cv2.THRESH_BINARY_INV)

	# open the image to smooth the edges
	mask = cv2.morphologyEx(src = mask, op = cv2.MORPH_OPEN, kernel = (5, 5))

	# Show the thresholded image
	# showImage('thresholded', mask)

	""" to remove false-positive regions, find the contours of the image.
	Then, find the contour with the largest area and draw that on a blank background.
	This method is from the Neeraj Kumar et al. paper"""

	# find the contours
	contours, hierarchy = cv2.findContours(image = mask, mode = cv2.RETR_LIST, method = cv2.CHAIN_APPROX_NONE)

	# Continue segmentation if the image has contours. Else, the segmentation must have failed
	# and we will return None as contours and a blank image as a mask.
	if contours:

		# find the areas for each contours
		# because the images in the LeafSnap dataset have a scale in the lower right,
		# areas are not calculated for contours in that region, and zero is substituted.
		areas = []
		max_x = int(mask.shape[0] * 0.8)
		max_y = int(mask.shape[1] * 0.8)

		for contour in contours:
			# for contours not in the lower right
			if (numpy.mean(contour[0, :, 0]) <= max_x and numpy.mean(contour[0, :, 1]) <= max_y):
				area = cv2.contourArea(contour)
				areas.append(area)
			else:
				areas.append(0)
		""" Continue the segmentation if the area of the leaf is greater than ten
		pixels. If it is not, the segmentation must have failed, or the leaf must be
		too small to identify"""
		if max(areas) > 10:

			# the contour with the biggest area is almost definitely the leaf.
			bigContour = contours[areas.index(max(areas))]
			# print bigContour

			mask = numpy.zeros(image.shape)
			cv2.fillPoly(img = mask, pts = bigContour, color = 255)
			# showImage('segmented', mask)

			# find the moments of the contour
			m = getcontourinfo(bigContour)

			# move the contour so that the mean is centered at zero, so the program will recognize
			# images leaves that occupy different parts of the image as being the same
			bigContour = bigContour.astype('float64') - m['mean']

			# scale the contours so the area is 12000 square pixels (arbitrarily chosen)

			# print m['area']
			bigContour = bigContour * (15000 / m['area'])

			# rotate the contour so that the principal axis is horizontal
			bigContour = bigContour.reshape(-1, 2)

			# find the angle between the horizontal and the principal axis
			angle = numpy.arctan2(m['b1'][1], m['b1'][0])

			# create a rotation matrix
			rotation  = numpy.array([[numpy.cos(angle), -numpy.sin(angle)], [numpy.sin(angle), numpy.cos(angle)]])
			# multiply the contours by the rotation matrix
			bigContour = numpy.dot(rotation, bigContour.transpose())

			# Now, create a mask for display.
			dispContour = [[200], [200]] + bigContour
			dispContour = dispContour.transpose().reshape(-1, 1, 2).astype('int32')
			# print 'dispContour', dispContour.reshape(-1, 2)
			mask = numpy.zeros((400, 400), dtype = 'uint8')

			cv2.fillPoly(img = mask, pts = dispContour, color = 255)

			# showImage('normalized', mask)
			# reduce the contour to a uniform number of points, given in the input as ppc.
			indices = numpy.linspace(0, bigContour.shape[1] - 1, 100).tolist()
			indices = [int(x) for x in indices]
			# print indices
			bigContour = numpy.array([bigContour[:, x] for x in indices])

		else:
			mask = numpy.zeros((400, 400), dtype = 'uint8')
			bigContour = None
	else:
		mask = numpy.zeros((400, 400), dtype = 'uint8')
		bigContour = None

	return mask, bigContour

def createdirectory(filename):
	"""parses the leafsnap dataset and returns a list of lists,
	detailing where the image files are. """
	directory = list(file(filename, mode = 'rb'))
	for i in range(len(directory)):
		directory[i] = directory[i].strip('\n')
		directory[i] = directory[i].split('\t')

	return directory

def createLibrary(directory):
	"""
	Goes through every image in the dataset, finds the contours, then performs
	principal component analysis on the contours.
	Returns a tuple with the PCA vectors, the species of tree, the ,
	 """

	print 'Creating Library'
	# length = 10
	length = len(directory) - 1
	# set up an empty list for the indices of the species included
	included = []
	# set up counters for successful segmentation
	successful = 0
	outof = 0

	# for entry in range(3, length, 2):
	for entry in range(1, length, 2):
		# segment the images
		mask, bigContour = segment(directory[entry][1])
		# showImage('semgented image' + str(entry), mask)
		# add the contours to the array we're using for PCA
		if bigContour is not None:
			# try to add the bigContour to the array for PCA
			try:
				pcaarray = numpy.append(pcaarray, bigContour.reshape(1, -1), axis = 0)
			# if the array doesn't exist, it will raise UnboundLocalError,
			# so we create the array
			except UnboundLocalError:
				pcaarray = bigContour.reshape(1, -1)
			included.append(entry)
			print 'segmented image ' + str(entry)
			successful += 1
			outof += 1

		else:
			print 'segmentation failed for image ' + str(entry)
			outof += 1

	print 'Successfully segmented' + str(successful) + '/' + str(outof) + ', ' + str(float(successful)/outof)

	# Now perform the PCA, find the principal components
	mean, eigenvectors = cv2.PCAComputeVar(data = pcaarray, retainedVariance = 0.99)
	# transform the data into the eigenvalue space
	# print pcaarray.shape
	newvectors = cv2.PCAProject(data = pcaarray, mean = mean, eigenvectors = eigenvectors)
	# print newvectors.shape
	# create two lists, one for responses and the other for species
	responses = newvectors.tolist()
	species = [directory[i][3] for i in included]

	assert(len(responses) == len(species))

	return(responses, species, mean, eigenvectors)


def testNovelImage(image, responses, species, mean, eigenvectors):
	global directory
	# segment the image
	_ , bigContour = segment(image)
	if bigContour is None:
		output = None
	else:
		bigContour = numpy.array(bigContour).reshape(1, -1)

		# project pca onto the contours
		resp = cv2.PCAProject(data = bigContour, mean = mean, eigenvectors = eigenvectors)

		# Calculate the distance to each response in the library
		distances = numpy.sum((responses-resp)**2, axis = 1).tolist()

		output = species[distances.index(min(distances))]

	return output

def main():
	# Load the library, and if it doesn't exist, create it
	try:
		responses, species, mean, eigenvectors = numpy.load('leefsnap_library_half.npz')
	except IOError:
		directory = createdirectory('leafsnap-dataset-images.txt')
		responses, species, mean, eigenvectors = createLibrary(directory)
		numpy.savez(file = 'leefsnap_library_half', args = (responses, species, mean, eigenvectors), keys = (responses, species, mean, eigenvectors))

	# length = 10
	length = len(directory)
	correct = 0
	total = 0
	for i in range(2, length, 2):
		output = testNovelImage(directory[i][1], responses, species, mean, eigenvectors)
		print 'testing image ' + str(i) + ', file_id: ' + str(directory[i][0])
		if output == directory[i][3]:
			print 'predicted: ' + output + ', actual: ' + directory[i][3] + '--CORRECT!'
			correct += 1
			total += 1
		elif output is None:
			print 'could not segment image' + str(i) + ', actual: ' + directory[i][3] +'WRONG'
			total += 1
		else:
			print 'predicted: ' + output + ', actual ' + directory[i][3] + '--WRONG!'
			total += 1
	print str(correct) + '/' + str(total) + ' correct, accuracy = ' + str((float(correct)/(total))*100) + '%'
	# segment('original.jpg')
main()
