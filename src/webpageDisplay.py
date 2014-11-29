from PIL import Image
import os
import numpy
import shutil
import theano
import theano.tensor as T

# to test, load the data into datasets
# run createWebpage(datasets, datasets[1][1].owner.inputs[0].get_value())
# or createWebpage(datasets, datasets[2][1].owner.inputs[0].get_value())

# npndArrayTestSet is the npndarray holding the images and the real target values
# predictions is the list holding the predicted values for each image in the test set
def createWebpage(datasets, predictions):
	incorrect_out_file = "output_bad.html"
	badfile = open(incorrect_out_file, 'w')
	correct_out_file = "output_good.html"
	goodfile = open(correct_out_file, 'w')
	dir_name = "images"

	if os.path.exists(dir_name):
		shutil.rmtree(dir_name)
	os.makedirs(dir_name)

	# house keeping
	badfile.write("<html>\n")
	goodfile.write("<html>\n")
	badfile.write("<table>\n")
	goodfile.write("<table>\n")

# predicitions = output for each example
# for prediction in predicitions
# 	if prediction equal to the target
#		write to correct output file
#	else
#		write to incorrect output file

	imgArr = datasets[1][0].get_value()
	targetArr = datasets[1][1].owner.inputs[0].get_value()

	for targetIndex in range(targetArr.size):
		temparr = numpy.reshape(imgArr[targetIndex], (32,32))
		img = Image.fromarray((temparr * 255).astype(numpy.uint8))
		imgpath = "%s/%d.png" % (dir_name, targetIndex)
		img.save(imgpath)

		if targetArr[targetIndex] == predictions[targetIndex]:
			#CORRECTLY CLASSIFIED
			goodfile.write("<tr>\n")
			goodfile.write('<td><img src="%s/%d.png" width=40></td>\n' % (dir_name, targetIndex))
			goodfile.write("<td>%d</td>\n" % predictions[targetIndex])
			goodfile.write("</tr>\n")
		else:
			#INCORRECTLY CLASSIFIED
			badfile.write("<tr>\n")
			badfile.write('<td><img src="%s/%d.png" width=40></td>\n' % (dir_name, targetIndex))
			badfile.write("<td>%d</td>\n" % predictions[targetIndex])
			badfile.write("</tr>\n")

	# house keeping
	badfile.write("</table>\n")
	goodfile.write("</table>\n")
	badfile.write("</html>\n")
	goodfile.write("</html>\n")

	badfile.close()
	goodfile.close()
