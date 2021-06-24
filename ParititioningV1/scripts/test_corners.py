import sys
import cv2 
import numpy as np
import random
from helpers.graph import *
from helpers.geometry import *;
import matplotlib.pyplot as plt


img = cv2.imread('/home/liseth/catkin_ws/maps/map3_modificado_cropped_ali.tif')
height, width = img.shape[:2]
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

gray = np.float32(gray)
dst = cv2.cornerHarris(gray, 2, 3, 0.04)
 
height, width = dst.shape
color = (0, 255, 0)

p_critical = []
p_corners = []

for y in range(0, height):
    for x in range(0, width):
        if dst.item(y, x) > 0.1 * dst.max():
        	p_corners.append([x,y])
        	cv2.circle(img, (x, y), 3, color, cv2.FILLED, cv2.LINE_AA)
cv2.imwrite('/home/liseth/catkin_ws/maps/map3_modificado_corners.tif',img)

boundary = [point(width, 0), point(0,0), point(0, height), point(width, height)];
obstacles = [];
obstacles.append(p_corners)

print obstacles

#sort by x-values
sorted_vertices = [];
for index,i in enumerate(obstacles):
	for j in i:
		j.append(index);
		sorted_vertices.append(j);
sorted_vertices.sort(key=lambda x: x[0]);

new_sorted_vertices = [];

for i in sorted_vertices:
	temp = point(i[0], i[1], i[2]);
	new_sorted_vertices.append(temp);

new_obstacles = [];
for index, i in enumerate(obstacles):
	temp_obs = [];
	for j in i:
		temp = point(j[0], j[1], index);
		temp_obs.append(temp);
	new_obstacles.append(temp_obs);	


#-----------------------------------------------------------
# Find vertical lines
open_line_segments = [];

y_limit_lower = boundary[0].y;
y_limit_upper = boundary[2].y;

for pt in new_sorted_vertices:
	curr_line_segment = [ point(pt.x, y_limit_lower), point(pt.x, y_limit_upper) ]; 
	lower_obs_pt = curr_line_segment[0];
	upper_obs_pt = curr_line_segment[1];
	upper_gone = False;
	lower_gone = False;
	break_now = False;

	# Find intersection points with the vertical proposed lines. the intersection function returns false if segments are same, so no need to worry about same segment checking
	for index,obs in enumerate(new_obstacles):
		# Add the first point again for the last line segment of a polygon.
		
		obs.append( obs[0] );
		for vertex_index in range(len(obs)-1 ):
			res = segment_intersection( curr_line_segment[0], curr_line_segment[1], obs[vertex_index],  obs[vertex_index+1]);
			if (res!=-1):
				if ( index == pt.obstacle ):
					if pt.equals( res ) == False:
						if ( res.y > pt.y ):
							upper_gone = True;
						elif ( res.y < pt.y ):
							lower_gone = True;	
				else:	
					if pt.equals( res ) == False:
						if ( upper_gone is False ):
							if ( (res.y > pt.y) and res.y < (upper_obs_pt.y) ):
								upper_obs_pt = res;
						if ( lower_gone is False ):
							if ( (res.y < pt.y) and (res.y > lower_obs_pt.y) ):
								lower_obs_pt = res;
			if( upper_gone is True and lower_gone is True ):
				break_now = True;

		#No need to check for current point anymore...completely blocked
		if(break_now is True):
			break;		

	# Draw the vertical cell lines
	if(lower_gone is False):
		cv2.line( img, (lower_obs_pt.x, lower_obs_pt.y), (pt.x, pt.y), (0, 0, 0), 1, 8);
		
	if(upper_gone is False):
		cv2.line( img, (pt.x, pt.y), (upper_obs_pt.x, upper_obs_pt.y), (0, 0, 0), 1, 8);

	# Add to the global segment list
	if (lower_gone and upper_gone):
		open_line_segments.append([None, None]);
	elif (lower_gone):
		open_line_segments.append([None, upper_obs_pt]);
	elif (upper_gone):
		open_line_segments.append([lower_obs_pt, None]);
	else:
		open_line_segments.append([lower_obs_pt, upper_obs_pt]);


cv2.imwrite('/home/liseth/catkin_ws/maps/map3_modificado_lines.tif',img)



