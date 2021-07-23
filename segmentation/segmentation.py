import numpy as np
import cv2 as cv
import random
import os
from skimage import io, color
import matplotlib.pyplot as plt
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries


def get_adjacent_black_white_pixels(img):
    r, c = img.shape
    white = []
    black = []
    
    for i in range(r):
        for j in range(c):
            if (j+1 != c and img[i,j] == 255 and img[i, j+1] == 0):
                white.append((i,j))
                black.append((i, j+1))
            if (j != 0 and img[i,j] == 255 and img[i, j-1] == 0):
                white.append((i,j))
                black.append((i, j-1))

    for j in range(c):
        for i in range(r):
            if (i+1 != r and img[i,j] == 255 and img[i+1,j] == 0):
                white.append((i,j))
                black.append((i+1, j))
            if (i != 0 and img[i,j] == 255 and img[i-1, j] == 0):
                white.append((i,j))
                black.append((i-1, j))


    white = np.array(white)
    white = np.unique(white, axis=0)

    black = np.array(black)
    black = np.unique(black, axis=0)
    return white,black


def get_adjacent_pairs(img,clusters):
    r, c = img.shape
    pairs = []
    
    for i in range(r):
        for j in range(c):
            if (j+1 != c and img[i,j] == 255 and img[i, j+1] == 0):
                white = clusters[i,j]
                black = clusters[i, j+1]
                pairs.append((white,black))
            if (j != 0 and img[i,j] == 255 and img[i, j-1] == 0):
                white = clusters[i,j]
                black = clusters[i, j-1]
                pairs.append((white,black))

    for j in range(c):
        for i in range(r):
            if (i+1 != r and img[i,j] == 255 and img[i+1,j] == 0):
                white = clusters[i,j]
                black = clusters[i+1, j]
                pairs.append((white,black))
            if (i != 0 and img[i,j] == 255 and img[i-1, j] == 0):
                white = clusters[i,j]
                black = clusters[i-1, j]
                pairs.append((white,black))

    return pairs


def color_thresholding_cs(seg_img,clusters, cielab):
    #get foreground background adjacent pairs
    adj_pairs = get_adjacent_pairs(seg_img,clusters)
    cs, distance_list = compute_cs(adj_pairs, cielab)
    
    repeat = False
   
    updated_img = np.copy(seg_img)
    while True:
        invert_list = []
        for pair,dst in zip(adj_pairs,distance_list):
            if (dst > cs):
                invert_list.append(pair[1])
                repeat = True
        for reg in invert_list:
            pts = np.argwhere(clusters == reg)
            for x,y in pts:
                updated_img[x,y] = 255
        #show_image(updated_img)
        if repeat:
            #The relabeling is repeated until there is no change in the labels of the segments.
            adj_pairs = get_adjacent_clusters(updated_img, clusters)
            cs, distance_list = compute_cs(adj_pairs, cielab)
            repeat = False
        else:
            break
            
    return updated_img


def color_color_thresholding_cp(binary_img, cielab, foreground, background):
    white,black = get_adjacent_black_white_pixels(binary_img)
    cp, distance_list = compute_cp(white, black, cielab)
    
    repeat = False
    
    while True:
        for dst, w in zip(distance_list,white):
            if dst < cp:
                binary_img[w[0]][w[1]] = 0
                repeat = True
        if repeat:
            white,black = get_adjacent_black_white_pixels(binary_img)
            cp, distance_list = compute_cp(foreground, background, cielab)
            repeat = False
        else:
            break
            
        return binary_img
            

def optical_flow(cap):
  _, frame0 = cap.read()
  _, frame1 = cap.read()
  for i in range(5):
    _, frame2 = cap.read()
  frame1 = cv.cvtColor(frame1,cv.COLOR_BGR2RGB)
  gray = color.rgb2gray(frame1)

  next = cv.cvtColor(frame2,cv.COLOR_BGR2GRAY)
  prvs = cv.cvtColor(frame0,cv.COLOR_BGR2GRAY)
  flowprev = cv.calcOpticalFlowFarneback(prvs,gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
  flownext = cv.calcOpticalFlowFarneback(next,gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
  mag1, ang1 = cv.cartToPolar(flowprev[...,0], flowprev[...,1])
  mag2, ang2 = cv.cartToPolar(flownext[...,0], flownext[...,1])
  u = (mag1*np.cos((180/np.pi/2)*ang1) + mag2*np.cos((180/np.pi/2)*ang2))/2
  v = (mag1*np.cos((180/np.pi/2)*ang1) + mag2*np.cos((180/np.pi/2)*ang2))/2
  max = np.max(u)
  u = u*(255/max)
  #show_image(u)
  max = np.max(v)
  v = v*(255/max)
  diff = cv.absdiff(next, prvs)
  gray = diff.copy()
  gray[gray > 10]  = 255
  gray[gray <= 10] = 0
  gray = cv.dilate(gray, None, iterations=4)
  gray = cv.erode (gray, None, iterations=4)
  #show_image(diff)
  show_image(gray)
  gray_plus = gray.copy()
  gray_plus[gray == 255] = 2

  #nehal's variable
  diff_img = np.copy(gray)

  return frame1,u,v,diff_img


def get_adjacent_black_white_pixels(img):
    r, c = img.shape
    white = []
    black = []
    
    for i in range(r):
        for j in range(c):
            if (j+1 != c and img[i,j] == 255 and img[i, j+1] == 0):
                white.append((i,j))
                black.append((i, j+1))
            if (j != 0 and img[i,j] == 255 and img[i, j-1] == 0):
                white.append((i,j))
                black.append((i, j-1))

    for j in range(c):
        for i in range(r):
            if (i+1 != r and img[i,j] == 255 and img[i+1,j] == 0):
                white.append((i,j))
                black.append((i+1, j))
            if (i != 0 and img[i,j] == 255 and img[i-1, j] == 0):
                white.append((i,j))
                black.append((i-1, j))


    white = np.array(white)
    white = np.unique(white, axis=0)

    black = np.array(black)
    black = np.unique(black, axis=0)
    return white,black


def get_saliency_map(img):
    saliency = cv.saliency.StaticSaliencyFineGrained_create()
    _, saliencyMap = saliency.computeSaliency(img)
    saliencyMap = (saliencyMap * 255).astype("uint8")
    
    return saliencyMap


def add_to_dict(dic, reg1, reg2):
    if(dic.get(reg1) == None):
        dic[reg1] = [reg2]
    else:
        dic[reg1].append(reg2)
    return


def get_adjacent_clusters(clusters, img):
    r, c = clusters.shape
    adj = dict()
    for i in range(r):
        for j in range(c):
            if (j+1 != c and img[i,j] == 255 and img[i, j+1] == 0):
                white = clusters[i,j]
                black = clusters[i,j+1]
                add_to_dict(adj, black, white)
            if (j != 0 and img[i,j] == 255 and img[i, j-1] == 0):
                white = clusters[i,j]
                black = clusters[i, j-1]
                add_to_dict(adj, black, white)

    for j in range(c):
        for i in range(r):
            if (i+1 != r and img[i,j] == 255 and img[i+1,j] == 0):
                white = clusters[i,j]
                black = clusters[i+1,j]
                add_to_dict(adj, black, white)
            if (i != 0 and img[i,j] == 255 and img[i-1, j] == 0):
                white = clusters[i,j]
                black = clusters[i-1, j]
                add_to_dict(adj, black, white)
    return adj


def compute_distance(p1, p2, color):
    dist = np.sqrt((color[p1][0]-color[p2][0])**2 + (color[p1][1]-color[p2][1])**2 +(color[p1][2]-color[p2][2])**2)
    return dist


def color_threshold(clusters,seg_img, color):
    dic = get_adjacent_clusters(clusters,seg_img)
    img = np.copy(seg_img)
    
    for bg,adj_list in dic.items():
        dist_list = dict()
        for adj in adj_list:
            dist = compute_distance(bg, adj, color)
            dist_list[adj] = dist
        
        nfg = 0 
        nbg = 0
        for reg,dist in dist_list.items():
            x,y = np.argwhere(clusters == reg)[0]
            isfg = seg_img[x,y]
            if(isfg == 0 and dist < 15 ):
                nbg = nbg + 1
            elif (isfg == 255 and dist < 15 ):
                nfg = nfg + 1    
        if(nfg > nbg):
            pts = np.argwhere(clusters == bg)
            for x,y in pts:
                img[x,y] = 255
        
    return img


def compute_cs(adjacent_pairs, cielab):
    cs = 0
    distance_list = []
    for frgrnd,bckgrnd in adjacent_pairs:
        dst = (cielab[frgrnd][0]-cielab[bckgrnd][0])**2 + (cielab[frgrnd][1]-cielab[bckgrnd][1])**2 +(cielab[frgrnd][2]-cielab[bckgrnd][2])**2 
        dst = np.sqrt(dst)
        cs = cs + dst
        distance_list.append(dst)
    cs = (cs/len(adjacent_pairs)) - 10
    return cs,distance_list


def compute_cp(foreground, background, cielab):
    cp = 0
    distance_list = []
    for f,b in zip(foreground,background):
        dst = (cielab[f[0]][f[1]][0]-cielab[b[0]][b[1]][0])**2 + (cielab[f[0]][f[1]][1]-cielab[b[0]][b[1]][1])**2 +(cielab[f[0]][f[1]][2]-cielab[b[0]][b[1]][2])**2 
        dst = np.sqrt(dst)
        cp = cp + dst
        distance_list.append(dst)
    return cp/len(distance_list), distance_list


def superpixels_slic(img, k):
    results = slic(img, n_segments = k, sigma = 5)
    # img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    # balck_img = np.zeros((results.shape[0],results.shape[1]))
    # boundaries = mark_boundaries(balck_img, results)
    return results


def get_centers(img, lab, k):
    centers = []
    l = lab[:,:,0]
    a = lab[:,:,1]
    b = lab[:,:,2]
    for j in range(k):
        center = []
        indices = np.where(img == j)
        if len(indices[0]) != 0:
            matrix  = [l[indices[0][i]][indices[1][i]] for i in range(len(indices[0]))]
            center.append(sum(matrix)/len(matrix))
            matrix  = [a[indices[0][i]][indices[1][i]] for i in range(len(indices[0]))]
            center.append(sum(matrix)/len(matrix))
            matrix  = [b[indices[0][i]][indices[1][i]] for i in range(len(indices[0]))]
            center.append(sum(matrix)/len(matrix))
            centers.append(center)
        else:
            centers.append([])
    return centers


def get_cielab(img):
    gray = color.rgb2gray(img)
    lab = color.rgb2lab(img)
    l = lab[:,:,0]
    # a = lab[:,:,1]
    # b = lab[:,:,2]
    # print(l.shape)
    return gray, lab


def get_forground(superpixels, dframe, lab):
    clusters = superpixels.copy()
    forground = []
    new_dframe = np.zeros((dframe.shape[0], dframe.shape[1]))
    new_dframe[dframe == 255] = 1
    for i in range(300):
        if(sum(new_dframe[clusters == i]) >= 0.5 * len(clusters[clusters == i])):
            forground.append(i)
    #forground = [ i  for i in range(300) if((len(clusters[clusters == i])-sum(dframe[clusters == i])) < 0)]
    first_threshold = np.zeros((lab.shape[0], lab.shape[1]))
    for i in forground:
        first_threshold[clusters == i] = 255

    return first_threshold, forground


def build_background(median1, median2, background, modified):
    th   = 2
    prev = cv.cvtColor(median1,cv.COLOR_BGR2RGB)
    curr_med = cv.cvtColor(median2, cv.COLOR_BGR2GRAY)
    prev_med = cv.cvtColor(median1,cv.COLOR_BGR2GRAY)
    diff_color = cv.absdiff(curr_med, prev_med)
    diff_color[diff_color >  th] = 255
    diff_color[diff_color <= th] = 0
    # black_index =  np.where(background == (0,0,0))
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(1,1))
    diff_color = cv.morphologyEx(diff_color, cv.MORPH_CLOSE, kernel)
    index = np.where(diff_color == 0)
    background = background.astype(np.uint8)
    for i in range(len(index[0])):
        if modified[index[0][i]][index[1][i]] == 0:
            background[index[0][i]][index[1][i]] = prev[index[0][i]][index[1][i]]
            modified[index[0][i]][index[1][i]] == 1
    
    return background


def get_background(frame, medians):
    background = np.zeros(frame.shape)
    modified   = np.zeros((frame.shape[0],frame.shape[1]))
    length = len(medians)
    for m in range(length+10):
        rand1 = random.randint(0, length-1)
        rand2 = random.randint(0, length-1)
        if rand1 != rand2:
            background = build_background(medians[rand1], medians[rand2], background, modified)
    return background


def get_medians(frames_count, frames_path, frames_names, num_median=100):
    medians = []
    num_iter = int(np.ceil(frames_count / num_median))
    end = 0
    j = 0
    for it in range(num_iter):
        frames  = []
        for fid in range(num_median):
            if it * num_median + fid == frames_count:
                end = 1
                break
            frame = cv.imread(frames_path + '/' + frames_names[j])
            j += 1
            frames.append(frame)
        # Calculate medianFrame through the timeline
        median = np.median(frames, axis=0).astype(dtype=np.uint8) 
        median = cv.cvtColor(median, cv.COLOR_BGR2RGB)
        medians.append(median) 
        if end == 1:
            break
    return medians, frame


def segment_images(frames_path, segmented_images_path):
    if not os.path.exists(segmented_images_path):
        os.makedirs(segmented_images_path)

    frames_names = os.listdir(frames_path)
    frames_names = sorted(frames_names)
    frames_count = len(frames_names)
    
    print("Calculating frames medians")
    medians, frame = get_medians(frames_count, frames_path, frames_names)
    print("Done calculating frames")
    
    background = get_background(frame, medians)
    cv.imshow('background', background)
    cv.waitKey()
    grayMedianFrame = cv.cvtColor(background, cv.COLOR_BGR2GRAY)
    
    for i in range(frames_count):
        print('processing frame: ', frames_path, '/', frames_names[i], sep='')
        frame = cv.imread(frames_path + '/' + frames_names[i])
        
        grayframe = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        dframe = cv.absdiff(grayframe, grayMedianFrame)
        _, dframe = cv.threshold(dframe, 30, 255, cv.THRESH_BINARY)
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        superpixels = superpixels_slic(frame, 300)
        gray,lab = get_cielab(frame)
        centers = get_centers(superpixels, frame, 300)

        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2,2))
        opening = cv.morphologyEx(dframe, cv.MORPH_OPEN, kernel)
        opening = cv.morphologyEx(opening, cv.MORPH_OPEN, kernel)
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(5,5))
        closing = cv.morphologyEx(opening, cv.MORPH_CLOSE, kernel)
        closing = cv.morphologyEx(closing, cv.MORPH_CLOSE, kernel)
        closing = cv.morphologyEx(closing, cv.MORPH_CLOSE, kernel)
        
        foreground, _ = get_forground(superpixels, dframe, lab)
        
        thresh_img = color_threshold(superpixels,foreground, centers)
        
        file_name = segmented_images_path + '/' + frames_names[i]
        print('saving frame: ', file_name)
        cv.imwrite(file_name, thresh_img)
        
