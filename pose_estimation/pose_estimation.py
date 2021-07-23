import cv2 as cv
import numpy as np
import numpy as np
import math
import imutils
import os

beta = 0.75


class BodyPart():
  def __init__(self, name='part'):
    self.name = name
    
    self.x = 0
    self.y = 0
    self.theta = 0
    self.l = 0
    self.w = 0
    
    self.children = []
    self.parent = None
    
    self.left_upper_corner = None
    self.right_upper_corner = None
    self.left_lower_corner = None
    self.right_lower_corner = None

    self.priority = 0
    self.area = 0
    self.Si = 0 # intersection of synthesized body part with foreground
    self.visited = 0 # number of time this body part was updated
  
  def setData(self, x, y, theta, l, w):
    self.x = x
    self.y = y
    self.theta = theta
    self.l = l
    self.w = w
    self.area = l * w
    self.setCorners()

  def updateValue(self, indx, lamda):
    if indx == 0:
      self.x += int(lamda)

    elif indx == 1:
      self.y += int(lamda)

    elif indx == 2:
      self.theta += lamda
      
    elif indx == 3:
      self.l += int(lamda)
      self.area = self.l * self.w

    else: 
      self.w += int(lamda)
      self.area = self.l * self.w

    self.setCorners()

  def addChildren(self, children):
    for child in children:
      self.children.append(child)
  
  def setParent(self, parent):
    self.parent = parent

  def getData(self):
    return (self.x, self.y, self.theta, self.l, self.w)

  def setCorners(self):
    if self.name == 'Torso':
      center = True
    else:
      center = False
    self.left_upper_corner = get_left_upper_corner(self.x, self.y, self.theta, self.l, self.w, center)
    self.right_upper_corner = get_right_upper_corner(self.x, self.y, self.theta, self.l, self.w, center)
    self.left_lower_corner = get_left_lower_corner(self.x, self.y, self.theta, self.l, self.w, center)
    self.right_lower_corner = get_right_lower_corner(self.x, self.y, self.theta, self.l, self.w, center)


# input : frame , initial background with no human in it
# output: binary image 
def segmentation (frame, background):
    if len(frame.shape) > 2:
      frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    if len(background.shape) > 2:
      background = cv.cvtColor(background, cv.COLOR_BGR2GRAY)
    diff = cv.absdiff(frame, background)
    diff[diff > 35] = 255
    diff[diff <= 35] = 0
    return diff


def get_body_height(fore):
    miniy = 0
    maxiy = 0
    for i in range(fore.shape[0]):
        r= fore[i]
        x = np.argmax(r)
        if r[x] == 255:
            miniy = i
            break
            
    for i in reversed(range(fore.shape[0])):
        r = fore[i]
        x = np.argmax(r)
        if r[x] == 255:
            maxiy = i
            break

    height = abs(maxiy - miniy)
    return height


def get_torso_center(foreImage):
    distMap= cv.distanceTransform(foreImage, cv.DIST_L2, 5)
    (yTorso, xTorso) = np.where(distMap == np.amax(distMap))
    return (xTorso[0], yTorso[0])


####################length of torso#############################
def get_torso_length(foreImage, lBody):
  meanLTorso = .33
  varLTorso = .001
  mu = meanLTorso * lBody
  sigma = np.sqrt(varLTorso * (lBody**2))
  lTorso = np.random.normal(mu, sigma)
  return lTorso
##################################################################


####################width of torso#############################
def get_torso_width(foreImage, wBody):
  meanWTorso = 1
  varWTorso = .001
  mu = meanWTorso * wBody
  sigma = np.sqrt(varWTorso * (wBody**2))
  wTorso = np.random.normal(mu, sigma)
  return wTorso
##################################################################


def get_torso_angle(foreImage):
  fore = foreImage.copy()
  # get horizontal histogram
  num_rows = foreImage.shape[0]
  distMap= cv.distanceTransform(foreImage, cv.DIST_L2, 5)
  (yFirst, xFirst) = np.where(distMap == np.amax(distMap))
  xFirst = int(xFirst[0])
  yFirst = int(yFirst[0])

  cropped_image = fore[min(yFirst + 5, num_rows - 1):, ]

  distMap= cv.distanceTransform(cropped_image, cv.DIST_L2, 5)
  (ySecond, xSecond) = np.where(distMap == np.amax(distMap))
  xSecond = int(xSecond[0])
  ySecond = int(ySecond[0]) + min(yFirst + 5, num_rows - 1)
  
  if abs(ySecond - yFirst) < 30:
    cropped_image = fore[0:max(yFirst - 5, 0), ]
    distMap = cv.distanceTransform(cropped_image, cv.DIST_L2, 5)
    if not distMap is None:
      (ySecond, xSecond) = np.where(distMap == np.amax(distMap))
      xSecond = int(xSecond[0])
      ySecond = int(ySecond[0])
    
  deltaY = ySecond - yFirst
  deltaX = xSecond - xFirst

  if deltaX != 0:
    theta = np.arctan(deltaY/deltaX) * 180.0 / np.pi
  else:
    theta = 90.0
  return 360
  #return abs(90 - theta)


def get_torso_model(image_R,face,img):
  lBody = get_body_height(img)
  wBody = 0.17 * lBody
  l = get_torso_length(img, lBody)
  w = get_torso_width(img, wBody)
  x,y= get_TFH(image_R,face,l)
  theta = get_torso_angle(img)
  torso_data = (x, y, theta, l, w)
  return torso_data


def get_right_upper_arm_model(torso_center_x, torso_center_y, torso_theta,torso_height, torso_w):
    
    meanHeight = .55 * torso_height
    varHeight = .02
    height = np.random.normal(meanHeight, varHeight)
    
    meanW = .2 * torso_w
    varW = .1
    width = np.random.normal(meanW, varW)
    
    (top_right_x,top_right_y) = get_right_upper_corner(torso_center_x, torso_center_y, torso_theta, torso_height, torso_w,True)
    top_right_y = top_right_y + (.5 * width)
    sigma_x = 1
    right_x = top_right_x
    right_y = top_right_y
    
    theta = np.random.normal(45,10)
    
    return right_x, right_y, theta, height, width


def get_left_upper_arm_model(torso_center_x, torso_center_y,torso_theta, torso_height, torso_w):
    meanHeight = .55 * torso_height
    varHeight = .02
    height = np.random.normal(meanHeight, varHeight)
    
    meanW = .2 * torso_w
    varW = .1
    width = np.random.normal(meanW, varW)
    
    (top_left_x,top_left_y) = get_left_upper_corner(torso_center_x, torso_center_y, torso_theta, torso_height, torso_w,True)
    top_left_y = top_left_y+(.5 * width)
    sigma_x = 3
    left_x = top_left_x
    left_y = top_left_y
    
    theta = np.random.normal(125, 10)
    
    return left_x, left_y, theta, height, width


def get_right_lower_arm_model(end_x, end_y, torso_height, torso_w):
    meanHeight = .55 * torso_height
    varHeight = .02
    height = np.random.normal(meanHeight, varHeight)
    
    meanW = .2 * torso_w
    varW = .1
    width = np.random.normal(meanW, varW)
    
    top_right_x = end_x
    top_right_y = end_y
    sigma_x = 1
    right_x = np.random.normal(top_right_x, sigma_x)
    right_y = np.random.normal(top_right_y, sigma_x)
    
    theta = np.random.normal(45, 10)
    
    return right_x, right_y, theta, height, width


def get_left_lower_arm_model(end_x, end_y, torso_height, torso_w):
    meanHeight = .55 * torso_height
    varHeight = .02
    height = np.random.normal(meanHeight, varHeight)
    
    meanW = .2 * torso_w
    varW = .1
    width = np.random.normal(meanW, varW)
    
    top_left_x = end_x
    top_left_y = end_y
    sigma_x = 3
    left_x = np.random.normal(top_left_x, sigma_x)
    left_y = np.random.normal(top_left_y, sigma_x)
    
    theta = np.random.normal(125, 10)
    
    return left_x, left_y, theta, height, width


def get_left_upper_leg_model(torso_center_x,torso_center_y,torso_theta,torso_height,torso_w):
    meanHeight = .7* torso_height
    varHeight = .01
    height = np.random.normal(meanHeight, varHeight)
    
    meanW = .35 * torso_w
    varW = .1
    width = np.random.normal(meanW, varW)
    
    (bottom_left_x,bottom_left_y) = get_left_lower_corner(torso_center_x, torso_center_y, torso_theta, torso_height, torso_w,True)
    
    
    bottom_left_x = bottom_left_x+(.5 * width)

    sigma_x = 0
    left_x = np.random.normal(bottom_left_x, sigma_x)
    left_y = np.random.normal(bottom_left_y, sigma_x)
    
    theta = np.random.normal(100, 10)
    
    return left_x, left_y, theta, height, width


def get_right_upper_leg_model(torso_center_x, torso_center_y,torso_theta, torso_height, torso_w):
    meanHeight = .7 * torso_height
    varHeight = .01
    height = np.random.normal(meanHeight, varHeight)
    
    meanW = .34 * torso_w
    varW = .1
    width= np.random.normal(meanW, varW)
    
    (top_right_x,top_right_y) = get_right_lower_corner(torso_center_x, torso_center_y, torso_theta, torso_height, torso_w,True)
    top_right_x = top_right_x - (.5 * width)

    sigma_x = 0
    right_x = np.random.normal(top_right_x, sigma_x)
    right_y = np.random.normal(top_right_y, sigma_x)
    
    theta = np.random.normal(80, 10)
    
    return right_x, right_y, theta, height, width


def get_left_lower_leg_model(end_x, end_y, torso_height, torso_w):
    meanHeight = .7 * torso_height
    varHeight = .01
    height = np.random.normal(meanHeight, varHeight)
    
    meanW = .35* torso_w
    varW = .1
    width= np.random.normal(meanW, varW)
    
    bottom_left_x = end_x
    bottom_left_y = end_y
    sigma_x = 0
    left_x = np.random.normal(bottom_left_x, sigma_x)
    left_y = np.random.normal(bottom_left_y, sigma_x)
    
    theta = np.random.normal(110, 10)
    
    return left_x, left_y, theta, height, width


def get_right_lower_leg_model(end_x, end_y, torso_height, torso_w):
    meanHeight = .7 * torso_height
    varHeight = .01
    height = np.random.normal(meanHeight, varHeight)
    
    meanW = .34 * torso_w
    varW = .1
    width= np.random.normal(meanW, varW)
    
    top_right_x = end_x
    top_right_y = end_y
    sigma_x = 0
    right_x = np.random.normal(top_right_x, sigma_x)
    right_y = np.random.normal(top_right_y, sigma_x)
    
    theta = np.random.normal(70, 10)
    
    return right_x, right_y, theta, height, width


def get_head_model(torso_center_x, torso_center_y, torso_height, torso_w):
    meanHeight = .35 * torso_height
    varHeight = .1
    height = np.random.normal(meanHeight, varHeight)
    
    meanW = .5* torso_w
    varW = .1
    width= np.random.normal(meanW, varW)
    
    top_x = torso_center_x
    top_y = torso_center_y - (.5 * torso_height)

    theta = np.random.normal(270, 5)
    
    return top_x, top_y, theta, height, width


def get_body_data(torso_center_x, torso_center_y, torso_theta, torso_height, torso_w):
    ############################## draw upper legs#####################################
    xll, yll, thetall, hll, wll = get_left_upper_leg_model(torso_center_x, torso_center_y, torso_theta,torso_height, torso_w)
    left_upper_leg_data = (xll, yll, thetall, hll, wll)
    endy_left_top_leg = yll + (hll * math.sin(math.radians(thetall)))
    endx_left_top_leg = xll + (hll * math.cos(math.radians(thetall)))
    xrl, yrl, thetarl, hrl, wrl = get_right_upper_leg_model(torso_center_x, torso_center_y,torso_theta, torso_height, torso_w)
    right_upper_leg_data = (xrl, yrl, thetarl, hrl, wrl)
    endy_right_top_leg = yrl + (hrl * math.sin(math.radians(thetarl)))
    endx_right_top_leg = xrl + (hrl * math.cos(math.radians(thetarl)))
     ############################## draw lower legs#######################################
    xlll, ylll, thetalll, hlll, wlll = get_left_lower_leg_model(endx_left_top_leg, endy_left_top_leg, torso_height, torso_w)
    left_lower_leg_data = (xlll, ylll, thetalll, hlll, wlll)
    
    xrll, yrll, thetarll, hrll, wrll = get_right_lower_leg_model(endx_right_top_leg, endy_right_top_leg, torso_height, torso_w)
    right_lower_leg_data = (xrll, yrll, thetarll, hrll, wrll)
    ########################draw upper arms####################################
    xla, yla, thetala, hla, wla = get_left_upper_arm_model(torso_center_x, torso_center_y,torso_theta, torso_height, torso_w)
    left_upper_arm_data = (xla, yla, thetala, hla, wla)
    endy_left_top_arm = yla + (hla * math.sin(math.radians(thetala)))
    endx_left_top_arm = xla + (hla * math.cos(math.radians(thetala)))
    
    xra, yra, thetara, hra, wra = get_right_upper_arm_model(torso_center_x, torso_center_y,torso_theta, torso_height, torso_w)
    right_upper_arm_data = (xra, yra, thetara, hra, wra)
    endy_right_top_arm = yra + (hra * math.sin(math.radians(thetara)))
    endx_right_top_arm = xra + (hra * math.cos(math.radians(thetara)))
    ###########################draw lower arms####################################
    xrla, yrla, thetarla, hrla, wrla = get_left_lower_arm_model(endx_left_top_arm, endy_left_top_arm, torso_height, torso_w)
    left_lower_arm_data = (xrla, yrla, thetarla, hrla, wrla)

    xlla, ylla, thetalla, hlla, wlla = get_right_lower_arm_model(endx_right_top_arm, endy_right_top_arm, torso_height, torso_w)
    right_lower_arm_data = (xlla, ylla, thetalla, hlla, wlla)
    ##########################draw Head#############################################
    x, y, theta, h, w = get_head_model(torso_center_x, torso_center_y, torso_height, torso_w)
    head_data = (x, y, theta, h, w)
    
    return head_data, left_upper_leg_data, right_upper_leg_data, left_lower_leg_data, right_lower_leg_data, left_upper_arm_data, right_upper_arm_data, left_lower_arm_data, right_lower_arm_data


def get_body_tree():
  torso = BodyPart('Torso')
  head = BodyPart('Head')
  left_upper_arm = BodyPart('Left Upper Arm')
  right_upper_arm = BodyPart('Right Upper Arm')
  left_upper_leg = BodyPart('Left Upper Leg')
  right_upper_leg = BodyPart('Right Upper Leg')
  left_lower_arm = BodyPart('Left Lower Arm')
  right_lower_arm = BodyPart('Right Lower Arm')
  left_lower_leg = BodyPart('Left Lower Leg')
  right_lower_leg = BodyPart('Right Lower Leg')

  left_lower_arm.setParent(left_upper_arm)
  right_lower_arm.setParent(right_upper_arm)
  left_lower_leg.setParent(left_upper_leg)
  right_lower_leg.setParent(right_upper_leg)

  left_upper_arm.addChildren([left_lower_arm])
  right_upper_arm.addChildren([right_lower_arm])
  left_upper_leg.addChildren([left_lower_leg])
  right_upper_leg.addChildren([right_lower_leg])

  head.setParent(torso)
  left_upper_arm.setParent(torso)
  right_upper_arm.setParent(torso)
  left_upper_leg.setParent(torso)
  right_upper_leg.setParent(torso)

  torso.addChildren([head, left_upper_arm, right_upper_arm, left_upper_leg, right_upper_leg])

  return torso, head, left_upper_arm, right_upper_arm, left_upper_leg, right_upper_leg, left_lower_arm, right_lower_arm, left_lower_leg, right_lower_leg


def get_left_upper_corner(x, y, theta, l, w, center=True):
  if center:
    x_left_upper_corner = int(x + l/2.0 * math.sin(math.radians(theta)) - w/2.0 * math.cos(math.radians(theta)))
    y_left_upper_corner = int(y - l/2.0 * math.cos(math.radians(theta)) - w/2.0 * math.sin(math.radians(theta)))
  else:
    cx = x + (0.5 * l * math.cos(math.radians(theta)))
    cy = y + (0.5 * l * math.sin(math.radians(theta)))
    return get_left_upper_corner(cx, cy, theta - 90, l, w)

  return (x_left_upper_corner, y_left_upper_corner)


def get_right_upper_corner(x, y, theta, l, w, center=True):
  if center:
    x_right_upper_corner = int(x + l/2.0 * math.sin(math.radians(theta)) + w/2.0 * math.cos(math.radians(theta)))
    y_right_upper_corner = int(y - l/2.0 * math.cos(math.radians(theta)) + w/2.0 * math.sin(math.radians(theta)))
  else:
    cx = x + (0.5 * l * math.cos(math.radians(theta)))
    cy = y + (0.5 * l * math.sin(math.radians(theta)))
    return get_right_upper_corner(cx, cy, theta - 90, l, w)

  return (x_right_upper_corner, y_right_upper_corner)


def get_left_lower_corner(x, y, theta, l, w, center=True):
  if center:
    x_left_lower_corner = int(x - l/2.0 * math.sin(math.radians(theta)) - w/2.0 * math.cos(math.radians(theta)))
    y_left_lower_corner = int(y + l/2.0 * math.cos(math.radians(theta)) - w/2.0 * math.sin(math.radians(theta)))
  else:
    cx = x + (0.5 * l * math.cos(math.radians(theta)))
    cy = y + (0.5 * l * math.sin(math.radians(theta)))
    return get_left_lower_corner(cx, cy, theta - 90, l, w)

  return (x_left_lower_corner, y_left_lower_corner)


def get_right_lower_corner(x, y, theta, l, w, center=True):
  if center:
    x_right_lower_corner = int(x - l/2.0 * math.sin(math.radians(theta)) + w/2.0 * math.cos(math.radians(theta)))
    y_right_lower_corner = int(y + l/2.0 * math.cos(math.radians(theta)) + w/2.0 * math.sin(math.radians(theta)))
  else:
    cx = x + (0.5 * l * math.cos(math.radians(theta)))
    cy = y + (0.5 * l * math.sin(math.radians(theta)))
    return get_right_lower_corner(cx, cy, theta - 90, l, w)

  return (x_right_lower_corner, y_right_lower_corner)


def draw_rectangle(img, left_upper_corner, right_upper_corner, left_lower_corner, right_lower_corner):
  (x_left_upper_corner,  y_left_upper_corner)  = left_upper_corner
  (x_right_upper_corner, y_right_upper_corner) = right_upper_corner
  (x_left_lower_corner,  y_left_lower_corner)  = left_lower_corner
  (x_right_lower_corner, y_right_lower_corner) = right_lower_corner

  if len(img.shape) == 2:
    image = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
  else:
    image = img
  image = cv.line(image, (x_left_upper_corner, y_left_upper_corner), (x_right_upper_corner, y_right_upper_corner), color=(255,0,0), thickness=2)
  image = cv.line(image, (x_left_upper_corner, y_left_upper_corner), (x_left_lower_corner, y_left_lower_corner), color=(255,0,0), thickness=2)
  image = cv.line(image, (x_right_upper_corner, y_right_upper_corner), (x_right_lower_corner, y_right_lower_corner), color=(255,0,0), thickness=2)
  image = cv.line(image, (x_left_lower_corner, y_left_lower_corner), (x_right_lower_corner, y_right_lower_corner), color=(255,0,0), thickness=2)
  
  return image


def draw_bounding_lines(img, data):
  left_upper_corner = get_left_upper_corner(*data)
  right_upper_corner = get_right_upper_corner(*data)
  left_lower_corner = get_left_lower_corner(*data)
  right_lower_corner = get_right_lower_corner(*data)

  (x_left_upper_corner,  y_left_upper_corner)  = left_upper_corner
  (x_right_upper_corner, y_right_upper_corner) = right_upper_corner
  (x_left_lower_corner,  y_left_lower_corner)  = left_lower_corner
  (x_right_lower_corner, y_right_lower_corner) = right_lower_corner

  if len(img.shape) == 2:
    image = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
  else:
    image = img
  image = cv.line(image, (x_left_upper_corner, y_left_upper_corner), (x_right_upper_corner, y_right_upper_corner), color=(255,0,0), thickness=2)
  image = cv.line(image, (x_left_upper_corner, y_left_upper_corner), (x_left_lower_corner, y_left_lower_corner), color=(255,0,0), thickness=2)
  image = cv.line(image, (x_right_upper_corner, y_right_upper_corner), (x_right_lower_corner, y_right_lower_corner), color=(255,0,0), thickness=2)
  image = cv.line(image, (x_left_lower_corner, y_left_lower_corner), (x_right_lower_corner, y_right_lower_corner), color=(255,0,0), thickness=2)

  return image


def face_detection(image, foreground_image):
  gray  = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
  faceCascade = cv.CascadeClassifier("frontface_info.xml")
  faces = faceCascade.detectMultiScale(
      gray,
      scaleFactor=1.1,
      minNeighbors=3,
      minSize=(30, 30),
      flags = cv.CASCADE_SCALE_IMAGE
  )
  # Draw a rectangle around the faces
  for (x, y, w, h) in faces:  
      cv.rectangle(image, (x, y), (x+w, y+h), (255,0 , 0), 2)

  return faces


def fix_torso(img, torso):
  x, y, theta, l, w = torso.getData()
  (x_left_upper_corner,  y_left_upper_corner)  = torso.left_upper_corner
  (x_right_upper_corner, y_right_upper_corner) = torso.right_upper_corner
  (x_left_lower_corner,  y_left_lower_corner)  = torso.left_lower_corner
  (x_right_lower_corner, y_right_lower_corner) = torso.right_lower_corner
  
  min_y = min(y_left_upper_corner, y_right_upper_corner)
  max_y = max(y_left_lower_corner, y_right_lower_corner)

  min_x = min(x_left_upper_corner, x_left_lower_corner)
  max_x = max(x_right_upper_corner, x_right_lower_corner)

  white_pixels_above_shoulders = 0
  for i, r in enumerate(img):
    if i == min_y:
      break
    white_pixels_above_shoulders += np.count_nonzero(r == 255)

  white_pixels_in_torso = 0
  for i in range(min_y, max_y + 1):
    for j in range(min_x, max_x + 1):
      white_pixels_in_torso += bool(img[i, j])


  ratio = white_pixels_above_shoulders * 1.0 / white_pixels_in_torso
  if ratio > 0.5:
    right_num_pixels = 0.09 * white_pixels_in_torso
    ratio = right_num_pixels * 1.0 / white_pixels_above_shoulders
    y = int((1 - ratio) * y)
  
  new_torso_data = (x, y, theta, l, w)
  torso.setData(*new_torso_data)


def get_values_within_box(img, points):
  # define polygon points
  points = np.array([points], dtype=np.int32)

  # draw polygon on input to visualize
  gray = img.copy()
  img_poly = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)
  cv.polylines(img_poly, [points], True, (0,0,255), 1)

  # create mask for polygon
  mask = np.zeros_like(gray)
  cv.fillPoly(mask, [points], (255))

  # get color values in gray image corresponding to where mask is white
  values = gray[np.where(mask == 255)]

  return values


def create_mask_body_part(img, points):
  points = np.array([points], dtype=np.int32)
  
  mask = np.zeros_like(img)
  cv.fillPoly(mask, [points], (255))
  return mask


def get_intersection_with_body_parts(img, index, body_parts_list):
  main_body_part = body_parts_list[index]
  lu, ru, rl, ll = main_body_part.left_upper_corner, main_body_part.right_upper_corner, main_body_part.right_lower_corner, main_body_part.left_lower_corner
  points = [lu, ru, rl, ll]
  main_part = create_mask_body_part(img, points)
  intersection = 0
  for i in range(0, len(body_parts_list)):
    if i != index:
      body_part = body_parts_list[i]
      points = [body_part.left_upper_corner, body_part.right_upper_corner, body_part.right_lower_corner, body_part.left_lower_corner]
      part = create_mask_body_part(img, points)
      intersection += np.sum(np.logical_and(part, main_part))
  
  return intersection


def update_importance(img, index, body_parts_list, weight):
  body_part = body_parts_list[index]
  lu = body_part.left_upper_corner
  ru = body_part.right_upper_corner
  ll = body_part.left_lower_corner
  rl = body_part.right_lower_corner

  values = get_values_within_box(img, [lu, ru, rl, ll])
  
  black_pixels = len(values[values == 0])
  white_pixels = len(values[values == 255])
  body_part.Si = white_pixels
  all_pixels = len(values)
  
  # get straight bounding box
  x_min = min(lu[0], ru[0], ll[0], rl[0])
  x_max = max(lu[0], ru[0], ll[0], rl[0])
  y_min = min(lu[1], ru[1], ll[1], rl[1])
  y_max = max(lu[1], ru[1], ll[1], rl[1])

  rect_points = [(x_min, y_min), (x_min, y_max), (x_max, y_max), (x_max, y_min)]
  rect_values = get_values_within_box(img, rect_points)

  # number of white pixels in bounding box, and not in body part
  minus_white_pixels = len(rect_values[rect_values == 255]) - white_pixels

  intersection = get_intersection_with_body_parts(img, index, body_parts_list)

  importance = (black_pixels / all_pixels) + (weight * ((minus_white_pixels * (all_pixels + intersection)) / (all_pixels**2)))
  
  # if body_part.name == 'Torso':
  #   importance *= 1.7
  if body_part.name == 'Left Upper Arm' or body_part.name == 'Right Upper Arm' or body_part.name == 'Left Upper Leg' or body_part.name == 'Right Upper Leg':
    importance *= 2.2

  body_part.priority = importance


def update_all_priorities(img, body_parts_list, w):
  for i in range(len(body_parts_list)):
    update_importance(img, i, body_parts_list, w)


def update_child(bp):
  if(bp.name == 'Torso'):
    for i in range(0,5):
        child_bp = bp.children[i]
        if(child_bp.name=='Head'): 
            x = bp.x
            y = bp.y - (.5 * bp.l)
        elif(child_bp.name=='Left Upper Arm'):
            (x,y) = get_left_upper_corner(bp.x, bp.y, bp.theta, bp.l, bp.w,True)
            y = y+(.5 * child_bp.w)    
        elif(child_bp.name=='Right Upper Arm'): 
            (x,y) = get_right_upper_corner(bp.x, bp.y, bp.theta, bp.l, bp.w,True)
            y = y + (.5 * child_bp.w)
        elif(child_bp.name=='Left Upper Leg'):
            (x,y) = get_left_lower_corner(bp.x, bp.y, bp.theta, bp.l, bp.w,True)
            x = x+(.5 * child_bp.w)
        else:
            (x,y) = get_right_lower_corner(bp.x, bp.y, bp.theta, bp.l, bp.w,True)
            x = x - (.5 * child_bp.w)
        child_bp.setData(x, y, child_bp.theta, child_bp.l, child_bp.w)
            
  else:
    y = bp.y + (bp.l * math.sin(math.radians(bp.theta)))
    x = bp.x + (bp.l * math.cos(math.radians(bp.theta)))
    if bp.name == 'head':
        pass
    child_bp = bp.children[0]
    child_bp.setData(x, y, child_bp.theta, child_bp.l, child_bp.w)


def update_parent(bp):
  parent_bp = bp.parent
  y = bp.y - (parent_bp.l * math.sin(math.radians(parent_bp.theta)))
  x = bp.x - (parent_bp.l * math.cos(math.radians(parent_bp.theta)))
  parent_bp.setData(x, y, parent_bp.theta, parent_bp.l, parent_bp.w)


def total_overlap(img , body_parts_list):
  intersection = 0
  for i in range(0, len(body_parts_list)):
    for j in range(1,len(body_parts_list)):
      body_part = body_parts_list[i]
      body_part2 = body_parts_list[j]
      points = [body_part.left_upper_corner, body_part.right_upper_corner, body_part.right_lower_corner, body_part.left_lower_corner]
      points2 = [body_part2.left_upper_corner, body_part2.right_upper_corner, body_part2.right_lower_corner, body_part2.left_lower_corner]
      part = create_mask_body_part(img, points)
      part2 = create_mask_body_part(img, points2)
      intersection += np.sum(np.logical_and(part, part2))
    
  return intersection


def get_posterior_probability(img, foreground_area, beta, body_parts_list):
  Si = 0
  pose_area = 0
  for bp in body_parts_list:
    Si += bp.Si
    pose_area += bp.area
  
  Su = pose_area + foreground_area - Si
  So = total_overlap(img, body_parts_list)
  return np.exp((Si - beta * So) * 1.0 / Su)


def rotate_image(img, angle):
  return imutils.rotate_bound(img, -angle)


def draw_all(img,body_parts_list):
    image = img
    for i in range(0, len(body_parts_list)):
        body_part = body_parts_list[i]
        image = draw_rectangle(image, body_part.left_upper_corner, body_part.right_upper_corner, body_part.left_lower_corner, body_part.right_lower_corner)
    return image


def draw_video_frame(img, body_parts_list, i, save_path):
    img[img == 255] = 0
    img_skeleton = create_skeleton(img,body_parts_list)
    
    name = save_path + '/' + str(i).zfill(7) + '.jpg'
    print("Saving frame", name)
    cv.imwrite(name, img_skeleton)


def create_skeleton(image,body_model):
  points = []
  positions = np.zeros(14) #head,torso,left_arm_up,left_arm_mid,left_arm_down,right_arm_up,right_arm_mid,right_arm_down,left_leg_mid,left_leg_down,right_leg_up,right_leg_mid,right_leg_down
  new_image = np.zeros((image.shape[0],image.shape[1],image.shape[2]))
  black_pixels = np.where(
    (new_image[:, :, 0] == 0) & 
    (new_image[:, :, 1] == 0) & 
    (new_image[:, :, 2] == 0))

  # set those pixels to white
  new_image[black_pixels] = [255, 255, 255]

  indx = 0 
  for i in range(0,len(body_model)):
      if body_model[i].name =='Head':
        new_image = cv.circle(new_image, (int(body_model[i].x),int(body_model[i].y)), 3, (0, 0, 255), 3)
        points.append((int(body_model[i].x),int(body_model[i].y)))
        positions[0] = indx 
        indx = indx+1
      elif body_model[i].name == 'Torso':
        new_image = cv.circle(new_image, (int(body_model[i].x),int(body_model[i].y)), 3, (0, 0, 255), 3)
        points.append((int(body_model[i].x),int(body_model[i].y)))
        positions[1] = indx 
        indx = indx+1
      elif body_model[i].name == 'Left Upper Arm':
        new_image = cv.circle(new_image, (int(body_model[i].x),int(body_model[i].y)), 3, (0, 128, 255), 3)
        points.append((int(body_model[i].x),int(body_model[i].y)))
        positions[2] = indx 
        indx = indx+1
      elif body_model[i].name == 'Left Lower Arm':
        new_image = cv.circle(new_image, (int(body_model[i].x),int(body_model[i].y)), 3, (0, 128, 255), 3)
        left_hand = (int((body_model[i].left_lower_corner[0]+body_model[i].right_lower_corner[0])/2),int((body_model[i].left_lower_corner[1]+body_model[i].right_lower_corner[1])/2))
        positions[3] = indx 
        positions[4] = indx+1 
        indx = indx+2
        new_image = cv.circle(new_image, left_hand, 3, (153,204,255), 3)
        points.append((int(body_model[i].x),int(body_model[i].y)))
        points.append(left_hand)
      
      elif body_model[i].name == 'Right Upper Arm':
        new_image = cv.circle(new_image, (int(body_model[i].x),int(body_model[i].y)), 3, (51,255,255), 3)
        points.append((int(body_model[i].x),int(body_model[i].y)))    
        positions[5] = indx
        indx = indx+1
      elif body_model[i].name == 'Right Lower Arm':
        new_image = cv.circle(new_image, (int(body_model[i].x),int(body_model[i].y)), 3, (51,255,255), 3)
        right_hand = (int((body_model[i].left_lower_corner[0]+body_model[i].right_lower_corner[0])/2),int((body_model[i].left_lower_corner[1]+body_model[i].right_lower_corner[1])/2))
        new_image = cv.circle(new_image, right_hand, 3, (51,255,255), 3)
        points.append((int(body_model[i].x),int(body_model[i].y)))
        points.append(right_hand)
        positions[6] = indx 
        positions[7] = indx+1 
        indx = indx+2
    
      elif body_model[i].name == 'Left Upper Leg':
        new_image = cv.circle(new_image, (int(body_model[i].x),int(body_model[i].y)), 3, (255,51 , 51), 3)
        points.append((int(body_model[i].x),int(body_model[i].y)))    
        positions[8] = indx
        indx = indx+1
      elif body_model[i].name == 'Left Lower Leg':
        new_image = cv.circle(new_image, (int(body_model[i].x),int(body_model[i].y)), 3, (255,178 , 102), 3)
        left_leg = (int((body_model[i].left_lower_corner[0]+body_model[i].right_lower_corner[0])/2),int((body_model[i].left_lower_corner[1]+body_model[i].right_lower_corner[1])/2))
        new_image = cv.circle(new_image, left_leg, 3, (255,178 , 102), 3)
        points.append((int(body_model[i].x),int(body_model[i].y)))
        points.append(left_leg)
        positions[9] = indx 
        positions[10] = indx+1 
        indx = indx+2
    
      elif body_model[i].name == 'Right Upper Leg':
        new_image = cv.circle(new_image, (int(body_model[i].x),int(body_model[i].y)), 3, (51,255,153), 3)
        points.append((int(body_model[i].x),int(body_model[i].y)))    
        positions[11] = indx
        indx = indx+1
      elif body_model[i].name == 'Right Lower Leg':
        new_image = cv.circle(new_image, (int(body_model[i].x),int(body_model[i].y)), 3, (51,255,51), 3)
        right_leg = (int((body_model[i].left_lower_corner[0]+body_model[i].right_lower_corner[0])/2),int((body_model[i].left_lower_corner[1]+body_model[i].right_lower_corner[1])/2))
        new_image = cv.circle(new_image, right_leg, 3, (51,255,51), 3)
        points.append((int(body_model[i].x),int(body_model[i].y)))
        points.append(right_leg)
        positions[12] = indx 
        positions[13] = indx+1 
        indx = indx+2

  middle_leg = (int((points[int(positions[11])][0]+points[int(positions[8])][0])/2),int((points[int(positions[11])][1]+points[int(positions[8])][1])/2))
  points.append(middle_leg)
  new_image = connect_points(new_image,positions,points)
  #new_image = cv.cvtColor(new_image, cv.COLOR_BGR2RGB)
  # show_image(new_image)

  return new_image


def connect_points(new_image,positions,points):
  #torso connected points
  torso_conn = [0,2,5]
  for i in torso_conn:
    new_image = cv.line(new_image, points[int(positions[i])], points[int(positions[1])], (0,0,255), 3)

  #legs and arm connections points
  l_a   = [2,5,8,11]
  color = [0,0,(0, 128, 255),(153,204,255),(153,204,255),(51,255,255),(51,255,255),(51,255,255),(255,51 , 51),(255,178 , 102),(255,178 , 102),(51,255,153),(51,255,51),(51,255,51)]
  for i in l_a:
    new_image = cv.line(new_image, points[int(positions[i])], points[int(positions[i+1])], color[i], 3)
    new_image = cv.line(new_image, points[int(positions[i+1])], points[int(positions[i+2])], color[i+1], 3)
  
  new_image = cv.line(new_image, points[14], points[int(positions[8])], (0,0,255), 3)
  new_image = cv.line(new_image, points[14], points[int(positions[11])], (0,0,255), 3)
  new_image = cv.line(new_image, points[14], points[int(positions[1])], (0,0,255), 3)
  return new_image
  

def init_visited(body_parts_list):
    for i in range(0, len(body_parts_list)):
        body_parts_list[i].visited=0
 

def change_value(body_part):
  #index = np.random.randint(0, 5)
    
  if(body_part.name=='Torso'):
    index = np.random.randint(0, 3)
    lamdas = [7, 5, 4, 2, 2]
  else:
    index=2
    lamdas = [7, 5, 10, 2, 2]
  eps = np.random.normal(0, lamdas[index])
  body_part.updateValue(index, eps)
  
        
  return index, eps


def get_TFH(image,face,height):
    face_center = (int(face[0]+face[2]/2),int(face[1]+face[3]/2))
    image = cv.circle(image, face_center, 3, (255,0 , 0), 3)
    shoulder = round(face[1]+face[3])
    image = cv.circle(image, (int(face_center[0]),int(shoulder)), 3, (255,0 , 0), 3)
    torso_center = (int(face_center[0]),int(round(shoulder+height/2)))
    image = cv.circle(image, torso_center, 3, (255,0 , 0), 3)
    return torso_center[0],torso_center[1]


def initial_pose(image_R, foreground_image, faces, foreground_area):
  torso, head, left_upper_arm, right_upper_arm, left_upper_leg, right_upper_leg, left_lower_arm, right_lower_arm, left_lower_leg, right_lower_leg = get_body_tree()
  torso_data = get_torso_model(image_R,faces[0],foreground_image)
  torso.setData(*torso_data)
  torso_data = torso.getData()
  head_data, left_upper_leg_data, right_upper_leg_data, left_lower_leg_data, right_lower_leg_data, left_upper_arm_data, right_upper_arm_data, left_lower_arm_data, right_lower_arm_data = get_body_data(*torso_data)

  head.setData(*head_data)
  left_upper_arm.setData(*left_upper_arm_data)
  right_upper_arm.setData(*right_upper_arm_data)
  left_upper_leg.setData(*left_upper_leg_data)
  right_upper_leg.setData(*right_upper_leg_data)
  left_lower_arm.setData(*left_lower_arm_data)
  right_lower_arm.setData(*right_lower_arm_data)
  left_lower_leg.setData(*left_lower_leg_data)
  right_lower_leg.setData(*right_lower_leg_data)

  body_parts_list = [torso, left_upper_arm, left_lower_arm, right_upper_arm, right_lower_arm, left_upper_leg, left_lower_leg, right_upper_leg, right_lower_leg, head]

  draw_all(foreground_image, body_parts_list)

  w = 0.2
  update_all_priorities(foreground_image, body_parts_list, w)

  bp_priority_based = body_parts_list.copy()
  bp_priority_based = sorted(bp_priority_based, reverse=True, key=lambda x: x.priority)

  posterior_prob = get_posterior_probability(foreground_image, foreground_area, beta, body_parts_list)
  return body_parts_list,bp_priority_based,posterior_prob


def build_pose(image_R, foreground_image, body_parts_list, bp_priority_based, posterior_prob, step, frame_num, foreground_area, w, save_path):
  if(step ==1):
    limit = 20
  else:
    limit=15

  bp = 1
  update_all_priorities(foreground_image, body_parts_list, w)
  posterior_prob = get_posterior_probability(foreground_image, foreground_area, beta, body_parts_list)
  bp_priority_based = sorted(bp_priority_based, reverse=True, key=lambda x: x.priority)
  init_visited(body_parts_list)
  
  for i in range(limit):
    if i < 9:
      bp = body_parts_list[i]
    else:
      bp = bp_priority_based[0]
      if bp_priority_based[0].priority < 0.71:
        break
  

    bp.visited += 1
    
    for k in range(20):
      posterior_prob =get_posterior_probability(foreground_image, foreground_area, beta, body_parts_list) 
      cur_priority = bp.priority
      cur_post = get_posterior_probability(foreground_image, foreground_area, beta, body_parts_list)
      if(((bp.priority > 1 and bp.visited>3) or (bp.priority > 0.95 and bp.visited>6) ) and step ==1   ):
          bp.updateValue(j,45)
          update_all_priorities(foreground_image, body_parts_list, w)
          temp_posterior = get_posterior_probability(foreground_image, foreground_area, beta, body_parts_list)
          bp.updateValue(j,-90)
          update_all_priorities(foreground_image, body_parts_list, w)
          new_posterior = get_posterior_probability(foreground_image, foreground_area, beta, body_parts_list)

          if(new_posterior < temp_posterior and cur_priority>1.3):
              bp.updateValue(j,90)
              update_all_priorities(foreground_image, body_parts_list, w)
          elif(cur_priority<1.3 and cur_post > new_posterior and cur_post>temp_posterior ):
              bp.updateValue(j,-45)
              update_all_priorities(foreground_image, body_parts_list, w)
              
      j, diff = change_value(bp)
      if bp.name == 'Torso' or bp.name == 'Left Upper Arm' or bp.name == 'Right Upper Arm' or bp.name == 'Left Upper Leg' or bp.name == 'Right Upper Leg':
          update_child(bp)
      update_all_priorities(foreground_image, body_parts_list, w)
        
      new_posterior = get_posterior_probability(foreground_image, foreground_area, beta, body_parts_list)

      if new_posterior <= posterior_prob:
        bp.updateValue(j, -2 * diff)

        if bp.name == 'Torso' or bp.name == 'Left Upper Arm' or bp.name == 'Right Upper Arm' or bp.name == 'Left Upper Leg' or bp.name == 'Right Upper Leg':
              update_child(bp)
        update_all_priorities(foreground_image, body_parts_list, w)
        new_posterior = get_posterior_probability(foreground_image, foreground_area, beta, body_parts_list)
        if new_posterior < posterior_prob:
          bp.updateValue(j, diff)
          if bp.name == 'Torso' or bp.name == 'Left Upper Arm' or bp.name == 'Right Upper Arm' or bp.name == 'Left Upper Leg' or bp.name == 'Right Upper Leg':
              update_child(bp)
          update_all_priorities(foreground_image, body_parts_list, w)

        else:
          posterior_prob = new_posterior
      else:
        posterior_prob = new_posterior
    if bp.name == 'Torso' or bp.name == 'Left Upper Arm' or bp.name == 'Right Upper Arm' or bp.name == 'Left Upper Leg' or bp.name == 'Right Upper Leg':
      update_child(bp)
    
    draw_all(foreground_image, body_parts_list)
    update_all_priorities(foreground_image, body_parts_list, w)
    bp_priority_based = sorted(bp_priority_based, reverse=True, key=lambda x: x.priority)
  
  draw_all(foreground_image, body_parts_list)
  
  best_body_parts_list = body_parts_list.copy()
  image = draw_all(foreground_image, best_body_parts_list)
  draw_video_frame(image, body_parts_list, frame_num, save_path)
  return body_parts_list, bp_priority_based, posterior_prob


def complete_video(frames_names_list, foreground_names_list, prev_frames, body_parts_list, bp_priority_based, posterior_prob, main_number, foreground_area, w, save_path):
  frames_count = len(frames_names_list)
  main_body_parts_list = body_parts_list.copy()
  main_bp_priority_based = bp_priority_based.copy()
  main_posterior_prob = posterior_prob.copy()
  for i in range(main_number - 1, -1, -1):
    frame = prev_frames[i]
    foreground_image = cv.imread(foreground_names_list[i])
    foreground_image = cv.cvtColor(foreground_image, cv.COLOR_RGB2GRAY)
    body_parts_list, bp_priority_based, posterior_prob = build_pose(frame, foreground_image, body_parts_list, bp_priority_based, posterior_prob, 2, i, foreground_area, w, save_path)
  
  body_parts_list = main_body_parts_list.copy()
  bp_priority_based = main_bp_priority_based.copy()
  posterior_prob = main_posterior_prob.copy()
  for i in range(main_number+1, frames_count):
    frame = cv.imread(frames_names_list[i])
    foreground_image = cv.imread(foreground_names_list[i])
    foreground_image = cv.cvtColor(foreground_image, cv.COLOR_RGB2GRAY)
    body_parts_list, bp_priority_based, posterior_prob = build_pose(frame, foreground_image, body_parts_list, bp_priority_based, posterior_prob, 2, i, foreground_area, w, save_path)


def get_poses(frames_path, segmented_frames_path, poses_path):
  if not os.path.exists(poses_path):
    os.makedirs(poses_path)
  
  frames_names = os.listdir(frames_path)
  frames_names = sorted(frames_names)
  
  segmented_frames_names = os.listdir(segmented_frames_path)
  segmented_frames_names = sorted(segmented_frames_names)
  
  frames_count = len(segmented_frames_names)
  
  frames_full_names = [frames_path + '/' + frames_names[i] for i in range(frames_count)]
  segmented_full_names = [segmented_frames_path + '/' + segmented_frames_names[i] for i in range(frames_count)]
  
  prev_frames = []
  for i in range(frames_count):
    frame = cv.imread(frames_full_names[i])
    prev_frames.append(frame)
    print("processing ", frames_path, '/', frames_names[i], sep='')
    foreground_image = cv.imread(segmented_full_names[i], cv.IMREAD_COLOR)
    foreground_area = np.count_nonzero(foreground_image[foreground_image == 255])
    faces = face_detection(frame, foreground_image)
    if faces != ():
        main_number = i
        image_R = frame.copy()
        foreground_image = cv.cvtColor(foreground_image, cv.COLOR_RGB2GRAY)
        break

  torso, head, left_upper_arm, right_upper_arm, left_upper_leg, right_upper_leg, left_lower_arm, right_lower_arm, left_lower_leg, right_lower_leg = get_body_tree()
  torso_data = get_torso_model(image_R,faces[0],foreground_image)
  torso.setData(*torso_data)
  torso_data = torso.getData()
  head_data, left_upper_leg_data, right_upper_leg_data, left_lower_leg_data, right_lower_leg_data, left_upper_arm_data, right_upper_arm_data, left_lower_arm_data, right_lower_arm_data = get_body_data(*torso_data)

  head.setData(*head_data)
  left_upper_arm.setData(*left_upper_arm_data)
  right_upper_arm.setData(*right_upper_arm_data)
  left_upper_leg.setData(*left_upper_leg_data)
  right_upper_leg.setData(*right_upper_leg_data)
  left_lower_arm.setData(*left_lower_arm_data)
  right_lower_arm.setData(*right_lower_arm_data)
  left_lower_leg.setData(*left_lower_leg_data)
  right_lower_leg.setData(*right_lower_leg_data)

  body_parts_list = [torso, left_upper_arm, left_lower_arm, right_upper_arm, right_lower_arm, left_upper_leg, left_lower_leg, right_upper_leg, right_lower_leg, head]

  draw_all(foreground_image, body_parts_list)

  w = 0.2
  update_all_priorities(foreground_image, body_parts_list, w)

  bp_priority_based = body_parts_list.copy()
  bp_priority_based = sorted(bp_priority_based, reverse=True, key=lambda x: x.priority)
  
  posterior_prob = get_posterior_probability(foreground_image, foreground_area, beta, body_parts_list)

  body_parts_list, bp_priority_based, posterior_prob = initial_pose(image_R, foreground_image, faces, foreground_area)
  body_parts_list, bp_priority_based, posterior_prob = build_pose(image_R, foreground_image, body_parts_list, bp_priority_based, posterior_prob, 1, main_number,  foreground_area, w, poses_path)

  main_body_parts_list1 = body_parts_list.copy()
  main_bp_priority_based1 = bp_priority_based.copy()
  main_posterior_prob1 = posterior_prob.copy()

  complete_video(frames_full_names, segmented_full_names, prev_frames, body_parts_list, bp_priority_based, posterior_prob, main_number, foreground_area, w, poses_path)

