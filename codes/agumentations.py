def agumentation(img,domain,x=0,y=0,h=0,w=0):
    '''
    this function helps to mask the different axis
    x - time and y - frequency in a spectrogram

    input enter image as a numpy array,domains( anyone among time,frequency,timeandfrequency),
    x,w for time
    y,h for frequency,
    x,w , y,h for time and frequency
    
    '''
    category_aug = domain

    if domain == 'frequency':
      img[y:y+h,:,:] = 0
    if domain == 'time':
      img[:,x:x+w,:] = 0
    if domain == 'timeandfrequency':
      img[x:x+w,y:y+h,:] =0
    return img  
