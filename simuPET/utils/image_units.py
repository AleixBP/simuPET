from simuPET import array_lib as np

def image_to_mm_split(y, x, img, boxes):
    
    N, M = img.shape if hasattr(img, "shape") else img
    xnew =  x*2*boxes.Rx/M - boxes.Rx
    ynew = boxes.Ry - y*2*boxes.Ry/N 

    return np.array([xnew, ynew]) #np.vstack((xnew, ynew))


def mm_to_image_split(x, y, img, boxes):

        N, M = img.shape if hasattr(img, "shape") else img
        xnew = (x + boxes.Rx) * (M/(2*boxes.Rx))
        ynew = N - (y + boxes.Ry) * (N/(2*boxes.Ry))

        return np.rint(np.array([ynew, xnew])).astype(int) #np.vstack((ynew, xnew))


def image_to_mm(yx, img, boxes):
    
    N, M = img.shape if hasattr(img, "shape") else img
    scale = np.array([-2*boxes.Ry/N, 2*boxes.Rx/M])[:, np.newaxis]
    offset = np.array([boxes.Ry, -boxes.Rx])[:, np.newaxis]
    return (scale*yx + offset)[::-1]


def mm_to_image(xy, img, boxes, round=False):

        N, M = img.shape if hasattr(img, "shape") else img
        offset = np.array([boxes.Rx, -boxes.Ry])[:, np.newaxis]
        scale = np.array([M/(2*boxes.Rx), -N/(2*boxes.Ry)])[:, np.newaxis]
        
        #return np.rint(scale*(xy + offset)).astype(int)[::-1]
        yx_new = (scale*(xy + offset))[::-1]
        return yx_new if not round else np.rint(yx_new).astype(int)


def pair_mm_to_image(xy, img, boxes, round=False): #when [[x1,y1],...,[xn,yn]]
    # equivalent to mm_to_image(xy.T, img, boxes).T

        N, M = img.shape if hasattr(img, "shape") else img
        offset = np.array([boxes.Rx, -boxes.Ry])
        scale = np.array([M/(2*boxes.Rx), -N/(2*boxes.Ry)])
        yx_new = (scale*(xy + offset))[:, ::-1]
        
        return yx_new if not round else np.rint(yx_new).astype(int)


def simpler_pair_mm_to_image(xy, shp, radius, round=False):

        alt = np.array([1., -1.])
        offset = alt*radius
        scale = alt*np.array(shp)/(2*radius)
        yx_new = (scale*(xy + offset))[:, ::-1]
        
        return yx_new if not round else np.rint(yx_new).astype(int)


def even_ceil(x):
    return int( 2*(np.ceil(x)//2) )


def odd_ceil(x):
    return even_ceil(x) + 1


def odd_floor(x):
    return even_ceil(x) - 1