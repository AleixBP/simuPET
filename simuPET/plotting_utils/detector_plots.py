alpha=0.002;
plt.figure(figsize=[10,10]);

plt.gca().add_patch(plt.Circle((0, 0), boxes.Rx-boxes.shift, fill=False, color='k'));
plt.gca().add_patch(plt.Rectangle((-boxes.Rx, -boxes.Ry), 2*boxes.Rx, 2*boxes.Ry, fill=False, color='k'));
plt.text(0,0, "FOV", ha="center", va="center")

plt.gca().add_patch(plt.Rectangle((boxes.right_box[2], boxes.right_box[0]), boxes.nLayers*boxes.layerThickness, boxes.layerLength, fill=False, color='k'));
plt.gca().add_patch(plt.Rectangle((boxes.top_box[2], boxes.top_box[0]), boxes.layerLength, boxes.nLayers*boxes.layerThickness, fill=False, color='k'));
plt.gca().add_patch(plt.Rectangle((-boxes.top_box[0], boxes.top_box[2]), -boxes.nLayers*boxes.layerThickness, boxes.layerLength, fill=False, color='k'));
plt.gca().add_patch(plt.Rectangle((boxes.right_box[0], -boxes.right_box[2]), boxes.layerLength, -boxes.nLayers*boxes.layerThickness, fill=False, color='k'));

point1 = np.array([boxes.right_box[2]+np.random.rand()*nLayers*boxes.layerThickness, boxes.right_box[0]+np.random.rand()*layerLength])
point2 = np.array([-boxes.top_box[0]-np.random.rand()*nLayers*boxes.layerThickness, boxes.top_box[2]+np.random.rand()*layerLength])
point3 = point2 + 0.5*(point1-point2)
plt.scatter(*point1, color="red")
plt.scatter(*point2, color="red")
plt.plot(*np.vstack((point1, point2)).T, color="red")
plt.scatter(*point3, color="violet")

from matplotlib.collections import LineCollection
x, y = np.meshgrid(np.linspace(*boxes.right_box[2:],boxes.nLayers ), np.linspace(*boxes.right_box[:2], int(boxes.layerLength/boxes.detLength)))

segs1 = np.stack((x,y), axis=2)
segs2 = segs1.transpose(1,0,2)
plt.gca().add_collection(LineCollection(segs1,linewidths=(0.1,), color='k'))
plt.gca().add_collection(LineCollection(segs2,linewidths=(0.1,), color='k'))

plt.scatter(*detections[0].T, alpha=alpha, c='blue');
plt.scatter(*detections[1].T, alpha=alpha, c='blue');
plt.xlabel("x axis [mm]")
plt.ylabel("y axis [mm]")
plt.show()