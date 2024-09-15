import numpy as np
from vgn.utils.transform import Transform, Rotation

affrdnce_label_dict = {
    'handover': 0,
    'cut': 1,
    'stab': 2,
    'lift': 3,
    'wrap': 4,
    'pour': 5,
    'wear': 6
}


affrdnce_label_color_dict = {
    'handover': np.array([125/255, 202/255, 92/255]),
    'cut': np.array([1.0, 0.0, 0.0]), # red
    'stab': np.array([1.0, 0.0, 0.0]), # red
    'lift': np.array([1.0, 1.0, 0.0]), # yellow
    'wrap': np.array([0.0, 1.0, 0.0]), # green
    'pour': np.array([0.0, 1.0, 1.0]), # light blue
    'wear': np.array([0.0, 0.0, 1.0]), # dark blue
}


def aff_labels_to_names(aff_labels):
    aff_names = []
    for key, value in affrdnce_label_dict.items():
        if value in aff_labels:
            aff_names.append(key)
    return aff_names

def aff_labels_to_colors(aff_labels):
    aff_colors = []
    for key, value in affrdnce_label_dict.items():
        if value in aff_labels:
            aff_colors.append(affrdnce_label_color_dict[key])
    return aff_colors

def viz_color_legend():
    # TODO: Do this in a better way
    # import matplotlib.pyplot as plt
    # import matplotlib.patches as mpatches
    # import time
    # # Create a list to hold the legend entries
    # legend_entries = []
    # for label, color in affrdnce_label_color_dict.items():
    #     # Create a patch for each entry in the dictionary
    #     legend_entries.append(mpatches.Patch(color=color, label=label))
    # # Plot dummy
    # plt.figure(figsize=(16, 6))
    # plt.axis('off')
    # # Add the legend to the plot
    # plt.legend(handles=legend_entries, loc='center', frameon=False)
    # # Display the plot non-blocking
    # plt.show()
    # time.sleep(0.01)

    # get legend image from path using open3d
    pass
    # legend_o3d_vis = o3d.visualization.Visualizer()
    # legend_img = o3d.io.read_image('aff_legend.png')
    # create a geometry
    # legend_img_geom = o3d.geometry.Image(legend_img)
    # legend_o3d_vis.create_window()
    # legend_o3d_vis.add_geometry(legend_img_geom)
    # legend_o3d_vis.poll_events()
    

def get_affordance(sim, aff_dataset, grasp, mesh_pose_list):
    # load correspoinding pointcloud from affnet dataset

    # # DEBUG:
    # print("Loaded dataset with {} objects".format(aff_dataset['num_objects']))
    # print("Num of semantic classes: ", len(aff_dataset["semantic_classes"]))
    # print("Semantic classes: ", aff_dataset["semantic_classes"])
    # print("Num of affordances: ", len(aff_dataset["affordance_classes"]))
    # print("All affordances: ", aff_dataset["affordance_classes"])


    ## Assign affordance to grasp
    grasp_center = grasp.pose.translation
    # Unfortunately VGN/GIGA grasps are not in the grasp frame we want (frame similar to PointNetGPD), so we need to transform them
    grasp_frame_rot =  grasp.pose.rotation * Rotation.from_euler('Y', np.pi/2) * Rotation.from_euler('Z', np.pi)
    grasp_tf = Transform(grasp_frame_rot, grasp_center).as_matrix()

    max_x_aff_dist = sim.gripper.finger_depth + 0.0025 # 0.25 cm extra
    min_x_aff_dist = - 0.0025 # 0.25 cm from before the grasp
    max_y_aff_dist = sim.gripper.max_opening_width/2.0 + 0.0025 # 0.25 cm extra
    min_y_aff_dist = -sim.gripper.max_opening_width/2.0 - 0.0025 # 0.25 cm extra
    max_z_aff_dist = 0.01 # 1 cm around the gripper
    min_z_aff_dist = -0.01 # 1 cm around the gripper

    current_aff = []
    current_aff_score = 0.0
    aff_score_type = 'count' # 'sum' or 'count' (count is better)
    considered_points = np.zeros((0,3))

    # loop through the mesh_pose_list
    for mesh_scale_pose in mesh_pose_list:
        mesh_path = mesh_scale_pose[0]
        scale = mesh_scale_pose[1]
        assert scale == 1.0 # Currently only works for scale 1.0
        pose = mesh_scale_pose[2]
        
        # get obj_id
        shape_id = mesh_path.split('/')[6]
        # check which sem class has this shapeid
        datapoint = None
        for sem_class in aff_dataset['semantic_classes']:
            # if key exists
            if shape_id in aff_dataset['data'][sem_class]:
                datapoint = aff_dataset['data'][sem_class][shape_id]
                break
        assert datapoint is not None, "Could not find shape_id {} in any semantic class".format(shape_id)
        # print("Checking object: ", sem_class)

        # load the corresponding pointcloud
        obj_pcl = datapoint['coordinates']
        # transform pcl to pose of the object
        obj_pcl = np.dot(obj_pcl, pose[:3,:3].T) + pose[:3,3]

        ## get indices of points that are "near" the grasp i.e. will be assigned to the grasp
        # convert obj_pcl to grasp frame
        # append one
        obj_pcl_hom = np.hstack((obj_pcl, np.ones((obj_pcl.shape[0], 1))))
        obj_pcl_local = np.linalg.inv(grasp_tf) @ obj_pcl_hom.T
        val_mask = obj_pcl_local[0,:] < max_x_aff_dist # too far X
        val_mask = val_mask & (obj_pcl_local[0,:] > min_x_aff_dist) # too close X
        val_mask = val_mask & (obj_pcl_local[1,:] < max_y_aff_dist) # too far Y
        val_mask = val_mask & (obj_pcl_local[1,:] > min_y_aff_dist) # too close Y
        val_mask = val_mask & (obj_pcl_local[2,:] < max_z_aff_dist) # too far Z
        val_mask = val_mask & (obj_pcl_local[2,:] > min_z_aff_dist) # too close Z
        if val_mask.sum() == 0:
            # print("No points found for this object")
            continue
        considered_points = np.vstack((considered_points, obj_pcl[val_mask]))
        # loop through the affordance labels
        for affdnce in datapoint['labels'].keys():
            # get the points that are near the grasp and have this affordance
            if aff_score_type == 'count':
                # count the number of non-zero scores from 3DAffNet
                affdnce_score = (datapoint['labels'][affdnce][val_mask] > 0.0).sum()
            elif aff_score_type == 'sum':
                affdnce_score = datapoint['labels'][affdnce][val_mask].sum() # sum the scores from 3DAffNet
            else:
                raise NotImplementedError
            # print("Affordance: ", affdnce)
            # print("Affordance score: ", affdnce_score)
            if affdnce_score > current_aff_score:
                current_aff = [affdnce]
                current_aff_score = affdnce_score
            elif (affdnce_score > 0.0) and (affdnce_score == current_aff_score):
                current_aff.append(affdnce)

    # # Debug:
    # print("Affordances assigned to grasp: ", current_aff)
    # print("Affordance score: ", current_aff_score)
    # print("Num of points considered: ", considered_points.shape[0])
    # # import open3d as o3d
    # # considered_points_pc = o3d.geometry.PointCloud()
    # # considered_points_pc.points = o3d.utility.Vector3dVector(considered_points)
    # # considered_points_pc.colors = o3d.utility.Vector3dVector(np.tile(np.array([0.9, 0.45, 0.0]), (np.asarray(considered_points_pc.points).shape[0], 1)))
    # # o3d.visualization.draw_geometries([considered_points_pc])

    # convert current_aff to labels
    affrdnce_labels = np.zeros(len(aff_dataset['affordance_classes']), dtype=np.int32)
    for affdnce in current_aff:
        affrdnce_labels[affrdnce_label_dict[affdnce]] = 1
    
    # print("Affordance labels: ", affrdnce_labels)
    return affrdnce_labels