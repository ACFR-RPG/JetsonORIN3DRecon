import argparse
import os, time
import pickle
import cv2

import torch
from torch.utils.data import DataLoader
from loguru import logger
from tqdm import tqdm

from models import NeuralRecon
from utils import SaveScene
from config import cfg, update_config
from datasets import find_dataset_def, transforms
from tools.kp_reproject import *
from tools.sync_poses import *
from transforms3d.quaternions import quat2mat

# python zed_demo.py --cfg ./config/zed_demo.yaml
import pyzed.sl as sl
from pprint import pprint
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
import numpy as np

import time
import threading

class DataCollector(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.daemon = True
        self._kill = threading.Event()

    def run(self):
        
        # Configs
        window_size=9
        min_angle=15
        min_distance=0.1
        ori_size=(1920, 1080)
        size=(640, 480)
        plot_x=100
        debug = False # Renders translation graphs

        print("DataCollector starting...")

        if debug:
            # Translation plotting debug
            fig, ax = plt.subplots()
            ax.set(xlim=(0, plot_x), ylim=(-3, 3))

            xarr, yarr, zarr = np.repeat(0, plot_x), np.repeat(0, plot_x), np.repeat(0, plot_x)
            x = np.linspace(0, plot_x, plot_x)
            xline, = ax.plot(x, xarr, 'r', lw=1, animated=True)
            yline, = ax.plot(x, yarr, 'g', lw=1, animated=True)
            zline, = ax.plot(x, zarr, 'b', lw=1, animated=True)
            fig.canvas.draw()
            plt.show(block=False)
            plt.pause(0.1)
            bg = fig.canvas.copy_from_bbox(fig.bbox)
            ax.draw_artist(xline)
            ax.draw_artist(yline)
            ax.draw_artist(zline)
            fig.canvas.blit(fig.bbox)

        # Create a Camera object
        zed = sl.Camera()

        # Create a InitParameters object and set configuration parameters
        init_params = sl.InitParameters()
        init_params.camera_resolution = sl.RESOLUTION.HD1080  # Use HD1080 video mode
        init_params.camera_fps = 60  # Set fps at 60
        init_params.coordinate_units = sl.UNIT.METER
        init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP

        # Open the camera
        err = zed.open(init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            exit(1)

        # Enable tracking
        tracking_params = sl.PositionalTrackingParameters()
        initial_position = sl.Transform()
        # Set the initial position of the Camera Frame at 1m80 above the World Frame
        initial_translation = sl.Translation()
        initial_translation.init_vector(0, 0, 0)
        initial_position.set_translation(initial_translation)
        tracking_params.set_initial_world_transform(initial_position)
        zed.enable_positional_tracking(tracking_params)
        zed_pose = sl.Pose()
        pose_data = sl.Transform()

        # Capture frames until kil
        zed_pose = sl.Pose()
        cam_poses = []
        cam_intrinsics = []
        i = 0
        image = sl.Mat()
        runtime_parameters = sl.RuntimeParameters()

        while(not self.killed()):
            if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
                i += 1
                # A new image is available if grab() returns SUCCESS
                zed.retrieve_image(image, sl.VIEW.LEFT)
                pic = cv2.resize(image.get_data(), size)
                cv2.imwrite(os.path.join(cfg.TEST.PATH, "images" , str(i).zfill(5) + '.jpg'), pic)
                
                properties = zed.get_camera_information().camera_configuration.calibration_parameters.right_cam
                # print(f"fx(px): {properties.fx}")
                # print(f"fy(px): {properties.fy}")
                # print(f"cx(px): {properties.cx}")
                # print(f"cy(px): {properties.cy}")
                cam_intrinsics.append([str(i).zfill(5), properties.fx, properties.fy, properties.cx, properties.cy])

                # Get the pose of the camera relative to the world frame
                state = zed.get_position(zed_pose, sl.REFERENCE_FRAME.WORLD)
                # Display translation and timestamp
                py_translation = sl.Translation()
                tx = zed_pose.get_translation(py_translation).get()[0]
                ty = zed_pose.get_translation(py_translation).get()[1]
                tz = zed_pose.get_translation(py_translation).get()[2]
                # print("Translation: tx: {0}, ty:  {1}, tz:  {2}, timestamp: {3}\n".format(tx, ty, tz, zed_pose.timestamp.get_milliseconds()))
                #Display orientation quaternion
                py_orientation = sl.Orientation()
                ox = zed_pose.get_orientation(py_orientation).get()[0]
                oy = zed_pose.get_orientation(py_orientation).get()[1]
                oz = zed_pose.get_orientation(py_orientation).get()[2]
                ow = zed_pose.get_orientation(py_orientation).get()[3]
                # print("Orientation: ox: {0}, oy:  {1}, oz: {2}, ow: {3}\n".format(ox, oy, oz, ow))

                if debug:
                    fig.canvas.restore_region(bg)

                    xarr = np.concatenate((xarr[1:], [tx]))
                    yarr = np.concatenate((yarr[1:], [ty]))
                    zarr = np.concatenate((zarr[1:], [tz]))

                    xline.set_ydata(xarr)
                    yline.set_ydata(yarr)
                    zline.set_ydata(zarr)

                    ax.draw_artist(xline)
                    ax.draw_artist(yline)
                    ax.draw_artist(zline)

                    fig.canvas.blit(fig.bbox)
                    fig.canvas.flush_events()

                cam_poses.append([str(i).zfill(5), tx, ty, tz, ox, oy, oz, ow])

                if not (i%100): 
                    print(f"Captured {i} frames.")
                    print("Press [Enter] to stop collecting.\n")

        # Close the camera
        zed.close()

        # load intrin and extrin
        print('Load intrinsics and extrinsics')
        
        # Created SyncedPoses.txt - Basically reformatted translation and quarternion file

        # Load the intrinsic dictionary - containing the intrinsic K matrix
        #
        #                   |fx s  cx|
        # K =               |0  fy cy|
        #                   |0  0  1 |
        #
        # fx, fy = focal lengths (fy = a*fx)   a = aspect ratio   s = skew factor (usually 0)  cx, cy = offsets
        
        # Intrinsic transformation matrix
        cam_intrinsic_dict = dict()
        for arr in cam_intrinsics:
            cam_dict = dict()
            cam_dict['K'] = np.array([
                [arr[1], 0, arr[3]],
                [0, arr[2], arr[4]],
                [0, 0, 1]
            ], dtype=float)
            cam_intrinsic_dict[arr[0]] = cam_dict

        # Downscaling
        for k, v in tqdm(cam_intrinsic_dict.items(), desc='Processing camera intrinsics...'):
            cam_intrinsic_dict[k]['K'][0, :] /= (ori_size[0] / size[0])
            cam_intrinsic_dict[k]['K'][1, :] /= (ori_size[1] / size[1])

        # Extrinsic transformation matrix
        cam_pose_dict = dict()
        for arr in cam_poses: 
            line_data = np.array(arr, dtype=float)
            fid = arr[0]
            trans = line_data[1:4]
            quat = line_data[4:]
            rot_mat = quat2mat(np.append(quat[-1], quat[:3]).tolist())
            rot_mat = rot_mat.dot(np.array([
                [1, 0, 0],
                [0, -1, 0],
                [0, 0, -1]
            ]))
            rot_mat = rotx(np.pi / 2) @ rot_mat
            trans = rotx(np.pi / 2) @ trans
            trans_mat = np.zeros([3, 4])
            trans_mat[:3, :3] = rot_mat
            trans_mat[:3, 3] = trans
            trans_mat = np.vstack((trans_mat, [0, 0, 0, 1]))
            cam_pose_dict[fid] = trans_mat
        
        # save_intrinsics_extrinsics
        if not os.path.exists(os.path.join(cfg.TEST.PATH, 'poses')):
            os.mkdir(os.path.join(cfg.TEST.PATH, 'poses'))
        for k, v in tqdm(cam_pose_dict.items(), desc='Saving camera extrinsics...'):
            np.savetxt(os.path.join(cfg.TEST.PATH, 'poses', '{}.txt'.format(k)), v, delimiter=' ')

        if not os.path.exists(os.path.join(cfg.TEST.PATH, 'intrinsics')):
            os.mkdir(os.path.join(cfg.TEST.PATH, 'intrinsics'))
        for k, v in tqdm(cam_intrinsic_dict.items(), desc='Saving camera intrinsics...'):
            np.savetxt(os.path.join(cfg.TEST.PATH, 'intrinsics', '{}.txt'.format(k)), v['K'], delimiter=' ')

        # generate fragment
        fragments = []

        all_ids = []
        ids = []
        count = 0
        last_pose = None

        # Keyframe selection (By angle / translation threshold)
        for id in tqdm(cam_intrinsic_dict.keys(), desc='Keyframes selection...'):
            cam_intrinsic = cam_intrinsic_dict[id]
            cam_pose = cam_pose_dict[id]

            if count == 0:
                ids.append(id)
                last_pose = cam_pose
                count += 1
            else:
                angle = np.arccos(
                    ((np.linalg.inv(cam_pose[:3, :3]) @ last_pose[:3, :3] @ np.array([0, 0, 1]).T) * np.array(
                        [0, 0, 1])).sum())
                dis = np.linalg.norm(cam_pose[:3, 3] - last_pose[:3, 3])
                if angle > (min_angle / 180) * np.pi or dis > min_distance:
                    ids.append(id)
                    last_pose = cam_pose
                    # Grouping into fragments of 9 keyframes
                    count += 1
                    if count == window_size:
                        all_ids.append(ids)
                        ids = []
                        count = 0

        # save fragments
        for i, ids in enumerate(tqdm(all_ids, desc='Saving fragments file...')):
            poses = []
            intrinsics = []
            for id in ids:
                # Moving down the X-Y plane in the ARKit coordinate to meet the training settings in ScanNet.
                cam_pose_dict[id][2, 3] += 1.5
                poses.append(cam_pose_dict[id])
                intrinsics.append(cam_intrinsic_dict[id]['K'])
            fragments.append({
                'scene': cfg.TEST.PATH.split('/')[-1],
                'fragment_id': i,
                'image_ids': ids,
                'extrinsics': poses,
                'intrinsics': intrinsics
            })

        with open(os.path.join(cfg.TEST.PATH, 'fragments.pkl'), 'wb') as f:
            pickle.dump(fragments, f)

        with open(os.path.join(cfg.TEST.PATH, 'fragments.txt'), 'w') as f:
            pprint(fragments, f)

        print("DataCollector exiting...")

    def kill(self): 
        self._kill.set()

    def killed(self): 
        return self._kill.isSet() 

def main():

    parser = argparse.ArgumentParser(description='NeuralRecon Real-time Demo')
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)
    # parse arguments and check
    args = parser.parse_args()
    update_config(cfg, args)

    while True:
        prompt = input("Would you like to collect a new set of data? [y/n]: ")
        if prompt == "y":
            dc = DataCollector()
            dc.start()
            input("Press [Enter] to stop training.")
            dc.kill()
            dc.join()
            break
        elif prompt == "n":
            break

    logger.info("Running NeuralRecon...")
    transform = [transforms.ResizeImage((640, 480)),
                transforms.ToTensor(),
                transforms.RandomTransformSpace(
                    cfg.MODEL.N_VOX, cfg.MODEL.VOXEL_SIZE, random_rotation=False, random_translation=False,
                    paddingXY=0, paddingZ=0, max_epoch=cfg.TRAIN.EPOCHS),
                transforms.IntrinsicsPoseToProjection(cfg.TEST.N_VIEWS, 4)]
    transformz = transforms.Compose(transform)
    ARKitDataset = find_dataset_def(cfg.DATASET)
    test_dataset = ARKitDataset(cfg.TEST.PATH, "test", transformz, cfg.TEST.N_VIEWS, len(cfg.MODEL.THRESHOLDS) - 1)
    data_loader = DataLoader(test_dataset, cfg.BATCH_SIZE, shuffle=False, num_workers=cfg.TEST.N_WORKERS, drop_last=False)

    # model
    logger.info("Initializing the model on GPU...")
    model = NeuralRecon(cfg).cuda().eval()
    model = torch.nn.DataParallel(model, device_ids=[0])

    # use the latest checkpoint file
    saved_models = [fn for fn in os.listdir(cfg.LOGDIR) if fn.endswith(".ckpt")]
    saved_models = sorted(saved_models, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    loadckpt = os.path.join(cfg.LOGDIR, saved_models[-1])
    logger.info("Resuming from " + str(loadckpt))
    state_dict = torch.load(loadckpt)
    model.load_state_dict(state_dict['model'], strict=False)
    epoch_idx = state_dict['epoch']
    save_mesh_scene = SaveScene(cfg)

    logger.info("Start inference..")
    duration = 0.
    gpu_mem_usage = []
    frag_len = len(data_loader)
    with torch.no_grad():
        for frag_idx, sample in enumerate(tqdm(data_loader)):
            # save mesh if: 1. SAVE_SCENE_MESH and is the last fragment, or
            #               2. SAVE_INCREMENTAL, or
            #               3. VIS_INCREMENTAL
            save_scene = (cfg.SAVE_SCENE_MESH and frag_idx == frag_len - 1) or cfg.SAVE_INCREMENTAL or cfg.VIS_INCREMENTAL

            start_time = time.time()
            outputs, loss_dict = model(sample, save_scene)
            duration += time.time() - start_time

            if cfg.REDUCE_GPU_MEM:
                # will slow down the inference
                torch.cuda.empty_cache()

            # vis or save incremental result.
            scene = sample['scene'][0]
            save_mesh_scene.keyframe_id = frag_idx
            save_mesh_scene.scene_name = scene.replace('/', '-')

            if cfg.SAVE_INCREMENTAL:
                save_mesh_scene.save_incremental(epoch_idx, 0, sample['imgs'][0], outputs)

            if cfg.VIS_INCREMENTAL:
                save_mesh_scene.vis_incremental(epoch_idx, 0, sample['imgs'][0], outputs)

            if cfg.SAVE_SCENE_MESH and frag_idx == frag_len - 1:
                assert 'scene_tsdf' in outputs, \
                """Reconstruction failed. Potential reasons could be:
                    1. Wrong camera poses.
                    2. Extremely difficult scene.
                    If you can run with the demo data without any problem, please submit a issue with the failed data attatched, thanks!
                """
                save_mesh_scene.save_scene_eval(epoch_idx, outputs)
            
            gpu_mem_usage.append(torch.cuda.memory_reserved())
            
    summary_text = f"""
    Summary:
        Total number of fragments: {frag_len} 
        Average keyframes/sec: {1 / (duration / (frag_len * cfg.TEST.N_VIEWS))}
        Average GPU memory usage (GB): {sum(gpu_mem_usage) / len(gpu_mem_usage) / (1024 ** 3)} 
        Max GPU memory usage (GB): {max(gpu_mem_usage) / (1024 ** 3)} 
    """
    print(summary_text)
    input("Press [Enter] to exit...")

    if cfg.VIS_INCREMENTAL:
        save_mesh_scene.close()

if __name__ == "__main__":
    main()