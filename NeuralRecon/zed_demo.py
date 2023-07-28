import argparse
import os, time
import pickle
import cv2
import cProfile

import torch
from torch.utils.data import IterableDataset, DataLoader
from loguru import logger
from tqdm import tqdm
from PIL import Image

from models import NeuralRecon
from utils import SaveScene
from config import cfg, update_config
from datasets import transforms
from tools.kp_reproject import *
from tools.sync_poses import *
from transforms3d.quaternions import quat2mat

import pyzed.sl as sl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
import numpy as np
from queue import Queue

import time
import threading

class ZEDStreamDataset(IterableDataset):
    def __init__(self, queue, mode, transforms, nviews, n_scales):
        super(ZEDStreamDataset, self).__init__()
        self.queue = queue
        self.mode = mode
        self.n_views = nviews
        self.transforms = transforms

        assert self.mode in ["train", "val", "test"]

        self.n_scales = n_scales
        self.epoch = None
        self.tsdf_cashe = {}
        self.max_cashe = 100

    def __iter__(self):
        return self
    
    def __next__(self):
        meta = self.queue.get()

        imgs = meta['images']
        intrinsics_list = meta['intrinsics']
        extrinsics_list = meta['extrinsics']
        intrinsics = np.stack(intrinsics_list)
        extrinsics = np.stack(extrinsics_list)
        
        items = {
            'imgs': imgs,
            'intrinsics': intrinsics,
            'extrinsics': extrinsics,
            'scene': meta['scene'],
            'fragment': meta['scene'] + '_' + str(meta['fragment_id']),
            'epoch': [self.epoch],
            'vol_origin': np.array([0, 0, 0])
        }

        if self.transforms is not None:
            items = self.transforms(items)

        return items

class DataCollectorThread(threading.Thread):
    def __init__(self, queue):
        threading.Thread.__init__(self)
        self.daemon = True
        self._kill = threading.Event()
        self.queue = queue

    def run(self):
        
        # Configs
        window_size=9
        min_angle=15
        min_distance=0.1
        ori_size=(1920, 1080)
        size=(640, 480)

        logger.info("DataCollector starting...")

        # Create a Camera object
        zed = sl.Camera()

        # Create a InitParameters object and set configuration parameters
        init_params = sl.InitParameters()
        init_params.camera_resolution = sl.RESOLUTION.HD1080  # Use HD1080 video mode
        init_params.camera_fps = 30  # Set fps at 30
        init_params.coordinate_units = sl.UNIT.METER
        init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP

        # Open the camera
        err = zed.open(init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            logger.error("Camera failed to open.")
            exit(1)
        zed.set_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE, 25) # Turn exposure as far down as possible
                                                                # For image clarity,
                                                                # Taking care not to black everything out

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
        image = sl.Mat()
        runtime_parameters = sl.RuntimeParameters()

        # fragments
        frag = {
            "ids":          [],
            "intrinsics":   [],
            "poses":        [],
            "images":       []
        }
        count = 0
        last_pose = None
        frags = 0
        i = 0

        while(not self.killed()):
            if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
                i += 1
                # A new image is available if grab() returns SUCCESS
                zed.retrieve_image(image, sl.VIEW.LEFT)
                img = Image.fromarray(cv2.resize(image.get_data(), size)).convert("RGB")
                # cv2.imwrite(os.path.join(cfg.TEST.PATH, "images" , str(i).zfill(5) + '.jpg'), img)
                
                properties = zed.get_camera_information().camera_configuration.calibration_parameters.left_cam
                # print(f"fx(px): {properties.fx}")
                # print(f"fy(px): {properties.fy}")
                # print(f"cx(px): {properties.cx}")
                # print(f"cy(px): {properties.cy}")

                # Get the pose of the camera relative to the world frame
                state = zed.get_position(zed_pose, sl.REFERENCE_FRAME.WORLD)
                # Display translation and timestamp
                py_translation = sl.Translation()
                tx = zed_pose.get_translation(py_translation).get()[0]
                ty = zed_pose.get_translation(py_translation).get()[1]
                tz = zed_pose.get_translation(py_translation).get()[2]
                
                # Display orientation quaternion
                py_orientation = sl.Orientation()
                ox = zed_pose.get_orientation(py_orientation).get()[0]
                oy = zed_pose.get_orientation(py_orientation).get()[1]
                oz = zed_pose.get_orientation(py_orientation).get()[2]
                ow = zed_pose.get_orientation(py_orientation).get()[3]
                
                # if not (i%100): print(f"Captured {i} frames.")
                
                cam_intrinsic = [str(i).zfill(5), properties.fx, properties.fy, properties.cx, properties.cy]
                cam_pose = [str(i).zfill(5), tx, ty, tz, ox, oy, oz, ow]

                # Load the intrinsic dictionary - containing the intrinsic K matrix
                #
                #                   |fx s  cx|
                # K =               |0  fy cy|
                #                   |0  0  1 |
                #
                # fx, fy = focal lengths (fy = a*fx)   a = aspect ratio   s = skew factor (usually 0)  cx, cy = offsets
                
                # Intrinsic transformation matrix
                cam_intrinsic = np.array([
                    [cam_intrinsic[1], 0, cam_intrinsic[3]],
                    [0, cam_intrinsic[2], cam_intrinsic[4]],
                    [0, 0, 1]
                ], dtype=float)
                
                # Downscaling
                cam_intrinsic[0, :] /= (ori_size[0] / size[0])
                cam_intrinsic[1, :] /= (ori_size[1] / size[1])

                # Extrinsic transformation matrix
                line_data = np.array(cam_pose, dtype=float)
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

                # Moving down the X-Y plane in the ARKit coordinate to meet the training settings in ScanNet.
                trans_mat[2, 3] += 1.5
                cam_pose = trans_mat

                # Keyframe selection (By angle / translation threshold)
                if count == 0:
                    frag["ids"].append(id)
                    frag["intrinsics"].append(cam_intrinsic)
                    frag["poses"].append(cam_pose)
                    frag["images"].append(img)
                    last_pose = cam_pose
                    count += 1
                else:
                    angle = np.arccos(
                        ((np.linalg.inv(cam_pose[:3, :3]) @ last_pose[:3, :3] @ np.array([0, 0, 1]).T) * np.array(
                            [0, 0, 1])).sum())
                    dis = np.linalg.norm(cam_pose[:3, 3] - last_pose[:3, 3])
                    if angle > (min_angle / 180) * np.pi or dis > min_distance:
                        frag["ids"].append(id)
                        frag["intrinsics"].append(cam_intrinsic)
                        frag["poses"].append(cam_pose)
                        frag["images"].append(img)
                        last_pose = cam_pose
                        # Grouping into fragments of 9 keyframes
                        count += 1
                        if count == window_size:
                            f = {
                                'scene': cfg.TEST.PATH.split('/')[-1],
                                'fragment_id': i,
                                'image_ids': [id for id in frag["ids"]],
                                'extrinsics': [pose for pose in frag["poses"]],
                                'intrinsics': [intrinsic for intrinsic in frag["intrinsics"]],
                                'images': [img for img in frag["images"]]
                            }
                            
                            # New fragment to queue
                            self.queue.put(f)
                            
                            frag["ids"].clear()
                            frag["poses"].clear()
                            frag["intrinsics"].clear()
                            frag["images"].clear()
                            count = 0

                            print(f"{frags} fragments.")
                            frags += 1

        # Close the camera
        zed.close()
        logger.info("DataCollector stopping...")

    def kill(self): 
        self._kill.set()

    def killed(self): 
        return self._kill.isSet()

class NeuralReconThread(threading.Thread):
    def __init__(self, queue):
        threading.Thread.__init__(self)
        self.daemon = True
        self._kill = threading.Event()
        self.queue = queue

    def run(self):
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

        logger.info("NeuralRecon starting...")
        transform = [transforms.ResizeImage((640, 480)),
                    transforms.ToTensor(),
                    transforms.RandomTransformSpace(
                        cfg.MODEL.N_VOX, cfg.MODEL.VOXEL_SIZE, random_rotation=False, random_translation=False,
                        paddingXY=0, paddingZ=0, max_epoch=cfg.TRAIN.EPOCHS),
                    transforms.IntrinsicsPoseToProjection(cfg.TEST.N_VIEWS, 4)]
        transformz = transforms.Compose(transform)
        
        test_dataset = ZEDStreamDataset(self.queue, "test", transformz, cfg.TEST.N_VIEWS, len(cfg.MODEL.THRESHOLDS) - 1)
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
        outputs = None

        logger.info("Start inference..")
        duration = 0.
        gpu_mem_usage = []
        data_iter = iter(data_loader)
        #frag_len = len(data_loader)
        frag_idx = 0

        with torch.no_grad():
            while (not self.killed()):
                sample = next(data_iter)
                frag_idx += 1
                print(f"Fragment {frag_idx}")

                # save mesh if: 1. SAVE_SCENE_MESH and is the last fragment, or
                #               2. SAVE_INCREMENTAL, or
                #               3. VIS_INCREMENTAL
                #save_scene = (cfg.SAVE_SCENE_MESH and frag_idx == frag_len - 1) or cfg.SAVE_INCREMENTAL or cfg.VIS_INCREMENTAL
                save_scene = cfg.SAVE_INCREMENTAL or cfg.VIS_INCREMENTAL

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

                gpu_mem_usage.append(torch.cuda.memory_reserved())
        
        summary_text = f"""
        Summary:
            Total number of fragments: {frag_idx} 
            Average keyframes/sec: {1 / (duration / (frag_idx * cfg.TEST.N_VIEWS))}
            Average GPU memory usage (GB): {sum(gpu_mem_usage) / len(gpu_mem_usage) / (1024 ** 3)} 
            Max GPU memory usage (GB): {max(gpu_mem_usage) / (1024 ** 3)} 
        """
        print(summary_text)
        print(gpu_mem_usage)

        if cfg.SAVE_SCENE_MESH:
            assert 'scene_tsdf' in outputs, \
            "Reconstruction failed. Potential reasons could be:\n\
                1. Wrong camera poses.\n\
                2. Extremely difficult scene.\n\
                If you can run with the demo data without any problem, please submit a issue with the failed data attatched, thanks!"
            save_mesh_scene.save_scene_eval(epoch_idx, outputs)

        if cfg.VIS_INCREMENTAL:
            save_mesh_scene.close()
        
        logger.info("NeuralRecon stopping...")

    def kill(self): 
        self._kill.set()

    def killed(self): 
        return self._kill.isSet() 

def main():
    q = Queue(1)
    dc = DataCollectorThread(q)
    nr = NeuralReconThread(q)
    dc.start()
    nr.start()

    input()

    dc.kill()
    nr.kill()
    dc.join()
    nr.join()   

# python zed_demo.py --cfg ./config/mine.yaml
if __name__ == "__main__":
    main()