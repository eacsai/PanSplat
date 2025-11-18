import random

from dataclasses import dataclass
from pathlib import Path
from typing import Literal
import os

import torch
import torchvision.transforms as tf
from einops import repeat
from jaxtyping import Float
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
import numpy as np

from .dataset import DatasetCfgCommon
from .types import Stage
from .view_sampler import ViewSampler

from src.model.encoder.unifuse.datasets.util import Equirec2Cube
from einops import rearrange
import torch.nn.functional as F
from torchvision import transforms

sat_range = 200

GrdImg_H = 160  # 256
GrdImg_W = 320  # 1024

pano_width = 320
pano_height = 160

depth_scale = 10.0  # from meter to decameter

# 坐标系转换：从(x东, y北, z上)转换为(x南, y下, z东)
# x_new = -y_old, y_new = -z_old, z_new = x_old
OpenCV_Transform = torch.tensor([
    [0, -1, 0, 0],  # x_new = -y_old
    [0, 0, -1, 0],  # y_new = -z_old
    [1, 0, 0, 0],   # z_new = x_old
    [0, 0, 0, 1]
], dtype=torch.float32)

SatMap_end_sidelength = 256


@dataclass
class DatasetVIGORCfg(DatasetCfgCommon):
    name: Literal["vigor"]
    roots: list[Path]
    max_fov: float
    make_baseline_1: bool
    augment: bool
    test_len: int
    test_chunk_interval: int
    test_datasets: list[dict]
    skip_bad_shape: bool = True
    near: float = -1.0
    far: float = -1.0
    baseline_scale_bounds: bool = True
    shuffle_val: bool = False


class DatasetVIGOR(Dataset):
    cfg: DatasetVIGORCfg
    stage: Stage
    view_sampler: ViewSampler

    to_tensor: tf.ToTensor
    chunks: list[Path]
    near: float = 0.1
    far: float = 1000.0

    def __init__(
        self,
        cfg: DatasetVIGORCfg,
        stage: Stage,
        view_sampler: ViewSampler,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.stage = stage
        self.view_sampler = view_sampler
        self.to_tensor = tf.ToTensor()
        # NOTE: update near & far; remember to DISABLE `apply_bounds_shim` in encoder
        if cfg.near != -1:
            self.near = cfg.near
        if cfg.far != -1:
            self.far = cfg.far

        # scan folders in cfg.roots[0]
        if stage == "predict":
            stage = "test"

        height = cfg.image_shape[0]
        height = max(height, 512)
        resolution = (height * 2, height)
        resolution = 'x'.join(map(str, resolution))
        
        # load data list
        self.root = '/data/qiwei/nips25/'
        self.T_in = 1
        self.T_out = 3
        self.is_train = True if stage == "train" else False
        
        self.city_list = ["Kansas"]
        # self.city_list = ["chicago"]
        # self.city_list = ["newyork","Orlando","Phoenix","SanFrancisco","seattle"]
        self.grd_in_sat_path = "GrdInSat_dist_dir_month_new"
        self.grd_dir = "Ground"
        self.sat_dir = "Satellite"
        self.depth_dir = 'depth_metric'

        self.SatMap_length = SatMap_end_sidelength
        self.satmap_transform = transforms.Compose([
            transforms.Resize(size=[self.SatMap_length, self.SatMap_length]),#sat_d*sat_d
            transforms.ToTensor(),
        ])

        Grd_h = GrdImg_H
        Grd_w = GrdImg_W

        self.grdimage_transform = transforms.Compose([
            transforms.Resize(size=[Grd_h, Grd_w]),#grd_H*grd_W
            transforms.ToTensor(),
        ])

        # #读取规定的
        # with open('dataloader/selected_test_files_2.txt', 'r') as f:
        #     # 去除每行的换行符和首尾空格
        #     self.data = [line.strip() for line in f if line.strip()]

        self.width = pano_width
        self.height = pano_height

        # #原始代码
        self.data = []
        for city in self.city_list:
            if self.is_train:
                data_list_path = os.path.join(self.root, city, 'train_list.txt')
            else:
                data_list_path = os.path.join(self.root, city, 'test_list_om.txt')
            with open(data_list_path, 'r') as f:
                file_name = f.readlines()
            for file in file_name:
                self.data.append(city + '/' + file[:-1])


        self.e2c_mono = Equirec2Cube(512, 1024, 256)


    def euler_to_rotation_matrix(self, roll, pitch, yaw):
        roll_rad = np.deg2rad(roll)
        pitch_rad = np.deg2rad(pitch)
        yaw_rad = np.deg2rad(yaw)
        R_roll = np.array([
            [1, 0, 0],
            [0, np.cos(roll_rad), -np.sin(roll_rad)],
            [0, np.sin(roll_rad), np.cos(roll_rad)]
        ])
        R_pitch = np.array([
            [np.cos(pitch_rad), 0, np.sin(pitch_rad)],
            [0, 1, 0],
            [-np.sin(pitch_rad), 0, np.cos(pitch_rad)]
        ])
        R_yaw = np.array([
            [np.cos(yaw_rad), -np.sin(yaw_rad), 0],
            [np.sin(yaw_rad), np.cos(yaw_rad), 0],
            [0, 0, 1]
        ])
        return R_yaw @ R_pitch @ R_roll

    def create_transformation_matrix(self, x, y, z, R):
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = [x, y, z]
        return T

    def get_point(self, all_id, sample_num):
        id_list = []
        if self.is_train:
            now_id = random.randint(0, max(all_id - sample_num, 0))
        else:
            now_id = 0
        for i in range(sample_num):
            id_list.append(now_id)
            now_id = min(now_id + 1, all_id-1)
        return id_list


    def __getitem__(self, idx):
        # Load the images.
        city, road, file_name = self.data[idx].split('/')

        # Create data dictionary
        data = {}
        scene = f"{city}/{road}"
        data['scene_id'] = scene
        input_ims = []
        target_Ts = []  

        context_indices = torch.tensor([0, 2])
        target_indices = torch.tensor([0, 1, 2])
        if not file_name.endswith('.png'):
            file_name = file_name + 'g'
        SatMap_name = os.path.join(self.root, city, road, self.sat_dir, file_name)

        with Image.open(SatMap_name, 'r') as SatMap:
            sat_map = SatMap.convert('RGB')
            orin_sat_S = sat_map.size[0]
            sat_map_tensor = self.satmap_transform(sat_map)
        input_ims.append(sat_map_tensor)
        orin_meter_per_pixel = sat_range/orin_sat_S
         # =================== read correspond grd ============================
        GrdInSat_file_path = os.path.join(self.root, city, road, self.grd_in_sat_path, file_name.replace('.png', '.txt'))
        with open(GrdInSat_file_path, 'r') as GrdInSat_f:
            sat2grd_name = GrdInSat_f.readlines()

        targets_indexes = np.array(self.get_point(len(sat2grd_name), self.T_out))

        draw_camera_pose = []
        grd_img_name = []
        depths = []
        confs = []

        for i in targets_indexes:
            #target_ims
            # print(sat2grd_name[i])
            grd_name, u, v, yaw, alt, o_alt = sat2grd_name[i].split(' ')
            u, v, yaw, alt, o_alt = float(u), float(v), float(yaw), float(alt), float(o_alt)
            draw_camera_pose.append([u, v, yaw, alt])
            grd_name = grd_name.split('/')[-1]

            left_img_name = os.path.join(self.root, city, road, self.grd_dir, grd_name)
            delta_E = (float(u) - orin_sat_S/2)*orin_meter_per_pixel
            delta_E = delta_E / depth_scale  # scale to decameter
            delta_N = -(float(v) - orin_sat_S/2)*orin_meter_per_pixel
            delta_N = delta_N / depth_scale  # scale to decameter
            delta_alt = alt - o_alt
            delta_alt = delta_alt / depth_scale  # scale to decameter
            
            # xinaghui code
            target_Ts_R = self.euler_to_rotation_matrix(0, yaw, 0)
            target_Ts_RT = self.create_transformation_matrix(-delta_N, -delta_alt, delta_E, target_Ts_R) # x是南，y是下 z是东

            target_Ts.append(target_Ts_RT)
            grd_img_name.append(left_img_name)

            depth_name = left_img_name.replace(self.grd_dir, self.depth_dir).replace('.png', '_depth.npy')
            depths.append(depth_name)

            conf_name = depth_name.replace('_depth.npy', '_conf.npy')  # in meter
            confs.append(conf_name)

        context_images = self.convert_images([grd_img_name[i] for i in context_indices])
        target_images = self.convert_images([grd_img_name[i] for i in target_indices])


        context_m_depths = [depths[i] for i in context_indices]
        target_m_depths = [depths[i] for i in target_indices]
        context_m_depths = self.convert_depths(context_m_depths) / depth_scale  # to decimeter
        target_m_depths = self.convert_depths(target_m_depths) / depth_scale  # to decimeter

        context_m_confs = [confs[i] for i in context_indices]
        target_m_confs = [confs[i] for i in target_indices]
        context_m_confs = self.convert_depths(context_m_confs)
        target_m_confs = self.convert_depths(target_m_confs)


        # load camera
        extrinsics = torch.stack([torch.from_numpy(T).float() for T in target_Ts]) 
        ref_cam = extrinsics[1:2]
        # 将所有相机转换到中间相机的坐标系下
        ref_cam_inv = torch.inverse(ref_cam)  # [1,4,4]
        extrinsics = torch.einsum("bij,bjk->bik", ref_cam_inv, extrinsics)  # [B,4,4]

        # Resize the world to make the baseline 1.
        context_extrinsics = extrinsics[context_indices]
        if context_extrinsics.shape[0] == 2 and self.cfg.make_baseline_1:
            a, b = context_extrinsics[:, :3, 3]
            scale = (a - b).norm()
            extrinsics[:, :3, 3] /= scale
        else:
            scale = 1

        intrinsics = torch.eye(3, dtype=torch.float32)
        fx, fy, cx, cy = 0.25, 0.5, 0.5, 0.5
        intrinsics[0, 0] = fx
        intrinsics[1, 1] = fy
        intrinsics[0, 2] = cx
        intrinsics[1, 2] = cy
        intrinsics = repeat(intrinsics, "h w -> b h w", b=len(extrinsics)).clone()

        # resize images for mono depth
        mono_images = F.interpolate(context_images, size=(256, 512), mode='bilinear')
        mono_images = F.interpolate(mono_images, size=(512, 1024), mode='bilinear')

        # Project the images to the cube.
        cube_image = []
        for img in mono_images:
            img = img.numpy()
            img = rearrange(img, "c h w -> h w c")
            img = self.e2c_mono.run(img)
            cube_image.append(img)
        cube_image = np.stack(cube_image)
        cube_image = rearrange(cube_image, "v h w c -> v c h w")

        nf_scale = scale if self.cfg.baseline_scale_bounds else 1.0
        data.update({
            "context": {
                "extrinsics": extrinsics[context_indices],
                "intrinsics": intrinsics[context_indices],
                "image": context_images,
                "mono_image": mono_images,
                "cube_image": cube_image,
                "near": self.get_bound("near", len(context_images)) / nf_scale,
                "far": self.get_bound("far", len(context_images)) / nf_scale,
                "index": context_indices,
                "depth": context_m_depths,
                # "mask": context_mask,
            },
            "target": {
                "extrinsics": extrinsics[target_indices],
                "intrinsics": intrinsics[target_indices],
                "image": target_images,
                "near": self.get_bound("near", len(target_indices)) / nf_scale,
                "far": self.get_bound("far", len(target_indices)) / nf_scale,
                "index": target_indices,
                "depth": target_m_depths,
                # "mask": target_mask,
            },
            "scene": scene,
        })
    
        return data

    def convert_poses(
        self,
        trans: Float[Tensor, "batch 3"],
        rots: Float[Tensor, "batch 3 3"],
    ) -> Float[Tensor, "batch 4 4"]:  # extrinsics
        b, _ = trans.shape

        # Convert the extrinsics to a 4x4 OpenCV-style W2C matrix.
        c2w = repeat(torch.eye(4, dtype=torch.float32), "h w -> b h w", b=b).clone()
        c2w[:, :3, :3] = rots
        c2w[:, :3, 3] = trans
        w2w = torch.tensor([  # X -> X, -Z -> Y, upY -> Z
            [1, 0, 0, 0],
            [0, 0, -1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
        ]).float()
        c2c = torch.tensor([  # rightx -> rightx, upy -> -downy, backz -> -forwardz
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1],
        ]).float()
        c2w = w2w @ c2w @ c2c
        return c2w

    def convert_images(
        self,
        images: list[str],
    ):
        torch_images = []
        for image in images:
            image = Image.open(image)
            image = image.convert('RGB')  # Ensure RGB format (3 channels)
            image = image.resize(self.cfg.image_shape[::-1], Image.LANCZOS)
            torch_images.append(self.to_tensor(image))
        return torch.stack(torch_images)

    def convert_depths(
        self,
        depths,
    ):
        torch_depths = []
        for depth in depths:
            depth = np.load(depth)
            depth = torch.tensor(depth, dtype=torch.float32)
            torch_depths.append(depth)
        return F.interpolate(torch.stack(torch_depths),
                             size=self.cfg.image_shape[::1], 
                             mode='bilinear', 
                             align_corners=False
                             )

    def get_bound(
        self,
        bound: Literal["near", "far"],
        num_views: int,
    ) -> Float[Tensor, " view"]:
        value = torch.tensor(getattr(self, bound), dtype=torch.float32)
        return repeat(value, "-> v", v=num_views)

    def __len__(self):
        return len(self.data)
