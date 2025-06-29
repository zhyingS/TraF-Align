import matplotlib.pyplot as plt
import numpy as np
import torch
import copy
from einops import rearrange

# from DEFU_paper.qualitative_visualization import vis_utils


def check_detection_boxes(lidar, box_list, cfg=None, detc=""):

    features = lidar["voxel_features"].cpu().numpy()
    coords = lidar["voxel_coords"].cpu().numpy()
    mask = coords[:, 0] <= 10  # only draw the first batch
    features = features[mask]
    features = features[:, :, :3]
    coords = coords[mask]
    ego_frame = cfg["dataset"]["frame_his"]

    ego_mask = coords[:, 0] < ego_frame
    cav_mask = coords[:, 0] >= ego_frame

    check_lidar_alignment(
        [features[cav_mask].reshape(-1, 3), features[ego_mask].reshape(-1, 3)],
        box_list,
        cfg,
        detc=detc,
    )


def check_lidar_alignment(lidars, box_list, cfg=None, detc="", ids=None):
    fig, ax = plt.subplots(2, 1)
    a = int(len(lidars) / 2)
    for lidar in lidars[:a]:
        ax[0].scatter(lidar[:, 0], lidar[:, 1], c="b", s=0.005)
        ax[1].scatter(lidar[:, 0], lidar[:, 2], c="b", s=0.005)

    for lidar in lidars[a:]:
        ax[0].scatter(lidar[:, 0], lidar[:, 1], s=0.005, c="k", alpha=0.5)
        ax[1].scatter(lidar[:, 0], lidar[:, 2], s=0.005, c="k", alpha=0.5)

    c_list = ["g", "r", "b", "k", "y"]
    if not isinstance(box_list, list):
        box_list = [box_list]
    if len(box_list) > 0:
        for i, boxes in enumerate(box_list):
            if isinstance(boxes, torch.Tensor):
                boxes = boxes.cpu().numpy()

            if boxes.shape[0] == 0:
                boxes = np.zeros((10, 7))

            if boxes.shape[1] == 7 or boxes.shape[1] == 8:  # for detection result boxes
                boxes = np.hstack((boxes[:, 0].reshape(-1, 1) * 0, boxes))

            R = np.zeros((boxes.shape[0], 2, 2))
            R[:, 0, 0] = np.cos(boxes[:, 7])
            R[:, 0, 1] = -np.sin(boxes[:, 7])
            R[:, 1, 0] = np.sin(boxes[:, 7])
            R[:, 1, 1] = np.cos(boxes[:, 7])

            boxlf = (
                np.hstack(
                    (boxes[:, 6].reshape(-1, 1), -boxes[:, 5].reshape(-1, 1))
                ).reshape(-1, 2)
                / 2
            )
            boxlr = (
                np.hstack(
                    (-boxes[:, 6].reshape(-1, 1), -boxes[:, 5].reshape(-1, 1))
                ).reshape(-1, 2)
                / 2
            )
            boxrr = (
                np.hstack(
                    (-boxes[:, 6].reshape(-1, 1), boxes[:, 5].reshape(-1, 1))
                ).reshape(-1, 2)
                / 2
            )
            boxrf = (
                np.hstack(
                    (boxes[:, 6].reshape(-1, 1), boxes[:, 5].reshape(-1, 1))
                ).reshape(-1, 2)
                / 2
            )
            box2d = np.hstack((boxlf, boxlr, boxrr, boxrf, boxlf)).reshape(
                boxes.shape[0], -1, 2
            )
            box2d = np.matmul(R, box2d.transpose(0, 2, 1)).transpose(0, 2, 1)

            box2d = box2d + boxes[:, 1:3][:, None, :]

            ids_ = []
            for j, box in enumerate(box2d):
                ax[0].plot(box[:, 0], box[:, 1], c=c_list[i], linewidth=1)
                if i == 0 and ids is not None:
                    if int(ids[j]) not in ids_:
                        ax[0].text(
                            1 + np.mean(box[:, 0]),
                            1 + np.mean(box[:, 1]),
                            int(ids[j]),
                            c="k",
                            fontsize=5,
                        )
                        ids_.append(int(ids[j]))

    ax[0].set_aspect(1)
    ax[1].set_aspect(1)
    ax[0].set_xlabel("x/m")
    ax[0].set_ylabel("y/m")
    ax[1].set_xlabel("x/m")
    ax[1].set_ylabel("z/m")
    plt.tight_layout()
    plt.subplots_adjust(hspace=0)

    plt.savefig(f"visualization/lidar{detc}.png", dpi=600, bbox_inches="tight")
    plt.close()


def vis_gt_offset(x_offset, offset_mask, cfg, field):

    offset_mask = offset_mask[0][0]
    lidar_range = cfg["voxelization"]["lidar_range"]
    grid_size = cfg["voxelization"]["grid_size"]

    offset = x_offset[0]
    _, h, w = offset.shape
    offset = offset.reshape(2, -1, h, w)

    sm = 17
    offset_ = np.zeros((2, 18, h, w))
    offset_[0, :sm, :, :] = offset[0, :, :, :]
    offset_[1, :sm, :, :] = offset[1, :, :, :]

    offset_ = offset_.reshape(-1, 2, 3, 3, h, w)
    offset_ = np.transpose(offset_, (1, 0, 2, 3, 4, 5))
    # while True:
    command = "y"  # input("continue yes or no? ")
    if command == "n":
        pass
    else:
        plt.imshow(offset_mask)
        # plt.show(block=True)
        plt.savefig(
            "visualization/offset_mask.png",
            dpi=600,
            bbox_inches="tight",
        )
        plt.close()
        a = torch.where(offset_mask == 1)

        x_ = 39  # int(input("please input the h query: "))
        y_ = 66  # int(input("please input the w query: "))
        fig, ax = plt.subplots(1, 3, sharex=True, sharey=True)
        ax[0].imshow(offset_mask)
        i = 1
        for head in offset_:

            hm = copy.deepcopy(field[0][0][0].cpu().numpy())
            h_offset = (head[0, :, :, x_, y_] - lidar_range[1]) / 0.8
            w_offset = (head[1, :, :, x_, y_] - lidar_range[0]) / 0.8
            mask = (
                (h_offset >= 0)
                * (h_offset <= grid_size[0] / 4)
                * (w_offset >= 0)
                * (w_offset <= grid_size[1] / 4)
            )
            hm[h_offset[mask].astype(np.int), w_offset[mask].astype(np.int)] = -2
            ax[i].imshow(hm)
            ax[i].set_ylim(x_ - 10, x_ + 10)
            ax[i].set_xlim(y_ - 10, y_ + 10)
            ax[i].invert_yaxis()
            i += 1
        plt.savefig("visualization/offset.png", dpi=600, bbox_inches="tight")


def vis_offset_val(gt_offsets, gt_offset_masks, x_offsets, cfg, hm, masks):
    x_offsets = rearrange(x_offsets, "(b n) c h w -> b n c h w", b=gt_offsets.shape[0])
    masks = rearrange(masks, "(b n) c h w -> b n c h w", b=gt_offsets.shape[0])

    batch, cav = 0, 0

    mask = masks[batch][cav]
    x_offset = x_offsets[batch][cav]
    gt_offset_mask = gt_offset_masks[batch][cav]
    gt_offset = gt_offsets[batch][cav]
    hm = hm[batch][cav][0]
    vehicel_mask = mask[0]

    lidar_range = cfg["voxelization"]["lidar_range"]
    grid_size = cfg["voxelization"]["grid_size"]
    head_num = cfg["model"]["deform"]["offset"]["heads"]
    kernel = cfg["model"]["deform"]["offset"]["kernel"][0]
    anchor_h, anchor_w = np.meshgrid(np.arange(kernel), np.arange(kernel))
    anchor_h, anchor_w = anchor_h - (kernel - 1) / 2, anchor_w - (kernel - 1) / 2

    offset = x_offset.cpu().numpy()
    gt_offset = gt_offset.cpu().numpy()
    gt_offset_mask = gt_offset_mask.cpu().numpy()

    _, h, w = offset.shape
    offset_ = offset.reshape(head_num, 2, kernel, kernel, h, w)

    _, h, w = gt_offset.shape
    gt_offset = gt_offset.reshape(2, -1, h, w)

    sm = cfg["model"]["deform"]["offset"]["supervise_num"]
    gt_offset_ = gt_offset

    gt_offset_ = gt_offset_.reshape(-1, 2, kernel, kernel, h, w)
    gt_offset_ = np.transpose(gt_offset_, (1, 0, 2, 3, 4, 5))

    # while True:
    command = "y"  # input("continue yes or no? ")
    if command == "n":
        pass
    else:
        # plt.show(block=True)
        fig, ax = plt.subplots(2, 2)
        ax[0, 0].imshow(vehicel_mask.cpu().numpy())

        ax[0, 1].imshow(hm.cpu().numpy())
        for i in range(2):
            for j in range(2):
                ax[i, j].invert_yaxis()

        plt.savefig(
            "visualization/heatmap_result_test.png",
            dpi=600,
            bbox_inches="tight",
        )
        plt.close()
        a = torch.where(vehicel_mask == 1)

        hm_ = hm.cpu().numpy()
        c = np.where(gt_offset_mask.sum(0) >= 1)
        h_query = 45  # int(input("please input the h query: "))
        w_query = 30  # int(input("please input the w query: "))
        fig, ax = plt.subplots(1, 2)
        i = 0

        factor = cfg["model"]["backbone"]["out_size_factor"][0]
        hm = hm_.copy()
        gt_hm = hm_.copy()
        for head in offset_:
            h = head[0, :, :, h_query, w_query]
            w = head[1, :, :, h_query, w_query]
            h_offset_ = (h + anchor_h) + h_query
            w_offset_ = (w + anchor_w) + w_query
            h_offset_ = np.clip(h_offset_, 0, hm.shape[0])
            w_offset_ = np.clip(w_offset_, 0, hm.shape[1])
            hm[h_offset_.astype(np.int), w_offset_.astype(np.int)] = -1
            h_offset = gt_offset_[i][0, :, :, h_query, w_query]
            w_offset = gt_offset_[i][1, :, :, h_query, w_query]

            mask = (
                (h_offset >= 0)
                * (h_offset <= grid_size[0] / factor)
                * (w_offset >= 0)
                * (w_offset <= grid_size[1] / factor)
            )
            gt_hm[h_offset[mask].astype(np.int), w_offset[mask].astype(np.int)] = -1
            gt_hm[h_query, w_query] = 0

            hm[h_query, w_query] = 0
            i += 1

        ax[0].imshow(hm)
        ax[0].set_title("learned offset")
        ax[0].invert_yaxis()

        ax[1].imshow(gt_hm)
        ax[1].set_title("gt offset")
        ax[1].invert_yaxis()

        plt.savefig(
            f"visualization/offset_val.png",
            dpi=600,
            bbox_inches="tight",
        )
        plt.close()


def vis_field(gt_fields, fields, detc="", mask=None):
    batch = 0
    fields = fields.cpu()
    fields = rearrange(fields, "(b n) c h w -> b n c h w", b=gt_fields.shape[0])
    cmap = "viridis"
    for cav in range(gt_fields.shape[1]):
        field = fields[batch][cav].clone()
        gt_field = gt_fields[batch][cav].cpu()

        fig, ax = plt.subplots(2, 2, sharex=True, sharey=True)

        ax[0, 1].imshow(field[0], cmap=cmap)
        ax[0, 0].imshow(gt_field[0], cmap=cmap)
        ax[0, 0].set_title("gt field")
        ax[0, 1].set_title("learned field")
        ax[1, 0].imshow(field[1], cmap=cmap)
        ax[1, 1].imshow(field[2], cmap=cmap)
        ax[1, 1].invert_yaxis()

        plt.savefig(
            f"visualization/field{detc}_veh{cav}.png",
            dpi=600,
            bbox_inches="tight",
        )
        plt.close()
        pass


h_range, w_range = [36, 45], [36, 41]


def vis_attention(offset, selected_indices, atts):
    index = []
    cav = 1
    softmax = torch.nn.Softmax(dim=-1)
    h_, w_ = offset.shape[-2:]
    for h in range(h_range[0], h_range[1]):
        for w in range(w_range[0], w_range[1]):
            index.append(h * w_ + w)
    att_index = selected_indices[cav][index]
    i = 0
    cmap = plt.get_cmap("RdPu")
    for attention_weight in atts:
        for head in range(attention_weight.shape[-1]):
            fig, ax = plt.subplots(1, 1)
            weight = attention_weight[cav][:, :, head][index]
            matrix = torch.zeros((att_index.shape[0], att_index.shape[0])).type_as(
                weight
            )
            for j, index_ in enumerate(index):
                att_index[att_index == index_] = j
            att_index[att_index > j] = 0
            for k in range(att_index.shape[0]):
                matrix[k, att_index[k]] = weight[k]
            matrix = matrix[1:-1, 1:-1]
            # matrix = softmax(matrix)
            im = ax.imshow(matrix.cpu().numpy(), cmap=cmap)
            ax.axis("off")
            # fig.colorbar(im,ax=ax)
            plt.savefig(
                f"visualization/attention_cav_{cav}_layer{i}_head{head}.png",
                dpi=600,
                bbox_inches="tight",
            )
            plt.close()
        i += 1
    pass


def z_score_normalize(x):
    return (x - torch.mean(x)) / torch.std(x)


def vis_backbone_map_paper(maps, cfg):
    import matplotlib.pyplot as plt
    import torch.nn.functional as F

    # maps: N x 256 x h x w
    len_ = maps.shape[0]
    maps = maps.detach()
    delay = cfg["wild_setting"]["async_overhead"]
    cmap = plt.get_cmap("RdPu")

    for i in range(len_):
        fig, ax = plt.subplots(1, 1)
        map = maps[i].mean(0)
        map_ = map[h_range[0] : h_range[1], w_range[0] : w_range[1]]
        h = map_.shape[0]
        map_ = map_.T

        im = ax.imshow(map_.cpu().numpy(), cmap=cmap, vmin=0, vmax=4)
        ax.invert_xaxis()
        ax.invert_yaxis()
        ax.axis("off")
        fig.colorbar(im, ax=ax)
        plt.savefig(
            f"visualization/backbone_{i}_{delay}ms.png",
            dpi=600,
            bbox_inches="tight",
        )
        plt.close()


def vis_offset_on_pcd(lidar, pred_offsets, hypes, selected_indices, atts):
    work_dir = "DEFU_paper/qualitative_visualization/"
    batch = 0
    query_points = np.array(
        [
            [61.7, 30],
            # [61.7,28],
            # [61.7,26],
            # [61.7,24],
            [61.7, 22],
            # [61.7,20],
        ]
    )
    pred_offsets = pred_offsets[0]
    pred_offsets = rearrange(
        pred_offsets,
        "(b n) c h w -> b n c h w",
        b=hypes["train_params"]["val_batch_size"],
    )

    for cav in [0, 1]:
        pred_offset = pred_offsets[batch][cav]
        for layer in range(len(atts)):
            for head in range(atts[0].shape[-1]):
                fig, ax = plt.subplots(query_points.shape[0], 1)
                for i, query_point in enumerate(query_points):
                    vis_utils.check_detection_boxes(lidar, [], cfg=hypes, ax=ax[i])
                    att_points, query_hw = form_offset_matrix(
                        pred_offset, query_point, hypes
                    )  # 18 x 2
                    h_, w_ = pred_offset.shape[-2:]
                    weight = atts[layer][cav][query_hw[0] * w_ + query_hw[1]][:, head]
                    plt_att_point(query_point, att_points, ax[i], weight)

                    ax[i].set_xlim(-33, -17)
                    ax[i].set_ylim(57, 65.7)
                    ax[i].axis("off")
                delay = hypes["wild_setting"]["async_overhead"]
                plt.savefig(
                    f"{work_dir}{delay}ms_offset/lidar_detection_offset_cav{cav}_layer_{layer}_head_{head}.png",
                    dpi=600,
                    bbox_inches="tight",
                )
                plt.close()


def vis_backbone_map(maps, cfg):
    # maps: N x 256 x h x w
    len_ = maps.shape[0]
    maps = maps.detach().cpu().numpy()
    delay = cfg["wild_setting"]["async_overhead"]
    cmap = plt.get_cmap("viridis")

    for i in range(len_):
        fig, ax = plt.subplots(1, 1)
        map = maps[i].sum(0)
        # map = maps[i][0]
        # map[0,0] = 2000
        # map[0,1] = 200
        im = ax.imshow(map, cmap=cmap)  # [30:50,35:45]
        ax.invert_yaxis()
        # ax.set_xlim(20,40)
        # ax.set_ylim(15,25)
        fig.colorbar(im, ax=ax)
        plt.savefig(
            f"visualization/backbone_{i}_{delay}ms.png",
            dpi=600,
            bbox_inches="tight",
        )
        plt.close()


def form_offset_matrix(offset, query_point, hypes):
    # offset: 36x50x60
    offset = offset.cpu().numpy()
    _, h, w = offset.shape
    head_num = hypes["model"]["deform"]["heads"]
    offset_ = offset.reshape(head_num, 2, 3, 3, h, w)

    lidar_range = hypes["voxelization"]["lidar_range"]
    factor = hypes["model"]["backbone"]["out_size_factor"][0]
    voxel_size = hypes["voxelization"]["voxel_size"][0] * factor
    x_, y_ = query_point
    x_ = ((x_ - lidar_range[0]) / voxel_size).astype(np.int)
    y_ = ((y_ - lidar_range[1]) / voxel_size).astype(np.int)

    # retrieve pixel-level offset
    anchor_h = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
    anchor_w = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    atts = []
    for head in offset_:
        h = head[0, :, :, y_, x_]
        w = head[1, :, :, y_, x_]
        h_offset = (h + anchor_h) + y_
        w_offset = (w + anchor_w) + x_
        x_att = (w_offset + 0.5) * voxel_size + lidar_range[0]
        y_att = (h_offset + 0.5) * voxel_size + lidar_range[1]
        atts.append(np.stack((x_att, y_att)).reshape(2, -1).T)

    return np.concatenate(atts), [x_, y_]


def plt_att_point(query_point, att_points, ax=None, weight=None):
    c1 = np.array([0, 63, 191]) / 255
    c2 = np.array([255, 192, 0]) / 255

    ax.scatter(
        -att_points[:, 1],
        att_points[:, 0],
        marker="s",
        s=200,
        c=weight.cpu().numpy(),
        cmap="viridis_r",
    )
    ax.scatter(-query_point[1], query_point[0], marker="*", s=400, c=c1)
