import cv2
import colorsys
import matplotlib.image as mpimg
from PIL import Image
import argparse
import os
import sys
from os import mkdir
import numpy as np
import torch
from torch.backends import cudnn
from matplotlib import pyplot as plt

sys.path.append('.')
import torchvision
import torch.utils.data
from .feature_extraction import extract_cnn_feature
from .evaluators import extract_extra_features
import shutil
from sklearn.manifold import TSNE
from tqdm import tqdm


def generate_colors(num_colors):
    """
    Generate distinct value by sampling on hls domain.

    Parameters
    ----------
    num_colors: int
        Number of colors to generate.

    Returns
    ----------
    colors_np: np.array, [num_colors, 3]
        Numpy array with rows representing the colors.

    """
    colors = []
    for i in np.arange(0., 360., 360. / num_colors):
        hue = i / 360.
        lightness = 0.5
        saturation = 0.9
        colors.append(colorsys.hls_to_rgb(hue, lightness, saturation))
    colors_np = np.array(colors) * 255.

    return colors_np


def plot_assignment(root, assign_hard, num_parts, img_name, size=None):
    """
    Blend the original image and the colored assignment maps.

    Parameters
    ----------
    root: str
        Root path for saving visualization results.
    assign_hard: np.array, [H, W]
        Hard assignment map (int) denoting the deterministic assignment of each pixel. Generated via argmax.
    num_parts: int, number of object parts.

    Returns
    ----------
    Save the result to root/assignment.png.

    """
    size = size
    # generate the numpy array for colors
    colors = generate_colors(num_parts)

    # coefficient for blending
    coeff = 0.4

    # load the input as RGB image, convert into numpy array
    # input = Image.open(os.path.join(root, 'input.png')).convert('RGB')
    input = Image.open(os.path.join(root, img_name)).convert('RGB')
    input = input.resize((size[1], size[0]), Image.ANTIALIAS)
    input_np = np.array(input).astype(float)

    # blending by each pixel
    for i in range(assign_hard.shape[0]): #128
        for j in range(assign_hard.shape[1]): #256
            assign_ij = assign_hard[i][j]
            input_np[i, j] = (1 - coeff) * input_np[i, j] + coeff * colors[assign_ij]

    # save the resulting image
    im = Image.fromarray(np.uint8(input_np))
    # im.save(os.path.join(root, 'assignment.png'))
    im.save(os.path.join(root, img_name))


def visualization_assignment(
        model, data_loader, dataset_name, args
):
    NPARTS=args.num_parts
    VISUALIZE_NUM=args.VISUALIZE_NUM
    SIZE_TEST=(args.height, args.width)
    save_dir=args.logs_dir
    model.eval()
    with torch.no_grad():
        for i, (imgs, fnames_list, pids, cids, domians) in enumerate(data_loader):
            for id, fnames in enumerate(fnames_list):
                if id >= VISUALIZE_NUM:
                    return

                model_outputs = extract_cnn_feature(model, imgs, args, middle_feat=True)
                assign = model_outputs['soft_assign'][id].unsqueeze(0)
                # _,_,att_list, assign,_,_ = model(input)

                # define root for saving results and make directories correspondingly
                root = os.path.join(save_dir, 'visualization', dataset_name)  # str(current_id)root = os.path.join('./visualization', dataset_name, pic_dir[0].split('/')[-1])
                os.makedirs(root, exist_ok=True)

                # # denormalize the image and save the input
                # # save_input = transforms.Normalize(mean=(0, 0, 0),std=(1/0.229, 1/0.224, 1/0.225))(input.data[0].cpu())
                # # save_input = transforms.Normalize(mean=(-0.485, -0.456, -0.406),std=(1, 1, 1))(save_input)
                #
                # save_input = transforms.Normalize(mean=(0, 0, 0), std=(1 / 0.229* 255, 1 / 0.224* 255, 1 / 0.225* 255))(
                #     input.data[0].cpu())
                # save_input = transforms.Normalize(mean=(-0.485 * 255, -0.456 * 255, -0.406 * 255), std=(1, 1, 1))(save_input)
                #
                # save_input = Image.open(pic_dir[0])
                # save_input = np.array(save_input)
                # save_input = torch.tensor(save_input).permute(2,0,1)
                #
                # save_input = torch.nn.functional.interpolate(save_input.unsqueeze(0), size=cfg.INPUT.SIZE_TEST, mode='bilinear', align_corners=False).squeeze(0)
                # img = torchvision.transforms.ToPILImage()(save_input)
                # # img.save(os.path.join(root, 'input.png'))  #pic_dir[0].split('/')[-1]
                try:
                    shutil.copy(fnames, os.path.join(root, fnames.split('/')[-1]))
                except:
                    # shutil.copy(os.path.join(fnames), os.path.join(root, fnames.split('/')[-1]))
                    continue
                # img.save(os.path.join(root, pic_dir[0].split('/')[-1]))

                # upsample the assignment and transform the attention correspondingly
                assign_reshaped = torch.nn.functional.interpolate(assign.data.cpu(), size=SIZE_TEST,
                                                                  mode='bilinear',
                                                                  align_corners=False)

                # # visualize the attention
                # for k in range(1):
                #     # attention vector for kth attribute
                #     att = att_list[k].view(
                #         1, cfg.INTERPRATABLE.NPARTS, 1, 1).data.cpu()
                #
                #     # multiply the assignment with the attention vector
                #     assign_att = assign_reshaped * att
                #
                #     # sum along the part dimension to calculate the spatial attention map   # ?
                #     attmap_hw = torch.sum(assign_att, dim=1).squeeze(0).numpy()
                #
                #     # normalize the attention map and merge it onto the input
                #     img = cv2.imread(os.path.join(root, pic_dir[0].split('/')[-1]))
                #     mask = attmap_hw / attmap_hw.max()  # ? 0-1
                #     img_float = img.astype(float) / 255.
                #     ddr = os.path.join(root, 'attentions')
                #     if ddr and not os.path.exists(ddr):
                #         mkdir(ddr)
                #     show_att_on_image(img_float, mask, os.path.join(ddr, pic_dir[0].split('/')[-1]))  # , 'attentions'

                # color_att = mpimg.imread(os.path.join(root, 'attentions' + '.png'))
                # axarr_assign_att[j, col_id].imshow(color_att)
                # axarr_assign_att[j, col_id].axis('off')

                # generate the one-channel hard assignment via argmax
                _, assign = torch.max(assign_reshaped, 1)

                # colorize and save the assignment
                plot_assignment(root, assign.squeeze(0).numpy(), NPARTS, fnames.split('/')[-1],
                                size=SIZE_TEST)
                # 画每个assign:
                # collect the assignment for the final image array
                color_assignment_name = os.path.join(root, fnames.split('/')[-1])
                color_assignment = mpimg.imread(color_assignment_name)

                os.makedirs(os.path.join(root, fnames.split('/')[-1].split('.')[0]), exist_ok=True)

                # plot the assignment for each dictionary vector
                for i in range(NPARTS):
                    img = torch.nn.functional.interpolate(assign_reshaped.data[:, i].cpu().unsqueeze(0),
                                                          size=SIZE_TEST, mode='bilinear', align_corners=False)
                    img = torchvision.transforms.ToPILImage()(img.squeeze(0))
                    img.save(os.path.join(root, fnames.split('/')[-1].split('.')[0], 'part_' + str(i) + '.png'))

                # save the array version
                # os.makedirs('./visualization/collected', exist_ok=True)
                # f_assign.savefig(os.path.join('./visualization/collected', 'assign.png'))
                # f_assign_att.savefig(os.path.join('./visualization/collected', 'attention.png'))

    print('Visualization finished!')

def Hbeta_torch(D, beta=1.0):
    P = torch.exp(-D.clone() * beta)

    sumP = torch.sum(P)

    H = torch.log(sumP) + beta * torch.sum(D * P) / sumP
    P = P / sumP

    return H, P

def x2p_torch(X, tol=1e-5, perplexity=30.0):
    """
        Performs a binary search to get P-values in such a way that each
        conditional Gaussian has the same perplexity.
    """

    # Initialize some variables
    print("Computing pairwise distances...")
    (n, d) = X.shape

    sum_X = torch.sum(X*X, 1)
    D = torch.add(torch.add(-2 * torch.mm(X, X.t()), sum_X).t(), sum_X)

    P = torch.zeros(n, n)
    beta = torch.ones(n, 1)
    logU = torch.log(torch.tensor([perplexity]))
    n_list = [i for i in range(n)]

    # Loop over all datapoints
    for i in range(n):

        # Print progress
        if i % 500 == 0:
            print("Computing P-values for point %d of %d..." % (i, n))

        # Compute the Gaussian kernel and entropy for the current precision
        # there may be something wrong with this setting None
        betamin = None
        betamax = None
        Di = D[i, n_list[0:i]+n_list[i+1:n]]

        (H, thisP) = Hbeta_torch(Di, beta[i])

        # Evaluate whether the perplexity is within tolerance
        Hdiff = H - logU
        tries = 0
        while torch.abs(Hdiff) > tol and tries < 50:

            # If not, increase or decrease precision
            if Hdiff > 0:
                betamin = beta[i].clone()
                if betamax is None:
                    beta[i] = beta[i] * 2.
                else:
                    beta[i] = (beta[i] + betamax) / 2.
            else:
                betamax = beta[i].clone()
                if betamin is None:
                    beta[i] = beta[i] / 2.
                else:
                    beta[i] = (beta[i] + betamin) / 2.

            # Recompute the values
            (H, thisP) = Hbeta_torch(Di, beta[i])

            Hdiff = H - logU
            tries += 1

        # Set the final row of P
        P[i, n_list[0:i]+n_list[i+1:n]] = thisP

    # Return final P-matrix
    return P


def pca_torch(X, no_dims=50):
    print("Preprocessing the data using PCA...")
    (n, d) = X.shape
    X = X - torch.mean(X, 0)

    (l, M) = torch.eig(torch.mm(X.t(), X), True)
    # split M real
    # this part may be some difference for complex eigenvalue
    # but complex eignevalue is meanless here, so they are replaced by their real part
    i = 0
    while i < d:
        if l[i, 1] != 0:
            M[:, i+1] = M[:, i]
            i += 2
        else:
            i += 1

    Y = torch.mm(X, M[:, 0:no_dims])
    return Y

def tsne(X, no_dims=2, initial_dims=50, perplexity=30.0):
    """
        Runs t-SNE on the dataset in the NxD array X to reduce its
        dimensionality to no_dims dimensions. The syntaxis of the function is
        `Y = tsne.tsne(X, no_dims, perplexity), where X is an NxD NumPy array.
    """

    # Check inputs
    if isinstance(no_dims, float):
        print("Error: array X should not have type float.")
        return -1
    if round(no_dims) != no_dims:
        print("Error: number of dimensions should be an integer.")
        return -1

    # Initialize variables
    X = pca_torch(X, initial_dims)
    (n, d) = X.shape
    max_iter = 1000
    initial_momentum = 0.5
    final_momentum = 0.8
    eta = 500
    min_gain = 0.01
    Y = torch.randn(n, no_dims)
    dY = torch.zeros(n, no_dims)
    iY = torch.zeros(n, no_dims)
    gains = torch.ones(n, no_dims)

    # Compute P-values
    P = x2p_torch(X, 1e-5, perplexity)
    P = P + P.t()
    P = P / torch.sum(P)
    P = P * 4.    # early exaggeration
    print("get P shape", P.shape)
    P = torch.max(P, torch.tensor([1e-21]))

    # Run iterations
    for iter in range(max_iter):

        # Compute pairwise affinities
        sum_Y = torch.sum(Y*Y, 1)
        num = -2. * torch.mm(Y, Y.t())
        num = 1. / (1. + torch.add(torch.add(num, sum_Y).t(), sum_Y))
        num[range(n), range(n)] = 0.
        Q = num / torch.sum(num)
        Q = torch.max(Q, torch.tensor([1e-12]))

        # Compute gradient
        PQ = P - Q
        for i in range(n):
            dY[i, :] = torch.sum((PQ[:, i] * num[:, i]).repeat(no_dims, 1).t() * (Y[i, :] - Y), 0)

        # Perform the update
        if iter < 20:
            momentum = initial_momentum
        else:
            momentum = final_momentum

        gains = (gains + 0.2) * ((dY > 0.) != (iY > 0.)).double() + (gains * 0.8) * ((dY > 0.) == (iY > 0.)).double()
        gains[gains < min_gain] = min_gain
        iY = momentum * iY - eta * (gains * dY)
        Y = Y + iY
        Y = Y - torch.mean(Y, 0)

        # Compute current value of cost function
        if (iter + 1) % 10 == 0:
            C = torch.sum(P * torch.log(P / Q))
            print("Iteration %d: error is %f" % (iter + 1, C))

        # Stop lying about P-values
        if iter == 100:
            P = P / 4.

    # Return solution
    return Y

def visualize_tsne(seen_domain, seen_list, model, domain_name, args):
    root = os.path.join(args.logs_dir, 'tsne_visualization')  # str(current_id)root = os.path.join('./visualization', dataset_name, pic_dir[0].split('/')[-1])
    os.makedirs(root, exist_ok=True)

    tsne_feat = []
    tsne_label = []

    tsne = TSNE(init='pca', random_state=0, n_components=2, learning_rate=100)
    for id, idata in enumerate(seen_list):
        iloader = seen_domain[idata]['test_loader']
        with torch.no_grad():
            for i, (imgs, fnames_list, pids, cids, domians) in tqdm(enumerate(iloader)):
                if i > 3:
                    break
                model_outputs = extract_cnn_feature(model, imgs, args, middle_feat=True)
                tsne_feat.append(model_outputs['outputs'])
                tsne_label.extend([id]*args.batch_size)
    tsne_feat = torch.cat(tsne_feat, dim=0)
    # with torch.no_grad():
    #     X_tsne = tsne(tsne_feat, 2, 50, 20.0)
    X_tsne = tsne.fit_transform(tsne_feat.cpu().numpy())
    plt.figure(figsize=(6, 6))
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=np.array(tsne_label))
    plt.axis('off')
    plt.savefig(os.path.join(root, f"{domain_name}.png"), dpi=600)


# Mean Magnitude of Channel
def plot_MMC(root, feat_area_size, dataset_name, **kwargs):
    labels = [_ for _ in range(feat_area_size)]
    bn_backbone_feat_stat = kwargs["bn_backbone_feat_stat"]
    bn_feat_stat = kwargs["bn_feat_stat"]
    global_feat_stat = kwargs["global_feat_stat"]
    part_feat_stat = kwargs["part_feat_stat"]

    bb_means = [np.mean(v) for _, v in bn_backbone_feat_stat.items()]
    bpf_means = [np.mean(v) for _, v in part_feat_stat.items()]
    bgf_means = np.mean(bn_feat_stat)

    x = np.arange(len(labels))  # the label locations
    width = 0.1  # the width of the bars

    fig, ax = plt.subplots()
    ax.bar(x - width/2, bb_means)
    for i in bpf_means:
        plt.axhline(y=i, ls="--", lw=2)
    plt.axhline(y=bgf_means, ls="--", lw=2)
    # Add some text for labels, title and custom x-axis tick labels, etc.
    # ax.set_ylabel('Scores')
    ax.set_title('Mean Magnitude of Channel')
    # ax.set_xticks(x, labels)
    ax.legend()

    # ax.bar_label(rects1, padding=3)
    fig.tight_layout()

    fig.savefig(os.path.join(root, f"MMCof{dataset_name}.jpg"))


def visualize_backbone_feat(
        model, data_loader, dataset_name, args
):
    backbone_area_size = 16*8
    part_feat_stat = {_:[] for _ in range(args.num_parts)}
    bn_feat_stat = []
    global_feat_stat = []
    bn_backbone_feat_stat = {_: [] for _ in range(backbone_area_size)}

    save_dir=args.logs_dir

    model.eval()
    with torch.no_grad():
        for i, (imgs, fnames_list, pids, cids, domians) in enumerate(data_loader):
            for id, fnames in enumerate(fnames_list):
                # define root for saving results and make directories correspondingly
                root = os.path.join(save_dir, 'visualization', dataset_name)  # str(current_id)root = os.path.join('./visualization', dataset_name, pic_dir[0].split('/')[-1])
                os.makedirs(root, exist_ok=True)

                model_outputs = extract_cnn_feature(model, imgs, args, middle_feat=True)
                bn_backbone_feat = model_outputs['bn_backbone_feat']
                global_feat = model_outputs['global_feat']
                bn_feat = model_outputs['bn_feat']
                bn_feat_part = model_outputs['bn_feat_part'].view(args.batch_size, args.num_parts, args.part_dim)
                for _ in range(args.num_parts):
                    part_feat_stat[_].extend(bn_feat_part.mean(dim=2)[:, _].tolist())
                for _ in range(backbone_area_size):
                    bn_backbone_feat_stat[_].append(bn_backbone_feat.mean(dim=1)[:, _].tolist())
                bn_feat_stat.extend(bn_feat.mean(dim=1).tolist())

                global_feat_stat.extend(global_feat.mean(dim=1).tolist())
            plot_MMC(
                root, backbone_area_size, dataset_name,
                part_feat_stat=part_feat_stat,
                bn_backbone_feat_stat=bn_backbone_feat_stat,
                bn_feat_stat=bn_feat_stat,
                global_feat_stat=global_feat_stat,
            )
    # return {
    #     "part_feat_stat": part_feat_stat,
    #     "bn_feat_stat": bn_feat_stat,
    #     "global_feat_stat": global_feat_stat,
    #     "bn_backbone_feat_stat": bn_backbone_feat_stat,
    # }