"""
This file is to check the obselete scipy's imresize behavior
"""
import numpy as np
import imageio
import cv2
from PIL import Image
from skimage.transform import resize
import scipy.misc as m  # needs imageio
import matplotlib.pyplot as plt

def main():
    img_path = f'data/frankfurt_000001_078803_leftImg8bit.png'
    lbl_path = f'data/frankfurt_000001_078803_gtFine_labelIds.png'
    img_size = (512, 1024)
    img = imageio.imread(img_path)  # ndarray
    print(img.shape)  # (1016, 2040, 3)
    m_img = m.imresize(img, img_size)  # after resize, img is still ndarray
    pil_img = np.array(Image.fromarray(img).resize((img_size[1], img_size[0]), Image.BILINEAR))  # correct
    # cv2_img = cv2.resize(img, (img_size[1], img_size[0]), interpolation=cv2.INTER_LINEAR)  # no
    # sk_img = (resize(img, img_size) * 255).astype(np.uint8)  # no
    
    # fig = plt.figure(figsize=(10, 10))
    # # Add subplot in 1st position
    # ax1 = fig.add_subplot(2, 2, 1)
    # ax1.imshow(m_img)
    # ax1.axis('off')
    # ax1.set_title("scipy")

    # # Add subplot in 2nd position
    # ax2 = fig.add_subplot(2, 2, 2)
    # ax2.imshow(pil_img)
    # ax2.axis('off')
    # ax2.set_title("PIL")

    # # Add subplot in 3rd position
    # ax3 = fig.add_subplot(2, 2, 3)
    # ax3.imshow(cv2_img)
    # ax3.axis('off')
    # ax3.set_title("cv2_img")

    # # Add subplot in 4th position
    # ax4 = fig.add_subplot(2, 2, 4)
    # ax4.imshow(sk_img)
    # ax4.axis('off')
    # ax4.set_title("sk_img")

    # # Save the figure as an image
    # plt.savefig(f'out_dir/four_ndarrays.png')
    # exit()

    print(img.shape)  # (512, 1024, 3)
    lbl = imageio.imread(lbl_path)
    print(lbl.shape)
    classes = np.unique(lbl)  # ndarray
    lbl = lbl.astype(float)
    # lbl = np.array(Image.fromarray(lbl).resize(size=(self.img_size[0], self.img_size[1])))
    scipy_lbl = m.imresize(lbl, img_size, "nearest", mode="F")
    pil_lbl = np.array(Image.fromarray(lbl).resize((img_size[1], img_size[0]), Image.NEAREST))  # yes
    lbl = lbl.astype(int)
    fig = plt.figure(figsize=(10, 10))
    # Add subplot in 1st position
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(scipy_lbl)
    ax1.axis('off')
    ax1.set_title("scipy")

    # Add subplot in 2nd position
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.imshow(pil_lbl)
    ax2.axis('off')
    ax2.set_title("PIL")

    # Save the figure as an image
    plt.savefig(f'out_dir/lbl_ndarrays.png')

    if not np.all(classes == np.unique(lbl)):
        print("WARN: resizing labels yielded fewer classes")


if __name__ == '__main__':
    main()
