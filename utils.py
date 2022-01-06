import matplotlib.pyplot as plt


def get_dataset_sample(dm):
    val_samples = next(iter(dm.visual_data_sample()))
    val_imgs, val_labels = val_samples[0], val_samples[1]

    fig, axs = plt.subplots(nrows=1, ncols=5, figsize=(10, 2))
    imgs_pl = val_imgs[:5].numpy().transpose((0, 2, 3, 1))

    for i, (img, label) in enumerate(zip(imgs_pl, val_labels)):
        axs[i].imshow(img)
        axs[i].set_xticks([])
        axs[i].set_yticks([])
        axs[i].set_title(f"Label = {dm.train_dataloader().dataset.classes[label.item()]}")

    return fig
