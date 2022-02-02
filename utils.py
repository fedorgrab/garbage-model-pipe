import matplotlib.pyplot as plt


def get_dataset_sample(dm):
    fig, axs = plt.subplots(nrows=5, ncols=5, figsize=(13, 13))

    for i, (batch, labels) in enumerate(dm.visual_data_sample()):
        for j, (image, label) in enumerate(zip(batch[:5], labels[:5])):
            img = image.numpy().transpose((1, 2, 0))
            axs[i][j].imshow(img)
            axs[i][j].set_xticks([])
            axs[i][j].set_yticks([])
            axs[i][j].set_title(
                f"{dm.train_dataloader().dataset.classes[label.item()]}"
            )

        if i >= 4:
            break

    return fig
