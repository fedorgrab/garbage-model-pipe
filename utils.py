import typing as t
import tarfile
import splitfolders
from google.oauth2.service_account import Credentials
from google.cloud import storage
import matplotlib.pyplot as plt

import constants


def get_google_service_credentials() -> t.Tuple["Credentials", str]:
    credentials = Credentials.from_service_account_info(
        info=constants.GOOGLE_ACCOUNT_CREDENTIALS, scopes=None, default_scopes=None
    )
    return credentials, constants.GOOGLE_ACCOUNT_CREDENTIALS.get("project_id")


def download_file_from_bucket(
    bucket_name: str, source_blob_name: str, destination_file_name: str
) -> None:
    credentials, project_id = get_google_service_credentials()
    storage_client = storage.Client(project=project_id, credentials=credentials)
    bucket = storage_client.bucket(bucket_name)

    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)


def upload_file_to_bucket(local_file_name: str, bucket: str, blob_name: str) -> None:
    credentials, project_id = get_google_service_credentials()
    client = storage.Client(credentials=credentials, project=project_id)
    bucket = client.get_bucket(bucket)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(local_file_name)


def download_data():
    print("Downloading data")
    local_data_tar_name = "./data.tar.gz"
    download_file_from_bucket(
        bucket_name=constants.BUCKET_NAME,
        source_blob_name=constants.STAGE_DATA_NAME,
        destination_file_name=local_data_tar_name,
    )
    f = tarfile.open(local_data_tar_name)
    f.extractall(path=constants.LOCAL_DATA_DIR)
    f.close()


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
