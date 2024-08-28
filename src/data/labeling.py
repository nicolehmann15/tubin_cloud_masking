# TODO:
# nicht registrierte Bilder checken
# wolkenlose / komplett bewölkte Bilder synthetisch registrieren
#  --> herausfinden, welche TIR-Werte eine Wolke / Land aus TUBIN-Sicht hat
#   --> auch gleich Masken generieren
# !!! wolkenlose Bilder gibt es nicht !!! nicht registrierte Bilder konnten aufgrund schlechter TIR-Daten nicht registriert werden !!!
# --> komplett bewölkte Bilder können verwendet werden

# aus registrierten TIFFs den VIS-Teil extrahieren und für segments.ai in .png speichern

# Ergebnisse aus segments.ai verarbeiten
#  --> Masken an TIFFs knüpfen
import os
import tifffile as tiff
import matplotlib.pyplot as plt
import cv2
import numpy as np
from segments import SegmentsClient, SegmentsDataset
from segments.utils import export_dataset, get_semantic_bitmap, bitmap2file

TUBIN_PATH = 'D:/Clouds/data/TUBIN/Preprocessing/buffer'
GT_PATH = 'C:/Users/n_leh/Desktop/Masterarbeit/Praxis/labeling'


def extract_vis_part():
    for campaign in os.listdir(TUBIN_PATH):
        campaign_registered = os.path.join(TUBIN_PATH, campaign + '/Registered')
        campaign_labeling = os.path.join(TUBIN_PATH, campaign + '/Labeling')
        if not os.path.isdir(campaign_labeling):
            os.mkdir(campaign_labeling)
        for reg_product in os.listdir(campaign_registered):
            prod_path = os.path.join(campaign_registered, reg_product)
            # first dimension: TIR / VIS
            # last dimension: RGB - auch für TIR --> IMREAD_GRAYSCALE!
            tiff_bands = tiff.imread(prod_path)
            vis_bands = cv2.cvtColor(tiff_bands[1], cv2.COLOR_BGR2RGB)

            vis_path = os.path.join(campaign_labeling, reg_product.split('.')[0] + '.png')
            cv2.imwrite(vis_path, vis_bands)

            #plt.imshow(tir_band, cmap='gray')
            #plt.show()

            #plt.subplot(1, 2, 1)
            #plt.imshow(tiff_bands[0])
            #plt.subplot(1, 2, 2)
            #plt.imshow(tiff_bands[1])

            #plt.subplot(1, 3, 1)
            #plt.imshow(tiff_bands[0, :, :, 0])
            #plt.subplot(1, 3, 2)
            #plt.imshow(tiff_bands[0, :, :, 1])
            #plt.subplot(1, 3, 3)
            #plt.imshow(tiff_bands[0, :, :, 2])
            #plt.show()
            #exit()


def load_segments_dataset():
    release_path = os.path.join(GT_PATH, 'TUBIN-Labeling-v3.json')
    # Initialize a SegmentsDataset from the release file
    client = SegmentsClient('2f85c781631c2b3ee42d369c7eafb3c8db7602fc')
    release = release_path # client.get_release()
    dataset = SegmentsDataset(release, labelset='ground-truth', filter_by=['reviewed'])

    # Export to semantic 8-bit format
    export_dataset(dataset, export_format='semantic')

    label_path = os.path.join(GT_PATH, 'labeled')

    for sample in dataset:
        # Print the sample name
        print(sample['name'])

        # Show the image
        #plt.imshow(sample['image'])
        #plt.show()

        # Show the instance segmentation label
        #plt.imshow(sample['segmentation_bitmap'])
        #plt.show()

        # Show the semantic segmentation label
        semantic_bitmap = get_semantic_bitmap(sample['segmentation_bitmap'], sample['annotations'])
        #plt.imshow(semantic_bitmap)
        #plt.show()

        labeled_prod_path = os.path.join(label_path, sample['name'])
        semantic_bitmap = np.expand_dims(semantic_bitmap, axis=2)
        cv2.imwrite(labeled_prod_path, semantic_bitmap * 255)


def upload_prelabels():
    release_path = os.path.join(GT_PATH, 'TUBIN-Labeling-Unlabeled Set.json')
    # Initialize a SegmentsDataset from the release file
    client = SegmentsClient('2f85c781631c2b3ee42d369c7eafb3c8db7602fc')
    release = release_path # client.get_release()
    unlabeled_dataset = SegmentsDataset(release, labelset='ground-truth')

    for sample in unlabeled_dataset:
        print(sample['name'])
        labels_path = os.path.join(GT_PATH, 'prelabels')
        annotation_file = os.path.join(labels_path, sample['name'])
        annotation = cv2.imread(annotation_file) * 255
        annotation = cv2.cvtColor(annotation, cv2.COLOR_BGR2GRAY) # np.expand_dims( , axis=2)

        # Upload the predictions to Segments.ai
        file = bitmap2file(annotation)
        asset = client.upload_asset(file, 'Label_' + sample['name'])
        attributes = {
            'format_version': '0.1',
            'annotations': [
                {
                    "id": 1,
                    "category_id": 1
                },
                {
                    "id": 2,
                    "category_id": 2
                }
            ],
            'segmentation_bitmap': {'url': asset.url},
        }
        client.add_label(sample['uuid'], 'ground-truth', attributes, label_status='PRELABELED')

    # label_path = os.path.join(GT_PATH, 'labeled')
    # for file in os.listdir(label_path):
    #    mask = cv2.imread(os.path.join(label_path, file))
    #    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    #    print(mask.shape)
    #    mask = np.array(mask).astype(np.float32)
    #    print(mask.shape)
    #    plt.imshow(mask)
    #    plt.show()


if __name__ == '__main__':
    # extract_vis_part()
    load_segments_dataset()
    #upload_prelabels()
