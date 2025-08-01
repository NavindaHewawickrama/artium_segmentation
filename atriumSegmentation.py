#%matplotlib notebook
from pathlib import Path
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from tqdm import tqdm


root = Path("./imagesTr")
label = Path("./labelsTr")

def change_img_to_label_path(path):
    parts = list(path.parts)
    parts[parts.index("imagesTr")] = "labelsTr"
    return Path(*parts)

sample_path = list(root.glob("la*"))[0]
sample_path_label = change_img_to_label_path(sample_path)

# print(sample_path_label)
# print(sample_path)

data = nib.load(sample_path)
label = nib.load(sample_path_label)

mri = data.get_fdata()
mask = label.get_fdata().astype(np.uint8)

# print(nib.aff2axcodes(data.affine))


#geeting a video from the image set
# from celluloid import Camera
# from IPython.display import HTML

# fig = plt.figure()
# camera = Camera(fig)

# for i in range(mri.shape[2]):
#     plt.imshow(mri[:,:,i], cmap="bone")
#     mask_ = np.ma.masked_where(mask[:,:,i]==0, mask[:,:,i])
#     plt.imshow(mask_, alpha=0.5)
#     camera.snap()
    
# animation = camera.animate()

# HTML(animation.to_html5_video())

#preprocess the dataset
def normalize(full_volume):
    mu = full_volume.mean()
    std = np.std(full_volume)
    normalized = (full_volume - mu) / std
    return normalized

def standardize(normalized):
    standardized = (normalized - normalized.min()) / (normalized.max() - normalized.min())
    return standardized

all_files = list(root.glob("la*"))
len(all_files)

save_root = Path("Preprocessed")

for counter, path_to_mri_data in enumerate(tqdm(all_files)):
    
    path_to_label = change_img_to_label_path(path_to_mri_data)
    
    mri = nib.load(path_to_mri_data)
    assert nib.aff2axcodes(mri.affine) == ("R","A","S")
    mri_data = mri.get_fdata()
    label_data = nib.load(path_to_label).get_fdata().astype(np.uint8)
    
    mri_data = mri_data[32:-32, 32:-32]
    label_data = label_data[32:-32, 32:-32]
    
    normalized_mri_data = normalize(mri_data)
    standardized_mri_data = standardize(normalized_mri_data)
    
    if counter < 17:
        current_path = save_root/"train"/str(counter)
    else:
        current_path = save_root/"val"/str(counter)
        
    for i in range(standardized_mri_data.shape[-1]):
        slice = standardized_mri_data[:,:,i]
        slice_path = current_path/"data"
        mask_path = current_path/"masks"
        slice_path.mkdir(parents=True,exist_ok=True)
        mask_path.mkdir(parents=True,exist_ok=True)