import os
import numpy as np

photo_feature_dir = 'data/photo_features'
photo_features = {}

MEAN = np.load(os.path.join(photo_feature_dir, 'mean.npy'))


def load_photo(photo_id):
  # global photo_features
  if photo_id not in photo_features.keys():
    fa = photo_id[0]
    la = photo_id[1]
    s_list = [fa.upper() + la.upper(), fa.upper() + la.lower(), fa.lower() + la.upper(), fa.lower() + la.lower()]
    for name in s_list:
      path = os.path.join(photo_feature_dir, name)
      if os.path.exists(path):
        photo_feature_path = os.path.join(path, photo_id) + '.npy'
        photo_features[photo_id] = np.load(photo_feature_path)
        break
      else:
        continue
    #photo_feature_path = os.path.join(photo_feature_dir, photo_id[:2], photo_id) + '.npy'
    #photo_features[photo_id] = np.load(photo_feature_path)
  return photo_features[photo_id]


def batch_review_normalize(docs):
  batch_size = len(docs)
  document_sizes = np.array([len(doc) for doc in docs], dtype=np.int32) #每个评论有少个句子[32,]
  document_size = document_sizes.max()
  sentence_sizes_ = [[len(sent) for sent in doc] for doc in docs] #每个句子有多少个词 32
  sentence_size = max(map(max, sentence_sizes_))

  norm_docs = np.zeros(shape=[batch_size, document_size, sentence_size], dtype=np.int32)  # == PAD
  sentence_sizes = np.zeros(shape=[batch_size, document_size], dtype=np.int32)
  for i, document in enumerate(docs):
    for j, sentence in enumerate(document):
      sentence_sizes[i, j] = sentence_sizes_[i][j]
      for k, word in enumerate(sentence):
        norm_docs[i, j, k] = word

  return norm_docs, document_sizes, sentence_sizes, document_size, sentence_size


def batch_image_normalize(batch_images, num_images):
  batch_size = len(batch_images)

  # norm_batch = np.ones(shape=[batch_size, num_images + 1, 4096], dtype=np.float32)
  norm_batch = np.ones(shape=[batch_size, num_images, 4096], dtype=np.float32)
  # norm_batch = norm_batch * MEAN

  for i, review_images in enumerate(batch_images):
    for j, image_id in enumerate(review_images):
      norm_batch[i, j, :] = load_photo(image_id)

  # locl_mean_img = np.expand_dims(np.mean(norm_batch, axis=1), axis=1)
  #
  # norm_batch = np.concatenate((norm_batch, locl_mean_img), axis=1)
  # print(norm_batch.shape)
  # print(locl_mean_img.shape)

  # assert 0 == 1

  # print(norm_batch.shape)
  # img1 = norm_batch[:, 0, :]
  # img2 = norm_batch[:, 1, :]
  # img3 = norm_batch[:, 2, :]
  #
  # img1 = np.expand_dims(img1, axis=1)
  # img2 = np.expand_dims(img2, axis=1)
  # img3 = np.expand_dims(img3, axis=1)
  #
  #
  # batch_imgs = np.concatenate((img1, img2, img3), axis=1)
  # #
  # # batch_imgs = img2
  #
  # return batch_imgs

  # assert 0 == 1
  #
  return norm_batch
