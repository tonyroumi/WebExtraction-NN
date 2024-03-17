
# import sys
# import utils
# import argparse
# import numpy as np
# import __init__paths

# def get_position_probabilities(position_maps, boxes):
#     box_i = 0
#     probs = np.zeros((boxes.shape[0],len(position_maps)),dtype=np.float32)

#     for box_i in range(boxes.shape[0]):
#         for cls in range(len(position_maps)):
#             map = position_maps[cls]
#             box = boxes[box_i]
#             box_map = map[box[1]:box[3],box[0]:box[2]]
#             box_cls_prob = np.mean(box_map)
#             probs[box_i,cls] = box_cls_prob
#     return probs

# def get_probabilities_with_position(boxes, local_probs, position_maps):
#     #-- get position probability for each box
#     position_probs = get_position_probabilities(position_maps, boxes)

#     #-- multiply with local prob
#     probs = (local_probs*position_probs)
#     return probs

# def get_results_with_position(boxes, local_probs, position_maps):
#     #-- get probabalities
#     probs = get_probabilities_with_position(boxes, local_probs, position_maps)[:,1:4]

#     #-- are the first 3 those with maximum probability
#     max_inds = np.argmax(probs,axis=0)
#     results = [0]*3
#     for cls in range(0,3):
#         if max_inds[cls] == cls:
#             results[cls]=1

#     return results


# def test_net(test_model, snapshot_path, test_data, test_iters, position_maps):
#     test_net = caffe.Net(test_model, snapshot_path, caffe.TEST)
#     test_net.layers[0].set_data(test_data)

#     price_results = []
#     name_results = []
#     image_results = []

#     position_price_results = []
#     position_name_results = []
#     position_image_results = []

#     # go through data
#     for i in range(test_iters):
#         test_net.forward()

#         #--- net results
#         price_results.append(test_net.blobs['web_price_accuracy'].data[0])
#         image_results.append(test_net.blobs['web_image_accuracy'].data[0])
#         name_results.append(test_net.blobs['web_name_accuracy'].data[0])

#         #--- results with position maps
#         local_probs = test_net.blobs['prob'].data[:,0:4,0,0]
#         boxes = test_net.blobs['boxes'].data[:,1:5]
#         results_with_position = get_results_with_position(boxes, local_probs, position_maps)
#         position_price_results.append(results_with_position[0])
#         position_image_results.append(results_with_position[1])
#         position_name_results.append(results_with_position[2])


#     # stop fetcher
#     test_net.layers[0].stop_fetcher()

#     # compute net results
#     image_accuracy = np.mean(image_results)
#     price_accuracy = np.mean(price_results)
#     name_accuracy = np.mean(name_results)

#     # compute position results
#     position_image_accuracy = np.mean(position_image_results)
#     position_price_accuracy = np.mean(position_price_results)
#     position_name_accuracy = np.mean(position_name_results)


#     net_results = (image_accuracy, price_accuracy, name_accuracy)
#     position_results = (position_image_accuracy, position_price_accuracy, position_name_accuracy)

#     return net_results, position_results

# #----- MAIN PART
# if __name__ == "__main__":

#     #--- Get params
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--split', type=int, help='Split number', required=True)
#     parser.add_argument('--test_model', type=str, default=None, help='test net prototxt', required=True)
#     parser.add_argument('--snapshot', help='Snapshot path',
#                         default='/storage/plzen1/home/gogartom/DOM-Extraction/data/imagenet_models/CaffeNet.v2.caffemodel', type=str)
#     parser.add_argument('--test_iters', type=int, default=1000, help='Number of iterations for testing')
#     parser.add_argument('--experiment', type=str, default=None, help='name of experiment', required=True)
#     parser.add_argument('--gauss_var', type=int, default=80, help='Variance of gaussian kernel')
#     args = parser.parse_args()

#     #-- Load params
#     split_name = str(args.split)
#     test_iters = args.test_iters
#     test_model = args.test_model
#     snapshot = args.snapshot
#     experiment = args.experiment
#     gauss_var = args.gauss_var

#     #--- GPU
#     caffe.set_mode_gpu()

#     #--- LOAD SMOTHED POSITION MAPS
#     position_maps = utils.load_position_maps(split_name, gauss_var)

#     #--- LOAD TEST DATA
#     test_data = utils.get_test_data_path(split_name)

#     #--- GET TEST RESULTS PATH
#     test_res_path = utils.get_result_path(experiment, split_name)

#     ###---  TEST
#     print('Testing')
#     sys.stdout.flush()

#     net_results, position_results = test_net(test_model, snapshot, test_data, test_iters, position_maps)

#     im_acc, price_acc, name_acc = net_results
#     print 'NET: image accuracy:', im_acc
#     print 'NET: price accuracy:', price_acc
#     print 'NET: name accuracy:', name_acc

#     p_im_acc, p_price_acc, p_name_acc = position_results
#     print 'NET+POSITION: image accuracy:', p_im_acc
#     print 'NET+POSITION: price accuracy:', p_price_acc
#     print 'NET+POSITION: name accuracy:', p_name_acc
#     sys.stdout.flush()

#     ###--- save results
#     with open(test_res_path, 'w+') as f:
#         f.write('NET: image accuracy: '+str(im_acc)+"\n")
#         f.write('NET: price accuracy: '+str(price_acc)+"\n")
#         f.write('NET: name accuracy: '+str(name_acc)+"\n")
#         f.write('\n')
#         f.write('NET+POSITION: image accuracy: '+str(p_im_acc)+"\n")
#         f.write('NET+POSITION: price accuracy: '+str(p_price_acc)+"\n")
#         f.write('NET+POSITION: name accuracy: '+str(p_name_acc)+"\n")

