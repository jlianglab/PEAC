import numpy as np
import cv2
import os

def display_current_rec(self, epoch, visuals,is_resize=True):
        
        # mean = (0.5, 0.5, 0.5)
        mean = (0.0, 0.0, 0.0)
        std = (1.0, 1.0, 1.0)
        images,recs,diffs = visuals['images'],visuals['recs'],visuals['diffs']
        concat_results = []
        for i in range(images.shape[0]):
            img = images[i].transpose((1,2,0))
            img = ((img * std + mean) * 255).astype(np.uint8)
            rec_img = recs[i].transpose((1,2,0))
            rec_img = ((rec_img * std + mean) * 255).astype(np.uint8)
            diff_map = self.plot_heatmap(diffs[i])
            concat_results.append(np.concatenate([img,rec_img,diff_map],axis=1))
        concat_results = np.concatenate(concat_results,axis=0)
        if is_resize:
            concat_results = cv2.resize(concat_results,dsize=None,fx=0.5,fy=0.5,interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(os.path.join(self.opt['checkpoint_dir'], 'visuals',  'img_rec_diff_' + str(epoch) + '.png'),concat_results[:,:,::-1])