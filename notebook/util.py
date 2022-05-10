import matplotlib.pyplot as plt
import numpy as np
import os.path as path
from matplotlib.patches import Circle, Wedge, Polygon, Ellipse
from matplotlib.collections import PatchCollection
import pandas as pd
import re
import itertools as it
import pickle
import cv2
import io
import datetime


class Sample:
    def __init__(self, img, coordinates, texts, name=None):
        self.name = name
        self.img = img
        self.coordinates = coordinates
        self.texts = texts
        self.centers = np.stack([np.average(self.coordinates[:, :, 0], axis=1), np.average(self.coordinates[:, :, 1], axis=1)], axis=1)
        
def show_sample(s, co_scale=(1,1), 
                figsize=(50, 20), 
                priority=None, filter_=None,
                box=True, cross=None, ax=None, imshow=True, box_text=False):
    
    ax1 = ax
    if ax1 is None:
        fig, ax1 = plt.subplots(figsize=figsize)
    
    ax1.xaxis.set_ticks_position('top')
    if imshow:
        ax1.imshow(s.img)
    
    if filter_ is None:
        filter_ = True; #true means all values, which means no filter
    
    if isinstance(co_scale, int):
        co_scale = (co_scale, co_scale)
    
    if co_scale == 'img':
        co_scale = s.img.shape[:2][::-1]
    
    if priority is not None:
        p_order = np.argsort(np.argsort(priority))
        cx, cy = s.centers[:,0] ,s.centers[:,1]
        bb = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        for x,y,p in zip(cx, cy, p_order):
            ax1.text(x*co_scale[0], y*co_scale[1], p, bbox=bb)
        #ax1.scatter(cx*co_scale[0], cy*co_scale[1], s=priority, c='red')
 
    if cross is not None:
        ax1.scatter([cross[0]*co_scale[0]], [cross[1]*co_scale[1]], [200], marker='x', c='purple')
        
    patches = []
    if box: 
        patches += [Polygon(coordinate*co_scale, True) for coordinate in s.coordinates]
    p = PatchCollection(patches, alpha=0.3)
    ax1.add_collection(p)
    
    if box_text:
        for cor, t in zip(s.coordinates, s.texts):
            ax1.text(cor[0][0]*co_scale[0], cor[0][1]*co_scale[1], t)

def load_data(fn, data_root):
    fp_img = path.join(data_root, 'imgs', fn + '.jpg')
    fp_map = path.join(data_root, 'map', fn + '.txt')
    fp_anno = path.join(data_root, 'annotation', fn + '.txt')

    with open(fp_map, 'r') as f:
        lines = f.readlines()

    tmp = [l.split(',') for l in lines]
    coordinates = [list(map(float, l[:8])) for l in tmp]
    coordinates = np.reshape(coordinates, (len(coordinates), 4, 2))
    texts = [','.join(l[8:])[:-1] for l in tmp]
    
    img = cv2.imread(fp_img)

    return Sample(img, coordinates, texts, name=fn)

def pipeline1(fn, data_root):
    sp = load_data(fn, data_root)
    sp = replace_non_alpha_2_space(sp) # replace non-alpha to space
    sp = split2word(sp) #split words by space
    sp = crop_n_normalize(sp) #crop n normalize both x and y axes
    sp = filter_words(sp, np.char.isalpha(sp.texts))
    sp.name = fn
    
    return sp

def split2word(sp):
    flatten = sp.coordinates.reshape(sp.coordinates.shape[0],-1)

    df = pd.DataFrame(flatten)
    df['tlen'] = [len(t) for t in sp.texts]

    df['width_per_char_on_top']=(df[2]-df[0])/df['tlen']
    df['height_per_char_on_top']=(df[3]-df[1])/df['tlen']
    df['width_per_char_on_bottom']=(df[4]-df[6])/df['tlen']
    df['height_per_char_on_bottom']=(df[5]-df[7])/df['tlen']

    sp_len = [list(map(len, t.split(' '))) for t in sp.texts]
    sp_texts = [t.split(' ') for t in sp.texts]

    tmp = [[[
      #i, sum(line[:i])+i, sum(line[:i])+i+le, 
      r[1][0]+(sum(line[:i])+i)*r[1]['width_per_char_on_top'], 
      r[1][1]+(sum(line[:i])+i)*r[1]['height_per_char_on_top'], 

      r[1][0]+(sum(line[:i])+i+le)*r[1]['width_per_char_on_top'], 
      r[1][1]+(sum(line[:i])+i+le)*r[1]['height_per_char_on_top'], 

      r[1][6]+(sum(line[:i])+i+le)*r[1]['width_per_char_on_bottom'], 
      r[1][7]+(sum(line[:i])+i+le)*r[1]['height_per_char_on_bottom'], 

      r[1][6]+(sum(line[:i])+i)*r[1]['width_per_char_on_bottom'], 
      r[1][7]+(sum(line[:i])+i)*r[1]['height_per_char_on_bottom']
      ],  st[i]
    ]
     for line, st, r in zip(sp_len, sp_texts, df.iterrows()) 
     for i, le in enumerate(line) ]

    cor, txt = zip(*tmp)
    cor, txt = list(cor), list(txt)

    cor = np.reshape(cor, (len(cor), 4, 2))
    
    return Sample(sp.img, cor, txt)

def crop_n_normalize(sp):
    xs = sp.coordinates[:, :, 0]
    xmin, xmax = xs.min(), xs.max()

    ys = sp.coordinates[:, :, 1]
    ymin, ymax = ys.min(), ys.max()

    co = np.copy(sp.coordinates)
    co[:, :, 0] = (co[:, :, 0] - xmin) / (xmax-xmin)
    co[:, :, 1] = (co[:, :, 1] - ymin) / (ymax-ymin)

    img = np.copy(sp.img)
    img = img[int(ymin):int(ymax)+1, int(xmin):int(xmax)+1, :]
    
    return Sample(img, co, np.copy(sp.texts))

def filter_words(sp, filter_):
    return Sample(np.copy(sp.img), np.copy(sp.coordinates)[filter_], np.copy(sp.texts)[filter_])

def replace_non_alpha_2_space(sp):
    texts = [re.sub('[^A-z0-9]', ' ', t) for t in sp.texts]
    return Sample(np.copy(sp.img), np.copy(sp.coordinates), texts)

def priority(x, y, x0, y0, p_range=300):
    '''
    determine the priority in the order of looking for anchor waypoint, prefer left then right, above then below
    '''
    #left_right_bias > 0 means position on the left has higher priority
    #top_bottom_bias > 0 means position on the top has higher priority than in the bottom
    left_right_bias, top_down_bias = 0,0 #x0/10, y0/10
    
    #x_priority_weight/y_priority_weight the ratio determine the weight priority in X over Y the same distant
    x_priority_weight, y_priority_weight = 5, 1
    p = ((x-left_right_bias-x0)**2*x_priority_weight + (y-top_down_bias-y0)**2*y_priority_weight)**0.5
    p = p/np.max(p)*p_range
    return p

def priority_eq(points, x0, y0):
    '''
    determine the priority in the order of looking for anchor waypoint, prefer closer waypointer
    '''
    x, y = points[:,0], points[:,1]
    p = ((x-x0)**2 + (y-y0)**2)**0.5
    #p = np.argsort(np.argsort(p))
    return p

def try_mapping_1(sp1, sp2, max_mapping=5):
    st1 = set(sp1.texts)
    st2 = set(sp2.texts)
    shared_keyword = np.sort(list(st1.intersection(st2)))

    # try to match the shared keywords between the 2 samples

    sp1_matched, sp2_matched = np.asarray([], dtype=np.int64), np.asarray([], dtype=np.int64)
    total_occur1, total_occur1 = 0, 0
    for keyword in shared_keyword:

        keyword_idx_1, keyword_idx_2 = np.argwhere(sp1.texts==keyword).flatten(), np.argwhere(sp2.texts==keyword).flatten()
        #due to memery limitation, limit the maxmum occurances that will be matched
        #cause i am use permutation to compare all the possible mapping at once
        #e.g. when mapping 9 occurances to 11 there are 19,958,400 possible matches
        # if choose to only map 5 to 11, the possible matches reduce to 55,440
        if len(keyword_idx_1)>len(keyword_idx_2):
            keyword_idx_2 = keyword_idx_2[:max_mapping]
        else:
            keyword_idx_1 = keyword_idx_1[:max_mapping]
        
        centers1, centers2 = sp1.centers[keyword_idx_1], sp2.centers[keyword_idx_2]

        # for each shared keyword: k
        # for its occurance in the 2 sampels: c1, c2
        # all the possible mapping: permutation(c1, c2)  (c1>=c2)

        if len(centers1)>=len(centers2):
            ct_more, ct_less = centers1, centers2
        else:
            ct_more, ct_less = centers2, centers1
        len_more, len_less = len(ct_more), len(ct_less)

        p = list(it.permutations(range(len_more), len_less))
        idx_more = np.array(p).reshape(len(p), len_less)
        idx_less = np.repeat(np.arange(len_less).reshape(1, len_less), len(p), axis=0)

        # all possible mappings for all the occurance of keyword k
        # get the norm of the differences in centers for each mapping set
        # calculate the avg and std of each mapping set, to be used as features

        diff = ct_more[idx_more] - ct_less[idx_less]
        norm_diff = np.linalg.norm(diff, axis=2)

        norm_diff_avg = np.average(norm_diff, axis=1)
        norm_diff_std = np.std(norm_diff, axis=1)

        ## for now just use the one has less average sum distance of the all mappings of k
        #### for some special case that, the content shifted just about, where the nth k in sample 1 overlap(position) with mth k in sample 2, which would give 0 or very small distance, but they are not the right match
        ## so in future should consider also the std

        corr_mapping = norm_diff_avg.argmin() #, norm_diff_std.argmin()

        if len(centers1)>=len(centers2):
            idx_sp1, idx_sp2 = idx_more[corr_mapping], idx_less[corr_mapping]
        else:
            idx_sp1, idx_sp2 = idx_less[corr_mapping], idx_more[corr_mapping]

        sp1_matched = np.concatenate((sp1_matched, keyword_idx_1[idx_sp1]))
        sp2_matched = np.concatenate((sp2_matched, keyword_idx_2[idx_sp2]))
        
    
    diff = sp1.centers[sp1_matched] - sp2.centers[sp2_matched]
    norm_diff = np.linalg.norm(diff, axis=1)
    norm_diff_avg = np.average(norm_diff)
    norm_diff_std = np.std(norm_diff)
    
    return sp1_matched, sp2_matched, len(shared_keyword), np.count_nonzero(np.isin(sp1.texts, shared_keyword)), np.count_nonzero(np.isin(sp2.texts, shared_keyword)), norm_diff_avg, norm_diff_std

def show_mapping(sp1, sp2, 
                figsize=(10, 10), 
                ax=None,
                mapping=None, imshow=True, invert_y=True):
    
    ax1 = ax

    if imshow:
        f, axs = plt.subplots(1, 3, figsize=(figsize[0]*3, figsize[1]))
        ax0, ax1, ax2 = axs
        ax0.imshow(sp1.img)
        ax2.imshow(sp2.img)
        
    if ax1 is None:
        fig, ax1 = plt.subplots(figsize=figsize)
    
    ax1.set_ylim(ymin=0, ymax=1)
    ax1.set_xlim(xmin=0, xmax=1)
    if invert_y:
        ax1.xaxis.set_ticks_position('top')
        ax1.invert_yaxis()
    
    for s, c in zip([sp1, sp2], ['red', 'green']):
        for cor, t in zip(s.centers, s.texts):
            ax1.text(cor[0], cor[1], t, c=c) #, bbox=dict(facecolor=c, alpha=0.5))
            #for cor, t in zip(sp1.coordinates, sp1.texts):
            #    ax1.text(cor[0][0], cor[0][1], t, bbox=dict(facecolor='blue', alpha=0.5))
            
    if mapping:
        for c1, c2 in zip(sp1.centers[mapping[0]], sp2.centers[mapping[1]]):
            xy = np.stack([c1, c2], axis=1)
            ax1.plot(xy[0, :], xy[1, :], linewidth=2,  c='black')
            
def process(pairs, samples, show_log=True):
    df = pd.DataFrame(columns=['sp1_idx', 'sp2_idx', 'sp1_matched', 'sp2_matched', 'n_matched', 'n_shared_kw', 'n_kw_occur_sp1', 'n_kw_occur_sp2', 'norm_diff_avg', 'norm_diff_std'])
    
    keys = list(samples.keys())
    
    #pair_idx, times, name1, name2, norm_avgs, norm_stds = [], [], [], [], [], []
    for i, pair in enumerate(pairs):
        bg = datetime.datetime.now()
        idx1, idx2 = pair
        idx1, idx2 = keys[idx1], keys[idx2]
        sp1, sp2 = samples[idx1], samples[idx2]
        sp1_matched, sp2_matched, n_shared_kw, n_kw_occur_sp1, n_kw_occur_sp2, norm_diff_avg, norm_diff_std = try_mapping_1(sp1, sp2)
        
        process_time = datetime.datetime.now()-bg
        #pair_idx.append(i)
        #times.append(process_time)        
        #name1.append(sp1.name)
        #name2.append(sp2.name)
        #norm_avgs.append(norm_diff_avg)
        #norm_stds.append(norm_diff_std)
        if show_log and i%5000==0:
            print(i, idx1, idx2, process_time)
            
        df.loc[len(df.index)] = [idx1, idx2, sp1_matched, sp2_matched, len(sp1_matched), n_shared_kw, n_kw_occur_sp1, n_kw_occur_sp2, norm_diff_avg, norm_diff_std]
        
    return df


def debug_process(pairs, samples):
    bg = datetime.datetime.now()
    n = len(pairs)
    print('processing:', n)
    df = process(pairs, samples)
    el = datetime.datetime.now()-bg
    print('process done:', n)
    print('elapse:', el)
    print('avg time per pair:', el/n)
    return df


def load_or_generate_samples_dict(fn, data_root):
    dumped_samples = path.join(data_root, 'temp', fn)

    if path.isfile(dumped_samples):
        with open(dumped_samples, 'rb') as f:
            samples = pickle.load(f)
    else:       
        samples = {fn:pipeline1(fn, data_root) for fn in files}
        #np.random.shuffle(samples)
        with open(dumped_samples, 'wb') as f:
            pickle.dump(samples, f)
    return samples

def load_or_generate_df(fn, data_root, samples):
    dumped_df = path.join(data_root, 'temp', fn)
    pairs = list(it.combinations(range(len(samples)), 2))

    if path.isfile(dumped_df):
        with open(dumped_df, 'rb') as f:
            df = pickle.load(f)
    else:       
        df = debug_process(pairs, samples)

        with open(dumped_df, 'wb') as f:
            pickle.dump(df, f)
    return df

def save_df(df, fn, data_root):
    dumped_df = path.join(data_root, 'temp', fn)
    with open(dumped_df, 'wb') as f:
        pickle.dump(df, f)