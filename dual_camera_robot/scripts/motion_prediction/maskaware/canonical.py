"""Canonical-8 v2: independent observed image XY/model depth; causal root fallback."""
import numpy as np
from ..pose_adapter import JOINT_NAMES, SOURCE_MAPPING, Z_SCALE
from ..sequence_utils import validate_times
VERSION = 'canonical_upper_body_v2_maskaware'


def adapt(raw, width, height, threshold=.3, max_age=.3):
    t = validate_times(raw['video_time_s'])
    xy, xyz, score = (np.asarray(raw[k]) for k in ('keypoints_2d','keypoints_3d','scores'))
    if xy.shape != (len(t),133,2) or xyz.shape != (len(t),133,3) or score.shape != (len(t),133):
        raise ValueError('Expected COCO133 arrays')
    if width <= 0 or height <= 0 or not 0 <= threshold <= 1 or not 0 <= max_age <= .3:
        raise ValueError('Invalid dimensions/threshold/age')
    detected = np.asarray(raw['person_detected'],bool)
    valid = np.asarray(raw['visibility'],bool) & (score >= threshold) & np.isfinite(score)
    valid &= np.isfinite(xy).all(-1) & np.isfinite(xyz[...,2]) & detected[:,None]
    coords = np.concatenate((xy / [width,height],xyz[...,2:]/Z_SCALE),-1)
    pose = np.stack([coords[:,ids].mean(1) for ids in SOURCE_MAPPING],1)
    observed = np.stack([valid[:,ids].all(1) for ids in SOURCE_MAPPING],1)
    conf = np.stack([score[:,ids].min(1) for ids in SOURCE_MAPPING],1)
    conf = np.where(observed,np.clip(conf,0,1),0)
    pose[~observed] = np.nan
    age = np.full((len(t),8),np.inf)
    input_pose = pose.copy(); imputed = np.zeros_like(observed)
    root = np.full((len(t),3),np.nan); root_source = np.full(len(t),'invalid',dtype='<U10')
    root_observed = np.zeros(len(t),bool); root_age = np.full(len(t),np.inf)
    last = np.full((8,3),np.nan); last_time = np.full(8,-np.inf)
    last_root = np.full(3,np.nan); last_root_time = -np.inf
    for i, time in enumerate(t):
        last[observed[i]] = pose[i,observed[i]]; last_time[observed[i]] = time
        age[i] = time-last_time
        short = ~observed[i] & (age[i] <= max_age+1e-8) & detected[i]
        input_pose[i,short] = last[short]; imputed[i] = short
        for j, source in ((1,'shoulders'),(0,'pelvis')):
            if observed[i,j]:
                root[i] = pose[i,j]; root_source[i] = source; root_observed[i] = True
                last_root,last_root_time = root[i].copy(),time
                break
        root_age[i] = time-last_root_time
        if not root_observed[i] and detected[i] and root_age[i] <= max_age+1e-8:
            root[i] = last_root; root_source[i] = 'held'
    root_valid = np.isfinite(root).all(-1)
    rel = pose-root[:,None]
    rv = observed & root_valid[:,None]
    rel[~rv] = np.nan
    return dict(pose=pose,visibility=observed,observed=observed,confidence=conf,age=age,
                imputed=imputed,input_pose=input_pose,input_valid=observed|imputed,
                root=root,root_valid=root_valid,root_observed=root_observed,root_age=root_age,
                root_source=root_source,root_relative_pose=rel,root_relative_visibility=rv,
                video_time_s=t,person_detected=detected,source_frame_index=raw['source_frame_index'])


def features(pose,mask,confidence,observed,imputed,age,relative,relative_mask):
    """No targets. Z stays model-relative; relative XYZ gets its own validity channel."""
    x = np.nan_to_num(pose,nan=0.,posinf=0.,neginf=0.)
    v = np.zeros_like(x); vm = np.zeros_like(observed)
    vm[1:] = observed[1:] & observed[:-1]
    v[1:] = np.diff(x,axis=0)*10
    v = np.where(vm[...,None],v,0)
    seq = np.concatenate((x,np.nan_to_num(relative),v,mask[...,None],confidence[...,None],
        observed[...,None],imputed[...,None],np.minimum(age,1.)[...,None],relative_mask[...,None]),-1).astype(np.float32)
    flat = seq.reshape(len(seq),-1)
    return seq, np.concatenate((flat.ravel(),flat.mean(0),flat.std(0))).astype(np.float32)
