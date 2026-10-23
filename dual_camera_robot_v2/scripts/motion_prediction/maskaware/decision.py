"""Class-dependent observed-chain quality and causal stability; same rule at every stage."""
import numpy as np
DEFAULT = {'min_quality':.15,'min_chain_fraction':.18,'min_hug_arm_fraction':.18,
           'partial_hug_factor':.7,'stable_frames':2,'max_gap_s':.3}
REASONS = ('no_person','insufficient_visible_joints','low_pose_quality','low_action_confidence','unstable_prediction')


def quality_inputs(observed,confidence,person):
    o=np.asarray(observed,bool)
    chains=np.stack([o[:,ids].all(-1).mean() for ids in ([2,4,6],[3,5,7])])
    arms=np.array([o[:,ids].any(-1).mean() for ids in ([4,6],[5,7])])
    return np.array([bool(np.asarray(person).any()),o.mean(),np.where(o,confidence,0).mean(),
                     *chains,*arms,o[:,[2,3]].any(-1).mean()],dtype=float)


def decide(probabilities,quality,episodes,times,threshold,classes,config=None):
    cfg={**DEFAULT,**(config or {})};p=np.asarray(probabilities)
    if p.ndim!=2 or p.shape[1]!=len(classes) or not np.isfinite(p).all() or not np.allclose(p.sum(-1),1,atol=1e-5):
        raise ValueError('Invalid action probabilities')
    if np.isscalar(threshold):
        thresholds=np.full(len(classes),float(threshold))
    elif hasattr(threshold,'keys'):
        missing=set(classes)-set(threshold)
        if missing:raise ValueError(f'Missing class thresholds: {sorted(missing)}')
        thresholds=np.asarray([threshold[label] for label in classes],dtype=float)
    else:
        thresholds=np.asarray(threshold,dtype=float)
    if thresholds.shape!=(len(classes),) or not np.isfinite(thresholds).all() or ((thresholds<0)|(thresholds>1)).any():
        raise ValueError('Invalid class confidence thresholds')
    stable=cfg['stable_frames']
    if np.isscalar(stable):
        stable_frames=np.full(len(classes),int(stable))
    elif hasattr(stable,'keys'):
        missing=set(classes)-set(stable)
        if missing:raise ValueError(f'Missing class stable frames: {sorted(missing)}')
        stable_frames=np.asarray([stable[label] for label in classes],dtype=int)
    else:
        stable_frames=np.asarray(stable,dtype=int)
    if stable_frames.shape!=(len(classes),) or (stable_frames<1).any():
        raise ValueError('Class stable frames must be positive integers')
    labels=np.full(len(p),-1,int);reasons=np.full(len(p),'',dtype='<U32');adjusted=p.max(-1).copy()
    previous=-1;streak=0;last_ep=None;last_t=-np.inf
    for i,(row,q,ep,t) in enumerate(zip(p,quality,episodes,times)):
        k=int(row.argmax()); label=classes[k];reason=''
        if q[0]==0:reason='no_person'
        elif q[1]<cfg['min_quality']:reason='low_pose_quality'
        elif label in ('handshake','fist_bump') and max(q[3:5])<cfg['min_chain_fraction']:
            reason='insufficient_visible_joints'
        elif label=='hug':
            if q[7]<cfg['min_hug_arm_fraction'] or max(q[5:7])<cfg['min_hug_arm_fraction']:
                reason='insufficient_visible_joints'
            elif min(q[5:7])<cfg['min_hug_arm_fraction']:
                adjusted[i]*=cfg['partial_hug_factor']
        if not reason and adjusted[i]<thresholds[k]:reason='low_action_confidence'
        if ep!=last_ep or t-last_t>cfg['max_gap_s']+1e-8 or reason or k!=previous:streak=0
        if not reason:
            streak+=1
            if streak<stable_frames[k]:reason='unstable_prediction'
            else:labels[i]=k
        previous=k if reason in ('','unstable_prediction') else -1
        reasons[i]=reason;last_ep=ep;last_t=t
    return labels,reasons,adjusted
