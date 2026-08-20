import numpy as np

from ov_probe.pixel_ovss import pixel_confusion
from ov_probe.remoteclip_potsdam_baseline import CLASSES, crop_views, pixel_confusion_fast, score_methods

def test_crop_views_reconstructs_masked_uint8():
    image=np.full((32,32,3),100,dtype=np.uint8); mask=np.zeros((8,8),dtype=bool); mask[2:6,2:6]=True
    context,masked,box=crop_views(image,mask,10,10,min_crop_size=16)
    assert context.dtype==np.uint8 and masked.dtype==np.uint8 and box[2]>box[0]
    assert np.any(masked<context)

def test_score_methods_full_support_has_all_methods():
    rng=np.random.default_rng(0); t=rng.normal(size=(4,5)); v=rng.normal(size=(4,5)); tp=rng.normal(size=(5,8)); vp=rng.normal(size=(5,8))
    tp=tp/np.linalg.norm(tp,axis=1,keepdims=True); vp=vp/np.linalg.norm(vp,axis=1,keepdims=True)
    pred,scores=score_methods(t,v,tp,vp)
    assert set(pred) == {"text_only", "C2", "SCC", "CTP"}
    assert all(x.shape == (4,) for x in pred.values())
    assert all(x.shape == (4, 5) for x in scores.values())
    assert np.array_equal(pred["C2"], pred["SCC"])
    assert np.array_equal(pred["SCC"], pred["CTP"])

def test_remoteclip_classes_are_potsdam_five():
    assert CLASSES==['impervious_surface','building','low_vegetation','tree','car']


def test_fast_confusion_is_identical_to_frozen_implementation():
    rng = np.random.default_rng(2)
    pred = rng.integers(0, 6, size=(11, 13), dtype=np.int64)
    gt = rng.integers(0, 6, size=(11, 13), dtype=np.int64)
    pred[0, :3] = 255
    gt[1, :3] = 255
    assert pixel_confusion_fast(pred, gt, CLASSES) == pixel_confusion(pred, gt, CLASSES)
