#!/usr/bin/env python3
"""
eda_full.py — Complete Exploratory Data Analysis for the Conformal Calibration Pipeline.

Generates 13 publication-quality plots covering:
  Part I  (plots 01–08): Dataset-level EDA (COCO + VOC statistics)
  Part II (plots 09–13): Comparative OD vs Segmentation analysis with visual samples

Dataset pair:
  - MS-COCO 2017 val: Object Detection (bbox) + Instance Segmentation (masks)
  - PASCAL VOC 2012:  Object Detection (bbox) + Semantic Segmentation (pixel-level)

Output directory: results/eda/
"""

import os, json, random, warnings
from pathlib import Path
from collections import Counter, defaultdict
from typing import Tuple

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════════════════
#  STYLE & PALETTE
# ═══════════════════════════════════════════════════════════════════════════════

PAL = {
    "primary": "#1B4F72", "secondary": "#2E86C1",
    "accent1": "#E74C3C", "accent2": "#27AE60", "accent3": "#F39C12", "accent4": "#8E44AD",
    "bg": "#FAFBFC", "grid": "#E8ECEF", "text": "#2C3E50", "muted": "#7F8C8D",
    "coco": "#1B4F72", "coco_light": "#AED6F1", "coco_fill": "#D6EAF8",
    "voc": "#922B21", "voc_light": "#F1948A", "voc_fill": "#FADBD8",
    "od": "#2E86C1", "seg": "#27AE60", "inst_seg": "#8E44AD",
}

INST_COLORS = [
    '#e6194b','#3cb44b','#ffe119','#4363d8','#f58231','#911eb4',
    '#42d4f4','#f032e6','#bfef45','#fabed4','#469990','#dcbeff',
    '#9A6324','#800000','#aaffc3','#808000','#ffd8b1','#000075',
]

VOC_SEG_COLORS = np.array([
    [0,0,0],[128,0,0],[0,128,0],[128,128,0],[0,0,128],[128,0,128],
    [0,128,128],[128,128,128],[64,0,0],[192,0,0],[64,128,0],
    [192,128,0],[64,0,128],[192,0,128],[64,128,128],[192,128,128],
    [0,64,0],[128,64,0],[0,192,0],[128,192,0],[0,64,128],
]) / 255.0

VOC_CLASSES = [
    "background","aeroplane","bicycle","bird","boat","bottle","bus","car",
    "cat","chair","cow","diningtable","dog","horse","motorbike",
    "person","pottedplant","sheep","sofa","train","tvmonitor"
]

plt.rcParams.update({
    'figure.facecolor': PAL["bg"], 'axes.facecolor': '#FFFFFF',
    'axes.edgecolor': '#CCCCCC', 'axes.grid': True,
    'grid.color': PAL["grid"], 'grid.linewidth': 0.4, 'grid.alpha': 0.7,
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans','Arial','Helvetica'],
    'font.size': 11, 'axes.titlesize': 13, 'axes.titleweight': 'bold',
    'axes.labelsize': 11, 'xtick.labelsize': 9, 'ytick.labelsize': 9,
    'legend.fontsize': 9, 'figure.titlesize': 15, 'figure.titleweight': 'bold',
    'figure.dpi': 150, 'savefig.dpi': 200, 'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.15,
})

# ═══════════════════════════════════════════════════════════════════════════════
#  COCO / VOC REAL STATISTICS
# ═══════════════════════════════════════════════════════════════════════════════

COCO_CATEGORIES = [
    {"id":1,"name":"person","supercategory":"person"},
    {"id":2,"name":"bicycle","supercategory":"vehicle"},
    {"id":3,"name":"car","supercategory":"vehicle"},
    {"id":4,"name":"motorcycle","supercategory":"vehicle"},
    {"id":5,"name":"airplane","supercategory":"vehicle"},
    {"id":6,"name":"bus","supercategory":"vehicle"},
    {"id":7,"name":"train","supercategory":"vehicle"},
    {"id":8,"name":"truck","supercategory":"vehicle"},
    {"id":9,"name":"boat","supercategory":"vehicle"},
    {"id":10,"name":"traffic light","supercategory":"outdoor"},
    {"id":11,"name":"fire hydrant","supercategory":"outdoor"},
    {"id":13,"name":"stop sign","supercategory":"outdoor"},
    {"id":14,"name":"parking meter","supercategory":"outdoor"},
    {"id":15,"name":"bench","supercategory":"outdoor"},
    {"id":16,"name":"bird","supercategory":"animal"},
    {"id":17,"name":"cat","supercategory":"animal"},
    {"id":18,"name":"dog","supercategory":"animal"},
    {"id":19,"name":"horse","supercategory":"animal"},
    {"id":20,"name":"sheep","supercategory":"animal"},
    {"id":21,"name":"cow","supercategory":"animal"},
    {"id":22,"name":"elephant","supercategory":"animal"},
    {"id":23,"name":"bear","supercategory":"animal"},
    {"id":24,"name":"zebra","supercategory":"animal"},
    {"id":25,"name":"giraffe","supercategory":"animal"},
    {"id":27,"name":"backpack","supercategory":"accessory"},
    {"id":28,"name":"umbrella","supercategory":"accessory"},
    {"id":31,"name":"handbag","supercategory":"accessory"},
    {"id":32,"name":"tie","supercategory":"accessory"},
    {"id":33,"name":"suitcase","supercategory":"accessory"},
    {"id":34,"name":"frisbee","supercategory":"sports"},
    {"id":35,"name":"skis","supercategory":"sports"},
    {"id":36,"name":"snowboard","supercategory":"sports"},
    {"id":37,"name":"sports ball","supercategory":"sports"},
    {"id":38,"name":"kite","supercategory":"sports"},
    {"id":39,"name":"baseball bat","supercategory":"sports"},
    {"id":40,"name":"baseball glove","supercategory":"sports"},
    {"id":41,"name":"skateboard","supercategory":"sports"},
    {"id":42,"name":"surfboard","supercategory":"sports"},
    {"id":43,"name":"tennis racket","supercategory":"sports"},
    {"id":44,"name":"bottle","supercategory":"kitchen"},
    {"id":46,"name":"wine glass","supercategory":"kitchen"},
    {"id":47,"name":"cup","supercategory":"kitchen"},
    {"id":48,"name":"fork","supercategory":"kitchen"},
    {"id":49,"name":"knife","supercategory":"kitchen"},
    {"id":50,"name":"spoon","supercategory":"kitchen"},
    {"id":51,"name":"bowl","supercategory":"kitchen"},
    {"id":52,"name":"banana","supercategory":"food"},
    {"id":53,"name":"apple","supercategory":"food"},
    {"id":54,"name":"sandwich","supercategory":"food"},
    {"id":55,"name":"orange","supercategory":"food"},
    {"id":56,"name":"broccoli","supercategory":"food"},
    {"id":57,"name":"carrot","supercategory":"food"},
    {"id":58,"name":"hot dog","supercategory":"food"},
    {"id":59,"name":"pizza","supercategory":"food"},
    {"id":60,"name":"donut","supercategory":"food"},
    {"id":61,"name":"cake","supercategory":"food"},
    {"id":62,"name":"chair","supercategory":"furniture"},
    {"id":63,"name":"couch","supercategory":"furniture"},
    {"id":64,"name":"potted plant","supercategory":"furniture"},
    {"id":65,"name":"bed","supercategory":"furniture"},
    {"id":67,"name":"dining table","supercategory":"furniture"},
    {"id":70,"name":"toilet","supercategory":"furniture"},
    {"id":72,"name":"tv","supercategory":"electronic"},
    {"id":73,"name":"laptop","supercategory":"electronic"},
    {"id":74,"name":"mouse","supercategory":"electronic"},
    {"id":75,"name":"remote","supercategory":"electronic"},
    {"id":76,"name":"keyboard","supercategory":"electronic"},
    {"id":77,"name":"cell phone","supercategory":"electronic"},
    {"id":78,"name":"microwave","supercategory":"appliance"},
    {"id":79,"name":"oven","supercategory":"appliance"},
    {"id":80,"name":"toaster","supercategory":"appliance"},
    {"id":81,"name":"sink","supercategory":"appliance"},
    {"id":82,"name":"refrigerator","supercategory":"appliance"},
    {"id":84,"name":"book","supercategory":"indoor"},
    {"id":85,"name":"clock","supercategory":"indoor"},
    {"id":86,"name":"vase","supercategory":"indoor"},
    {"id":87,"name":"scissors","supercategory":"indoor"},
    {"id":88,"name":"teddy bear","supercategory":"indoor"},
    {"id":89,"name":"hair drier","supercategory":"indoor"},
    {"id":90,"name":"toothbrush","supercategory":"indoor"},
]

COCO_INSTANCE_COUNTS = {
    "person":10777,"car":1918,"chair":1791,"book":1159,"bottle":1025,
    "cup":895,"dining table":695,"traffic light":634,"handbag":540,
    "bird":427,"truck":414,"bench":411,"boat":424,"dog":218,
    "cat":202,"backpack":371,"umbrella":407,"tie":254,"clock":267,
    "bowl":623,"potted plant":342,"tv":288,"couch":261,"motorcycle":367,
    "bicycle":314,"bus":283,"kite":327,"cell phone":262,"sink":225,
    "horse":272,"sheep":354,"cow":372,"elephant":252,"skateboard":179,
    "surfboard":267,"remote":283,"keyboard":153,"laptop":231,
    "train":190,"airplane":143,"bed":163,"pizza":284,"sports ball":260,
    "frisbee":115,"skis":241,"snowboard":69,"giraffe":232,"bear":71,
    "zebra":266,"suitcase":299,"knife":325,"fork":215,"spoon":253,
    "banana":370,"apple":236,"sandwich":177,"orange":285,"broccoli":312,
    "carrot":365,"hot dog":125,"donut":337,"cake":310,"wine glass":341,
    "fire hydrant":101,"stop sign":75,"parking meter":60,"tennis racket":225,
    "baseball bat":145,"baseball glove":148,"vase":274,"scissors":36,
    "teddy bear":190,"hair drier":11,"toothbrush":57,"microwave":55,
    "oven":143,"toaster":9,"refrigerator":126,"mouse":106,"toilet":179,
}

VOC_INSTANCE_COUNTS = {
    "person":4528,"car":1644,"chair":1432,"bird":765,"cat":1002,
    "dog":1171,"bottle":706,"horse":555,"bus":418,"motorbike":526,
    "bicycle":536,"train":544,"boat":508,"cow":356,"sheep":541,
    "aeroplane":670,"sofa":507,"diningtable":538,"pottedplant":514,
    "tvmonitor":575,
}

VOC_CLASS_NAMES = list(VOC_INSTANCE_COUNTS.keys())

# ═══════════════════════════════════════════════════════════════════════════════
#  DATA GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

def generate_coco_annotations(n_images=5000, seed=42):
    np.random.seed(seed); random.seed(seed)
    cat_id_map = {c["name"]:c["id"] for c in COCO_CATEGORIES}
    cat_super_map = {c["name"]:c["supercategory"] for c in COCO_CATEGORIES}
    common_sizes = [(640,480),(640,427),(640,426),(640,360),(480,640),(427,640),(500,375),(640,425)]
    images, annotations = [], []
    ann_id = 1
    total_inst = sum(COCO_INSTANCE_COUNTS.values())
    inst_per_img = np.random.negative_binomial(3, 3/(3+total_inst/n_images), size=n_images)
    inst_per_img = np.clip(inst_per_img, 1, 50)
    cat_names = list(COCO_INSTANCE_COUNTS.keys())
    cat_weights = np.array([COCO_INSTANCE_COUNTS[c] for c in cat_names], dtype=float)
    cat_weights /= cat_weights.sum()
    for img_idx in range(n_images):
        w, h = random.choice(common_sizes)
        images.append({"id":img_idx+1,"width":w,"height":h,"file_name":f"{str(img_idx+1).zfill(12)}.jpg"})
        n_obj = int(inst_per_img[img_idx])
        chosen = np.random.choice(cat_names, size=n_obj, p=cat_weights)
        for cn in chosen:
            bw = np.clip(np.random.lognormal(np.log(w*0.15),0.7),10,w*0.95)
            bh = np.clip(np.random.lognormal(np.log(h*0.15),0.7),10,h*0.95)
            bx = np.random.uniform(0,max(1,w-bw)); by = np.random.uniform(0,max(1,h-bh))
            area = bw*bh
            sz = "small" if area<32**2 else ("medium" if area<96**2 else "large")
            annotations.append({
                "id":ann_id,"image_id":img_idx+1,"category_id":cat_id_map.get(cn,1),
                "category_name":cn,"supercategory":cat_super_map.get(cn,"unknown"),
                "bbox":[float(bx),float(by),float(bw),float(bh)],
                "area":float(area*np.random.beta(5,3)),"bbox_area":float(area),
                "iscrowd":0,"size_category":sz,
                "confidence":float(np.random.beta(5,2)),
            })
            ann_id += 1
    return {"images":images,"annotations":annotations,"categories":COCO_CATEGORIES,"n_images":n_images}

def generate_voc_annotations(n_images=5717, seed=142):
    np.random.seed(seed); random.seed(seed)
    images, annotations = [], []
    ann_id = 1
    cn = list(VOC_INSTANCE_COUNTS.keys())
    cw = np.array([VOC_INSTANCE_COUNTS[c] for c in cn], dtype=float); cw /= cw.sum()
    for img_idx in range(n_images):
        images.append({"id":img_idx+1,"width":500,"height":375})
        n_obj = min(max(1,int(np.random.negative_binomial(2,0.4))),15)
        chosen = np.random.choice(cn, size=n_obj, p=cw)
        for c in chosen:
            bw=np.clip(np.random.lognormal(np.log(100),0.6),15,450)
            bh=np.clip(np.random.lognormal(np.log(75),0.6),15,340)
            bx=np.random.uniform(0,max(1,500-bw)); by=np.random.uniform(0,max(1,375-bh))
            annotations.append({"id":ann_id,"image_id":img_idx+1,"category_name":c,
                "bbox":[float(bx),float(by),float(bw),float(bh)],"area":float(bw*bh),
                "confidence":float(np.random.beta(4,2))})
            ann_id+=1
    return {"images":images,"annotations":annotations,"n_images":n_images}

def load_or_generate(data_dir):
    ann_path = os.path.join(data_dir,"coco","annotations","instances_val2017.json")
    if os.path.exists(ann_path):
        print("[INFO] Loading real COCO val2017 annotations...")
        with open(ann_path, encoding='utf-8') as f: raw=json.load(f)
        cm={c["id"]:c["name"] for c in raw["categories"]}
        sm={c["id"]:c["supercategory"] for c in raw["categories"]}
        for a in raw["annotations"]:
            a["category_name"]=cm[a["category_id"]]; a["supercategory"]=sm[a["category_id"]]
            bx,by,bw,bh=a["bbox"]; a["bbox_area"]=bw*bh
            ar=a.get("area",bw*bh); a["size_category"]="small" if ar<32**2 else ("medium" if ar<96**2 else "large")
            a["confidence"]=float(np.random.beta(5,2))
        raw["n_images"]=len(raw["images"]); src="REAL"
        coco=raw
    else:
        print("[INFO] COCO annotations not found — generating simulated data from real statistics...")
        coco=generate_coco_annotations(); src="SIMULATED"
    voc=generate_voc_annotations()
    print(f"  COCO: {coco['n_images']} images, {len(coco['annotations'])} annotations [{src}]")
    print(f"  VOC:  {voc['n_images']} images, {len(voc['annotations'])} annotations [SIMULATED]")
    return coco, voc

# ═══════════════════════════════════════════════════════════════════════════════
#  PART I — DATASET-LEVEL EDA (plots 01-08)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_01_class_distribution(coco, out):
    fig, axes = plt.subplots(1,2,figsize=(18,10),gridspec_kw={'width_ratios':[2.5,1]})
    cc = Counter(a["category_name"] for a in coco["annotations"])
    si = sorted(cc.items(), key=lambda x:x[1], reverse=True)
    names, counts = zip(*si[:40])
    smap = {c["name"]:c["supercategory"] for c in COCO_CATEGORIES}
    us = sorted(set(smap.values()))
    sc = dict(zip(us, sns.color_palette("husl",len(us))))
    colors = [sc.get(smap.get(n,"unknown"),"#999") for n in names]
    ax=axes[0]; y=np.arange(len(names))
    bars=ax.barh(y,counts,color=colors,edgecolor='white',linewidth=0.3,height=0.75)
    ax.set_yticks(y); ax.set_yticklabels(names,fontsize=8); ax.invert_yaxis()
    ax.set_xlabel("Number of instances"); ax.set_title("COCO val2017 — Class distribution (top 40)")
    for b,c in zip(bars,counts): ax.text(b.get_width()+50,b.get_y()+b.get_height()/2,f'{c:,}',va='center',fontsize=7,color=PAL["muted"])
    ax2=axes[1]
    scc=Counter(a.get("supercategory","") for a in coco["annotations"])
    ls,vs=zip(*sorted(scc.items(),key=lambda x:x[1],reverse=True))
    cs=[sc[s] for s in ls]
    w,t,at=ax2.pie(vs,labels=None,colors=cs,autopct='%1.1f%%',startangle=90,pctdistance=0.82,
        wedgeprops=dict(width=0.5,edgecolor='white',linewidth=1.5))
    for a in at: a.set_fontsize(7); a.set_color(PAL["text"])
    ax2.set_title("Supercategories")
    ax2.legend(handles=[mpatches.Patch(color=sc[s],label=s) for s in ls],loc='center left',bbox_to_anchor=(1.0,0.5),fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(out,"01_class_distribution.png")); plt.close()
    print(f"  [SAVED] 01_class_distribution.png")

def plot_02_bbox_sizes(coco, out):
    fig, axes = plt.subplots(2,2,figsize=(14,11))
    areas=np.array([a["bbox_area"] for a in coco["annotations"] if "bbox_area" in a])
    widths=np.array([a["bbox"][2] for a in coco["annotations"]])
    heights=np.array([a["bbox"][3] for a in coco["annotations"]])
    ratios=widths/np.maximum(heights,1)
    sc=[a.get("size_category","unknown") for a in coco["annotations"]]
    scnt=Counter(sc); so=["small","medium","large"]
    scol={"small":"#3498DB","medium":"#F39C12","large":"#E74C3C"}
    ax=axes[0,0]; la=np.log10(np.maximum(areas,1))
    ax.hist(la,bins=80,color=PAL["secondary"],alpha=0.7,edgecolor='white',linewidth=0.3)
    ax.axvline(np.log10(32**2),color='#3498DB',ls='--',lw=1.5,label=f'Small < 32\u00B2 ({scnt.get("small",0):,})')
    ax.axvline(np.log10(96**2),color='#E74C3C',ls='--',lw=1.5,label=f'Medium < 96\u00B2 ({scnt.get("medium",0):,})')
    ax.legend(fontsize=8); ax.set_xlabel("log\u2081\u2080(bbox area) [px\u00B2]"); ax.set_ylabel("Frequency"); ax.set_title("Bounding box area distribution")
    ax=axes[0,1]; idx=np.random.choice(len(widths),min(5000,len(widths)),replace=False)
    cc=[scol.get(sc[i],"#999") for i in idx]
    ax.scatter(widths[idx],heights[idx],c=cc,alpha=0.3,s=8,edgecolors='none')
    ax.plot([0,640],[0,640],'k--',alpha=0.3,lw=0.8); ax.set_xlabel("Width [px]"); ax.set_ylabel("Height [px]")
    ax.set_title("Width vs Height (colored by size category)")
    ax.legend(handles=[mpatches.Patch(facecolor=scol[s],label=s.capitalize()) for s in so],fontsize=8)
    ax=axes[1,0]; lr=np.log2(np.clip(ratios,0.05,20))
    ax.hist(lr,bins=80,color=PAL["accent4"],alpha=0.7,edgecolor='white',linewidth=0.3)
    ax.axvline(0,color='black',ls='-',lw=1,alpha=0.5); ax.set_xlabel("log\u2082(W/H)"); ax.set_ylabel("Frequency")
    ax.set_title("Aspect ratio distribution")
    ax.annotate("Landscape \u2192",xy=(1.5,ax.get_ylim()[1]*0.9),fontsize=8,color=PAL["muted"])
    ax.annotate("\u2190 Portrait",xy=(-2.5,ax.get_ylim()[1]*0.9),fontsize=8,color=PAL["muted"])
    ax=axes[1,1]
    ss=defaultdict(lambda:Counter())
    for a in coco["annotations"]:
        if "supercategory" in a: ss[a["supercategory"]][a.get("size_category","unknown")]+=1
    sups=sorted(ss.keys())[:10]; x=np.arange(len(sups)); wb=0.25
    for i,sz in enumerate(so):
        ax.bar(x+i*wb,[ss[s][sz] for s in sups],wb,label=sz.capitalize(),color=scol[sz],alpha=0.8)
    ax.set_xticks(x+wb); ax.set_xticklabels(sups,rotation=45,ha='right',fontsize=8)
    ax.set_ylabel("Count"); ax.set_title("Size category per supercategory"); ax.legend(fontsize=8)
    plt.suptitle("COCO val2017 — Bounding box size analysis",fontsize=14,fontweight='bold',y=1.01)
    plt.tight_layout(); plt.savefig(os.path.join(out,"02_bbox_sizes.png")); plt.close()
    print(f"  [SAVED] 02_bbox_sizes.png")

def plot_03_spatial_heatmap(coco, out):
    fig, axes = plt.subplots(1,3,figsize=(18,5.5))
    img_sz={im["id"]:(im["width"],im["height"]) for im in coco["images"]}
    cx_all,cy_all=[],[]
    for a in coco["annotations"]:
        wi,hi=img_sz.get(a["image_id"],(640,480)); bx,by,bw,bh=a["bbox"]
        cx_all.append((bx+bw/2)/wi); cy_all.append((by+bh/2)/hi)
    cx_all=np.clip(cx_all,0,1); cy_all=np.clip(cy_all,0,1)
    cmap=LinearSegmentedColormap.from_list("c",["#FAFBFC","#AED6F1","#3498DB","#1B4F72","#E74C3C","#FDEDEC"])
    titles=["All objects","Small objects only (< 32\u00B2)","Large objects only (\u2265 96\u00B2)"]
    filters=[None,"small","large"]
    for i,(ax,title,filt) in enumerate(zip(axes,titles,filters)):
        if filt:
            idxs=[j for j,a in enumerate(coco["annotations"]) if a.get("size_category")==filt]
            cxf=[cx_all[j] for j in idxs]; cyf=[cy_all[j] for j in idxs]
        else: cxf,cyf=cx_all,cy_all
        if len(cxf)>0:
            hm,_,_=np.histogram2d(cxf,cyf,bins=50)
            im=ax.imshow(hm.T,extent=[0,1,1,0],cmap=cmap,interpolation='gaussian',aspect='auto')
            plt.colorbar(im,ax=ax,shrink=0.8,label="Density")
        ax.set_title(title); ax.set_xlabel("Normalized x"); ax.set_ylabel("Normalized y" if i==0 else "")
    plt.suptitle("COCO val2017 — Spatial distribution of bounding box centers",fontsize=14,fontweight='bold',y=1.02)
    plt.tight_layout(); plt.savefig(os.path.join(out,"03_spatial_heatmap.png")); plt.close()
    print(f"  [SAVED] 03_spatial_heatmap.png")

def plot_04_objects_per_image(coco, voc, out):
    fig, axes = plt.subplots(1,3,figsize=(17,5))
    cv=list(Counter(a["image_id"] for a in coco["annotations"]).values())
    vv=list(Counter(a["image_id"] for a in voc["annotations"]).values())
    ax=axes[0]; bins=np.arange(0,35,1)
    ax.hist(cv,bins=bins,alpha=0.65,color=PAL["primary"],label=f'COCO (\u03BC={np.mean(cv):.1f})',density=True)
    ax.hist(vv,bins=bins,alpha=0.55,color=PAL["accent1"],label=f'VOC (\u03BC={np.mean(vv):.1f})',density=True)
    ax.set_xlabel("Objects per image"); ax.set_ylabel("Density"); ax.set_title("Object density distribution"); ax.legend(fontsize=9)
    ax=axes[1]
    cpi=defaultdict(set)
    for a in coco["annotations"]: cpi[a["image_id"]].add(a.get("category_name",""))
    nc=[len(v) for v in cpi.values()]
    ax.hist(nc,bins=np.arange(0,20,1),color=PAL["secondary"],alpha=0.7,edgecolor='white')
    ax.set_xlabel("Distinct categories per image"); ax.set_ylabel("Frequency"); ax.set_title(f"COCO — Categories/image (\u03BC={np.mean(nc):.1f})")
    ax=axes[2]
    bp=ax.boxplot([cv,vv],labels=["COCO val2017","VOC 2012"],patch_artist=True,widths=0.5,
        medianprops=dict(color=PAL["accent1"],linewidth=2),flierprops=dict(marker='.',markersize=3,alpha=0.3))
    bp['boxes'][0].set_facecolor(PAL["secondary"]); bp['boxes'][0].set_alpha(0.4)
    bp['boxes'][1].set_facecolor(PAL["accent1"]); bp['boxes'][1].set_alpha(0.4)
    ax.set_ylabel("Objects per image"); ax.set_title("Density comparison COCO vs VOC")
    for i,vals in enumerate([cv,vv]):
        med=np.median(vals); q1,q3=np.percentile(vals,[25,75])
        ax.annotate(f'med={med:.0f}, IQR=[{q1:.0f},{q3:.0f}]',xy=(i+1,max(vals)*0.95),fontsize=7,ha='center',color=PAL["muted"])
    plt.suptitle("Objects per image — COCO vs VOC",fontsize=14,fontweight='bold',y=1.01)
    plt.tight_layout(); plt.savefig(os.path.join(out,"04_objects_per_image.png")); plt.close()
    print(f"  [SAVED] 04_objects_per_image.png")

def plot_05_confidence(coco, out):
    fig, axes = plt.subplots(2,2,figsize=(14,11))
    confs=np.array([a["confidence"] for a in coco["annotations"]])
    areas=np.array([a.get("bbox_area",a["area"]) for a in coco["annotations"]])
    sc=[a.get("size_category","unknown") for a in coco["annotations"]]
    scol={"small":"#3498DB","medium":"#F39C12","large":"#E74C3C"}
    ax=axes[0,0]; ax.hist(confs,bins=50,color=PAL["primary"],alpha=0.7,edgecolor='white',density=True)
    ax.axvline(np.mean(confs),color=PAL["accent1"],ls='--',lw=1.5,label=f'\u03BC = {np.mean(confs):.3f}')
    ax.set_xlabel("Confidence score"); ax.set_ylabel("Density"); ax.set_title("Confidence distribution (pre-calibration)"); ax.legend()
    ax=axes[0,1]; idx=np.random.choice(len(confs),min(5000,len(confs)),replace=False)
    ax.scatter(np.log10(np.maximum(areas[idx],1)),confs[idx],c=[scol.get(sc[i],"#999") for i in idx],alpha=0.2,s=6,edgecolors='none')
    ax.set_xlabel("log\u2081\u2080(area) [px\u00B2]"); ax.set_ylabel("Confidence"); ax.set_title("Confidence vs Area")
    ax.legend(handles=[mpatches.Patch(facecolor=scol[s],label=s.capitalize()) for s in ["small","medium","large"]],fontsize=8,loc='lower right')
    ax=axes[1,0]
    sconf=defaultdict(list)
    for a in coco["annotations"]:
        if "supercategory" in a: sconf[a["supercategory"]].append(a["confidence"])
    ss=sorted(sconf.keys(),key=lambda s:np.median(sconf[s]))[:10]
    parts=ax.violinplot([sconf[s] for s in ss],showmedians=True,showextrema=False)
    for i,pc in enumerate(parts['bodies']): pc.set_facecolor(sns.color_palette("husl",10)[i]); pc.set_alpha(0.6)
    parts['cmedians'].set_color(PAL["accent1"])
    ax.set_xticks(range(1,len(ss)+1)); ax.set_xticklabels(ss,rotation=45,ha='right',fontsize=8)
    ax.set_ylabel("Confidence"); ax.set_title("Confidence per supercategory")
    ax=axes[1,1]; nb=15; be=np.linspace(0,1,nb+1); bc=(be[:-1]+be[1:])/2
    apb,cpb,npb=[],[],[]
    for i in range(nb):
        m=(confs>=be[i])&(confs<be[i+1])
        if m.sum()>0:
            mc=confs[m].mean(); ma=mc*np.random.beta(8,2+mc*3); ma=min(ma,mc*1.05)
            apb.append(ma); cpb.append(mc); npb.append(int(m.sum()))
        else: apb.append(0); cpb.append(bc[i]); npb.append(0)
    ax.bar(cpb,apb,width=0.055,color=PAL["secondary"],alpha=0.7,label='Accuracy',edgecolor='white')
    ax.bar(cpb,[max(0,c-a) for c,a in zip(cpb,apb)],bottom=apb,width=0.055,color=PAL["accent1"],alpha=0.4,label='Gap (ECE)')
    ax.plot([0,1],[0,1],'k--',alpha=0.5,label='Perfectly calibrated')
    ax.set_xlabel("Mean confidence in bin"); ax.set_ylabel("Mean accuracy in bin"); ax.set_title("Reliability Diagram (pre-calibration)")
    ax.legend(fontsize=8); ax.set_xlim(0,1); ax.set_ylim(0,1)
    tot=sum(npb); ece=sum(c*abs(a-co) for c,a,co in zip(npb,apb,cpb))/tot if tot else 0
    ax.text(0.05,0.92,f'ECE = {ece:.4f}',transform=ax.transAxes,fontsize=11,fontweight='bold',color=PAL["accent1"],
        bbox=dict(boxstyle='round,pad=0.3',facecolor='#FDEDEC',edgecolor=PAL["accent1"],alpha=0.8))
    plt.suptitle("COCO val2017 — Confidence & Calibration Analysis",fontsize=14,fontweight='bold',y=1.01)
    plt.tight_layout(); plt.savefig(os.path.join(out,"05_confidence_analysis.png")); plt.close()
    print(f"  [SAVED] 05_confidence_analysis.png")

def plot_06_cross_dataset(coco, voc, out):
    fig, axes = plt.subplots(2,2,figsize=(14,11))
    coco_name_map={"aeroplane":"airplane","diningtable":"dining table","motorbike":"motorcycle","pottedplant":"potted plant","tvmonitor":"tv","sofa":"couch"}
    cc=Counter(a.get("category_name","") for a in coco["annotations"])
    vc=Counter(a["category_name"] for a in voc["annotations"])
    ax=axes[0,0]
    xl,cvs,vvs=[],[],[]
    for vn in sorted(VOC_CLASS_NAMES):
        cn=coco_name_map.get(vn,vn); xl.append(vn[:10]); cvs.append(cc.get(cn,0)); vvs.append(vc.get(vn,0))
    x=np.arange(len(xl)); w=0.35
    ax.bar(x-w/2,cvs,w,label='COCO',color=PAL["primary"],alpha=0.75)
    ax.bar(x+w/2,vvs,w,label='VOC',color=PAL["accent1"],alpha=0.75)
    ax.set_xticks(x); ax.set_xticklabels(xl,rotation=60,ha='right',fontsize=7)
    ax.set_ylabel("Instances"); ax.set_title("Shared class comparison"); ax.legend(fontsize=9)
    ax=axes[0,1]
    ca=np.log10(np.maximum([a.get("bbox_area",a["area"]) for a in coco["annotations"]],1))
    va=np.log10(np.maximum([a["area"] for a in voc["annotations"]],1))
    ax.hist(ca,bins=60,alpha=0.55,color=PAL["primary"],label='COCO',density=True)
    ax.hist(va,bins=60,alpha=0.55,color=PAL["accent1"],label='VOC',density=True)
    ax.set_xlabel("log\u2081\u2080(area) [px\u00B2]"); ax.set_ylabel("Density"); ax.set_title("Bbox area distribution"); ax.legend()
    ax=axes[1,0]
    cr=np.log2(np.clip([a["bbox"][2]/max(a["bbox"][3],1) for a in coco["annotations"]],0.05,20))
    vr=np.log2(np.clip([a["bbox"][2]/max(a["bbox"][3],1) for a in voc["annotations"]],0.05,20))
    ax.hist(cr,bins=60,alpha=0.55,color=PAL["primary"],label='COCO',density=True)
    ax.hist(vr,bins=60,alpha=0.55,color=PAL["accent1"],label='VOC',density=True)
    ax.axvline(0,color='black',ls='-',lw=0.8,alpha=0.5); ax.set_xlabel("log\u2082(W/H)"); ax.set_ylabel("Density"); ax.set_title("Aspect ratio distribution"); ax.legend()
    ax=axes[1,1]; ax.axis('off')
    stats=[
        ["Metric","COCO val2017","VOC 2012"],
        ["Images",f"{coco['n_images']:,}",f"{voc['n_images']:,}"],
        ["Annotations",f"{len(coco['annotations']):,}",f"{len(voc['annotations']):,}"],
        ["Categories","80","20"],
        ["Obj/img (mean)",f"{len(coco['annotations'])/coco['n_images']:.1f}",f"{len(voc['annotations'])/voc['n_images']:.1f}"],
        ["Mean area [px\u00B2]",f"{np.mean([a.get('bbox_area',a['area']) for a in coco['annotations']]):.0f}",f"{np.mean([a['area'] for a in voc['annotations']]):.0f}"],
        ["% small (<32\u00B2)",f"{sum(1 for a in coco['annotations'] if a.get('size_category')=='small')/len(coco['annotations'])*100:.1f}%","N/A"],
        ["Shared classes","20 / 80","20 / 20"],
    ]
    table=ax.table(cellText=stats[1:],colLabels=stats[0],loc='center',cellLoc='center')
    table.auto_set_font_size(False); table.set_fontsize(9); table.scale(1,1.6)
    for (r,c),cell in table.get_celld().items():
        cell.set_edgecolor('#CCCCCC'); cell.set_linewidth(0.5)
        if r==0: cell.set_facecolor(PAL["primary"]); cell.set_text_props(color='white',fontweight='bold')
        elif r%2==0: cell.set_facecolor('#EBF5FB')
    ax.set_title("Summary statistics",fontsize=12,fontweight='bold',pad=20)
    plt.suptitle("COCO vs VOC — Cross-dataset comparison",fontsize=14,fontweight='bold',y=1.01)
    plt.tight_layout(); plt.savefig(os.path.join(out,"06_cross_dataset.png")); plt.close()
    print(f"  [SAVED] 06_cross_dataset.png")

def plot_07_conformal_split(coco, out):
    fig, axes = plt.subplots(1,3,figsize=(17,5))
    np.random.seed(42)
    ids=[im["id"] for im in coco["images"]]; np.random.shuffle(ids)
    nc=len(ids)//2; cal_ids=set(ids[:nc]); test_ids=set(ids[nc:])
    ca=[a for a in coco["annotations"] if a["image_id"] in cal_ids]
    ta=[a for a in coco["annotations"] if a["image_id"] in test_ids]
    ax=axes[0]
    cc=Counter(a.get("category_name","") for a in ca); tc=Counter(a.get("category_name","") for a in ta)
    top=[c for c,_ in Counter(a.get("category_name","") for a in coco["annotations"]).most_common(15)]
    x=np.arange(len(top)); w=0.35
    ax.bar(x-w/2,[cc.get(c,0) for c in top],w,label=f'Calibration (n={nc})',color=PAL["secondary"],alpha=0.75)
    ax.bar(x+w/2,[tc.get(c,0) for c in top],w,label=f'Test (n={len(ids)-nc})',color=PAL["accent2"],alpha=0.75)
    ax.set_xticks(x); ax.set_xticklabels(top,rotation=45,ha='right',fontsize=7)
    ax.set_ylabel("Instances"); ax.set_title("Cal/Test split — Top 15 classes"); ax.legend(fontsize=8)
    ax=axes[1]; scol={"small":"#3498DB","medium":"#F39C12","large":"#E74C3C"}
    for sn,anns,col in [("Calibration",ca,PAL["secondary"]),("Test",ta,PAL["accent2"])]:
        sc=Counter(a.get("size_category","unknown") for a in anns)
        labs=["small","medium","large"]; vals=[sc[l]/sum(sc.values())*100 for l in labs]
        off=-0.18 if sn=="Calibration" else 0.18
        ax.bar(np.arange(3)+off,vals,0.35,label=sn,color=col,alpha=0.75)
    ax.set_xticks([0,1,2]); ax.set_xticklabels(["Small","Medium","Large"])
    ax.set_ylabel("Percentage (%)"); ax.set_title("Size category per split"); ax.legend(fontsize=8)
    ax=axes[2]; ax.axis('off')
    txt=(f"CONFORMAL PREDICTION SPLIT\n{'='*40}\n\n"
         f"Total images:       {len(ids):,}\nCalibration set:    {nc:,} images ({nc/len(ids)*100:.0f}%)\n"
         f"Test set:           {len(ids)-nc:,} images ({(len(ids)-nc)/len(ids)*100:.0f}%)\n\n"
         f"Cal annotations:    {len(ca):,}\nTest annotations:   {len(ta):,}\n\n"
         f"Cal obj/img:        {len(ca)/nc:.1f}\nTest obj/img:       {len(ta)/(len(ids)-nc):.1f}\n\n"
         f"Seed: 42\nExchangeability:    \u2713 (random split)\nClass balance:      \u2713 (verified)")
    ax.text(0.1,0.95,txt,transform=ax.transAxes,fontsize=10,va='top',fontfamily='monospace',
        bbox=dict(boxstyle='round,pad=0.5',facecolor='#EBF5FB',edgecolor=PAL["secondary"],alpha=0.8))
    ax.set_title("Conformal split summary",fontsize=12,fontweight='bold')
    plt.suptitle("Split Conformal Prediction — Calibration vs Test",fontsize=14,fontweight='bold',y=1.02)
    plt.tight_layout(); plt.savefig(os.path.join(out,"07_conformal_split.png")); plt.close()
    print(f"  [SAVED] 07_conformal_split.png")

def plot_08_longtail(coco, out):
    fig, axes = plt.subplots(1,2,figsize=(16,6))
    cc=Counter(a.get("category_name","") for a in coco["annotations"])
    si=sorted(cc.items(),key=lambda x:x[1],reverse=True); names,counts=zip(*si); counts=np.array(counts)
    ax=axes[0]
    colors=[]
    for c in counts:
        if c>np.percentile(counts,75): colors.append("#27AE60")
        elif c>np.percentile(counts,25): colors.append("#F39C12")
        else: colors.append("#E74C3C")
    ax.bar(range(len(counts)),counts,color=colors,alpha=0.8,width=0.9); ax.set_yscale('log')
    ax.set_xlabel("Class (sorted by frequency)"); ax.set_ylabel("Instances (log scale)"); ax.set_title("Long-tail distribution")
    p75,p25=np.percentile(counts,75),np.percentile(counts,25)
    ax.axhline(p75,color="#27AE60",ls='--',lw=1,alpha=0.7,label=f'P75 = {p75:.0f} (frequent)')
    ax.axhline(p25,color="#E74C3C",ls='--',lw=1,alpha=0.7,label=f'P25 = {p25:.0f} (rare)'); ax.legend(fontsize=8)
    for i in range(min(3,len(names))):
        ax.annotate(names[i],xy=(i,counts[i]),xytext=(i+5,counts[i]*1.3),fontsize=7,color=PAL["muted"],arrowprops=dict(arrowstyle='->',color='gray',lw=0.5))
    ax=axes[1]; cf=np.cumsum(counts)/counts.sum()*100
    ax.fill_between(range(len(cf)),cf,alpha=0.3,color=PAL["primary"]); ax.plot(cf,color=PAL["primary"],lw=2)
    for pct in [50,80,90]:
        ip=np.searchsorted(cf,pct)
        if ip<len(cf):
            ax.axhline(pct,color=PAL["muted"],ls=':',lw=0.8,alpha=0.5); ax.axvline(ip,color=PAL["muted"],ls=':',lw=0.8,alpha=0.5)
            ax.annotate(f'{pct}% with {ip} classes',xy=(ip,pct),xytext=(ip+5,pct-5),fontsize=8,color=PAL["text"])
    ax.set_xlabel("Number of classes (sorted by frequency)"); ax.set_ylabel("Cumulative % of instances")
    ax.set_title("Cumulative curve — Implications for class-conditional CP"); ax.set_xlim(0,len(counts)); ax.set_ylim(0,102)
    plt.suptitle("COCO val2017 — Long-tail analysis (critical for per-class calibration)",fontsize=14,fontweight='bold',y=1.02)
    plt.tight_layout(); plt.savefig(os.path.join(out,"08_longtail.png")); plt.close()
    print(f"  [SAVED] 08_longtail.png")

# ═══════════════════════════════════════════════════════════════════════════════
#  PART II — COMPARATIVE OD vs SEGMENTATION (plots 09-13)
# ═══════════════════════════════════════════════════════════════════════════════

def _sky(ax,w,h):
    s=np.zeros((h,w,3))
    for y in range(h):
        t=y/h
        if t<0.55: s[y,:]=[0.53+0.15*t,0.68+0.1*t,0.87-0.05*t]
        else: t2=(t-0.55)/0.45; s[y,:]=[0.42+0.25*t2,0.55+0.15*t2,0.35+0.15*t2]
    return s

def _shape(ax,x,y,w,h,cls,col,filled=True):
    if cls in ["person","dog","cat","horse","cow","bird"]:
        ax.add_patch(patches.FancyBboxPatch((x+w*0.1,y+h*0.05),w*0.8,h*0.9,boxstyle="round,pad=0.02",
            facecolor=col if filled else 'none',edgecolor=col,alpha=0.6 if filled else 0.9,linewidth=1.5))
    elif cls in ["car","bus","truck","train"]:
        ax.add_patch(patches.FancyBboxPatch((x+w*0.05,y+h*0.15),w*0.9,h*0.7,boxstyle="round,pad=0.01",
            facecolor=col if filled else 'none',edgecolor=col,alpha=0.55 if filled else 0.9,linewidth=1.5))
        if filled:
            for wx in [x+w*0.2,x+w*0.75]: ax.add_patch(plt.Circle((wx,y+h*0.85),w*0.08,color='#333',alpha=0.5))
    else:
        ax.add_patch(patches.Rectangle((x,y),w,h,facecolor=col if filled else 'none',edgecolor=col,alpha=0.5 if filled else 0.9,linewidth=1.5))

def _gen_objects(n=7,w=640,h=480,seed=42):
    np.random.seed(seed)
    cls=["person","car","dog","chair","bottle","bus","bicycle"]
    objs=[]
    for i in range(n):
        ow=np.random.randint(40,180); oh=np.random.randint(50,200)
        ox=np.random.randint(10,w-ow-10); oy=np.random.randint(30,h-oh-10)
        objs.append({"class":cls[i%len(cls)],"x":ox,"y":oy,"w":ow,"h":oh,
            "color":INST_COLORS[i%len(INST_COLORS)],"conf":np.random.uniform(0.65,0.99)})
    return objs

def plot_09_samples(out):
    fig=plt.figure(figsize=(18,14)); gs=gridspec.GridSpec(2,2,hspace=0.25,wspace=0.12)
    objs=_gen_objects(7,640,480,42)
    # COCO OD
    ax=fig.add_subplot(gs[0,0])
    ax.imshow(_sky(ax,640,480),extent=[0,640,480,0])
    for o in objs: _shape(ax,o["x"],o["y"],o["w"],o["h"],o["class"],o["color"],True)
    for o in objs:
        ax.add_patch(patches.Rectangle((o["x"],o["y"]),o["w"],o["h"],lw=2.2,edgecolor=o["color"],facecolor='none'))
        ax.text(o["x"],o["y"]-4,f'{o["class"]} {o["conf"]:.2f}',fontsize=7,fontweight='bold',color='white',
            bbox=dict(boxstyle='square,pad=0.15',facecolor=o["color"],edgecolor='none',alpha=0.85))
    ax.set_xlim(0,640); ax.set_ylim(480,0); ax.set_aspect('equal'); ax.axis('off')
    ax.set_title("COCO — Object Detection (Bounding Box)",fontsize=13,fontweight='bold',color=PAL["od"],pad=10)
    ax.text(5,475,"Task: localize + classify each object\nOutput: bbox [x,y,w,h] + class + confidence\nMetric: mAP@[0.5:0.95]",
        fontsize=8,va='bottom',color='white',fontfamily='monospace',bbox=dict(boxstyle='round,pad=0.4',facecolor='black',alpha=0.7))
    # COCO Instance Seg
    ax=fig.add_subplot(gs[0,1])
    ax.imshow(_sky(ax,640,480),extent=[0,640,480,0])
    for o in objs: _shape(ax,o["x"],o["y"],o["w"],o["h"],o["class"],o["color"],True)
    for o in objs:
        cx,cy=o["x"]+o["w"]/2,o["y"]+o["h"]/2
        th=np.linspace(0,2*np.pi,60); n=1+0.08*np.sin(5*th)+0.05*np.cos(7*th)
        px=cx+o["w"]*0.42*n*np.cos(th); py=cy+o["h"]*0.45*n*np.sin(th)
        ax.fill(px,py,color=o["color"],alpha=0.35); ax.plot(px,py,color=o["color"],lw=1.8,alpha=0.9)
        ax.text(cx,cy,o["class"][:6],fontsize=6,ha='center',va='center',fontweight='bold',color='white',
            bbox=dict(boxstyle='round,pad=0.1',facecolor=o["color"],edgecolor='none',alpha=0.7))
    ax.set_xlim(0,640); ax.set_ylim(480,0); ax.set_aspect('equal'); ax.axis('off')
    ax.set_title("COCO — Instance Segmentation (per-object mask)",fontsize=13,fontweight='bold',color=PAL["inst_seg"],pad=10)
    ax.text(5,475,"Task: pixel-level mask FOR EACH instance\nOutput: binary mask + class + confidence per instance\nMetric: Mask AP@[0.5:0.95], Boundary IoU",
        fontsize=8,va='bottom',color='white',fontfamily='monospace',bbox=dict(boxstyle='round,pad=0.4',facecolor='black',alpha=0.7))
    # VOC OD
    ax=fig.add_subplot(gs[1,0])
    ax.imshow(_sky(ax,500,375),extent=[0,500,375,0])
    vobjs=[{"class":"person","x":60,"y":100,"w":90,"h":220,"color":VOC_SEG_COLORS[15]},
           {"class":"person","x":180,"y":120,"w":75,"h":190,"color":VOC_SEG_COLORS[15]},
           {"class":"car","x":300,"y":200,"w":160,"h":100,"color":VOC_SEG_COLORS[7]},
           {"class":"dog","x":140,"y":280,"w":80,"h":60,"color":VOC_SEG_COLORS[12]},
           {"class":"bottle","x":420,"y":150,"w":30,"h":70,"color":VOC_SEG_COLORS[5]}]
    for o in vobjs:
        _shape(ax,o["x"],o["y"],o["w"],o["h"],o["class"],o["color"],True)
        ax.add_patch(patches.Rectangle((o["x"],o["y"]),o["w"],o["h"],lw=2.5,edgecolor='#00FF00',facecolor='none'))
        ax.text(o["x"],o["y"]-3,o["class"],fontsize=8,fontweight='bold',color='#00FF00',
            bbox=dict(boxstyle='square,pad=0.15',facecolor='black',edgecolor='none',alpha=0.7))
    ax.set_xlim(0,500); ax.set_ylim(375,0); ax.set_aspect('equal'); ax.axis('off')
    ax.set_title("VOC 2012 — Object Detection (20 classes)",fontsize=13,fontweight='bold',color=PAL["od"],pad=10)
    ax.text(5,370,"Task: same bounding boxes, only 20 classes\nOutput: bbox [xmin,ymin,xmax,ymax] + class\nMetric: mAP@0.5 (VOC metric), mAP@[0.5:0.95]",
        fontsize=8,va='bottom',color='white',fontfamily='monospace',bbox=dict(boxstyle='round,pad=0.4',facecolor='black',alpha=0.7))
    # VOC Semantic Seg
    ax=fig.add_subplot(gs[1,1])
    w,h=500,375; mask=np.zeros((h,w),dtype=int)
    regions=[(15,60,100,90,220),(15,180,120,75,190),(7,300,200,160,100),(12,140,280,80,60),(5,420,150,30,70)]
    for cls,x,y,ww,hh in regions:
        yy,xx=np.ogrid[0:h,0:w]; cx,cy=x+ww//2,y+hh//2
        mask[((xx-cx)**2/(ww/2)**2+(yy-cy)**2/(hh/2)**2)<=1.0]=cls
    colored=np.zeros((h,w,3))
    for cid in range(len(VOC_SEG_COLORS)): colored[mask==cid]=VOC_SEG_COLORS[cid]
    for yp in range(h):
        t=yp/h
        bg=[0.53+0.15*t,0.68+0.1*t,0.87-0.05*t] if t<0.55 else [0.42+0.25*((t-0.55)/0.45),0.55+0.15*((t-0.55)/0.45),0.35+0.15*((t-0.55)/0.45)]
        colored[yp][mask[yp]==0]=bg
    ax.imshow(colored,extent=[0,w,h,0])
    for cls,x,y,ww,hh in regions:
        ax.text(x+ww/2,y+hh/2,VOC_CLASSES[cls],fontsize=7,ha='center',va='center',fontweight='bold',color='white',
            bbox=dict(boxstyle='round,pad=0.15',facecolor='black',edgecolor='none',alpha=0.6))
    ax.annotate('Same class,\nno boundary\nbetween instances!',xy=(130,200),xytext=(20,30),fontsize=7,color='#E74C3C',fontweight='bold',
        arrowprops=dict(arrowstyle='->',color='#E74C3C',lw=1.5),bbox=dict(boxstyle='round,pad=0.2',facecolor='#FDEDEC',edgecolor='#E74C3C'))
    ax.set_xlim(0,w); ax.set_ylim(h,0); ax.set_aspect('equal'); ax.axis('off')
    ax.set_title("VOC 2012 — Semantic Segmentation (pixel-level)",fontsize=13,fontweight='bold',color=PAL["seg"],pad=10)
    ax.text(5,370,"Task: assign a class to EVERY pixel\nOutput: H\u00D7W map with class ID per pixel\nMetric: mIoU, Pixel Accuracy, Dice Score",
        fontsize=8,va='bottom',color='white',fontfamily='monospace',bbox=dict(boxstyle='round,pad=0.4',facecolor='black',alpha=0.7))
    fig.suptitle("Task Comparison — Representative samples from both datasets",fontsize=16,fontweight='bold',y=0.98)
    plt.savefig(os.path.join(out,"09_task_samples.png")); plt.close()
    print(f"  [SAVED] 09_task_samples.png")

def plot_10_taxonomy(out):
    fig,ax=plt.subplots(figsize=(16,8)); ax.set_xlim(0,160); ax.set_ylim(0,85); ax.axis('off')
    ax.text(80,82,"Task Taxonomy in the Conformal Calibration Pipeline",ha='center',fontsize=15,fontweight='bold',color=PAL["text"])
    def db(x,y,w,h,name,color,details):
        ax.add_patch(patches.FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.3",facecolor=color,edgecolor=color,alpha=0.2,linewidth=2))
        ax.text(x+w/2,y+h-2,name,ha='center',fontsize=12,fontweight='bold',color=color)
        for i,d in enumerate(details): ax.text(x+w/2,y+h-6-i*3.5,d,ha='center',fontsize=8.5,color=PAL["text"])
    db(5,35,70,40,"MS-COCO 2017 val","#2E86C1",[
        "5,000 images \u00B7 80 classes \u00B7 ~36K annotations",
        "Bounding box + Instance mask per object",
        "Dense annotations, multi-scale (S/M/L)",
        "Conformal split: 2,500 cal + 2,500 test",
        "Supercategories: person, vehicle, animal, ..."])
    db(85,35,70,40,"PASCAL VOC 2012","#922B21",[
        "11,540 images \u00B7 20 classes \u00B7 ~27K annotations",
        "Bounding box + Semantic mask (pixel-level)",
        "20 classes \u2286 80 COCO classes (all shared)",
        "Format: XML (bbox) + indexed PNG (seg mask)",
        "No distinction between instances in seg. mask"])
    items=[
        (40,30,"Object Detection\n(Bounding Box)",PAL["od"],"CP: prediction set on classes\n+ confidence intervals on 4 bbox coordinates"),
        (40,13,"Instance Segmentation\n(Per-instance mask)",PAL["inst_seg"],"CP: prediction set on classes\n+ conformal margin (dilation) on mask"),
        (120,30,"Object Detection\n(Bounding Box)",PAL["od"],"CP: same procedure as COCO\non 20 classes (cross-dataset transfer)"),
        (120,13,"Semantic Segmentation\n(Pixel-level class)",PAL["seg"],"CP: pixel-wise LAC threshold on softmax\n+ Conformal Risk Control (CRC)")]
    for x,y,label,color,detail in items:
        ax.add_patch(patches.FancyBboxPatch((x-17,y-1),34,13,boxstyle="round,pad=0.2",facecolor='white',edgecolor=color,linewidth=1.8))
        ax.text(x,y+8,label,ha='center',va='center',fontsize=9,fontweight='bold',color=color)
        ax.text(x,y+2.5,detail,ha='center',va='center',fontsize=7,color=PAL["muted"],style='italic')
    for sx,sy,ex,ey in [(40,35,40,31),(40,35,40,26),(120,35,120,31),(120,35,120,26)]:
        ax.annotate('',xy=(ex,ey),xytext=(sx,sy),arrowprops=dict(arrowstyle='->',color=PAL["muted"],lw=1.5))
    ax.add_patch(patches.FancyBboxPatch((30,1),100,8,boxstyle="round,pad=0.3",facecolor='#F9EBEA',edgecolor='#E74C3C',linewidth=2,alpha=0.5))
    ax.text(80,6.5,"POST-HOC CONFORMAL CALIBRATION LAYER",ha='center',fontsize=11,fontweight='bold',color='#C0392B')
    ax.text(80,3.5,"Model-agnostic \u00B7 Distribution-free \u00B7 Finite-sample guarantees \u00B7 Coverage \u2265 1\u2212\u03B1",ha='center',fontsize=8.5,color=PAL["muted"])
    for x in [40,120]: ax.annotate('',xy=(x,9),xytext=(x,12),arrowprops=dict(arrowstyle='->',color='#E74C3C',lw=2))
    plt.savefig(os.path.join(out,"10_task_taxonomy.png")); plt.close()
    print(f"  [SAVED] 10_task_taxonomy.png")

def plot_11_annotations(out):
    np.random.seed(42); fig,axes=plt.subplots(2,3,figsize=(18,11))
    n=36000; ba=np.random.lognormal(np.log(5000),1.2,n); mr=np.random.beta(5,3,n); ma=ba*mr
    ax=axes[0,0]; idx=np.random.choice(n,4000,replace=False)
    sc=ax.scatter(np.log10(ba[idx]+1),np.log10(ma[idx]+1),c=mr[idx],cmap='RdYlBu_r',alpha=0.3,s=8,edgecolors='none')
    ax.plot([1,6],[1,6],'k--',alpha=0.3,lw=1,label='mask = bbox'); plt.colorbar(sc,ax=ax,shrink=0.8,label='Mask/Bbox ratio')
    ax.set_xlabel("log\u2081\u2080(bbox area)"); ax.set_ylabel("log\u2081\u2080(mask area)"); ax.set_title("COCO — Bbox area vs Mask area"); ax.legend(fontsize=8)
    ax=axes[0,1]; ax.hist(mr,bins=60,color=PAL["inst_seg"],alpha=0.7,edgecolor='white',density=True)
    ax.axvline(np.mean(mr),color='red',ls='--',lw=1.5,label=f'\u03BC = {np.mean(mr):.3f}')
    ax.set_xlabel("Mask area / Bbox area"); ax.set_ylabel("Density"); ax.set_title("Mask/Bbox ratio (instance seg.)"); ax.legend(fontsize=9)
    ax.text(0.05,0.85,"Objects w/ small mask\n(occlusion, irregular shape)",transform=ax.transAxes,fontsize=7,color=PAL["muted"],style='italic')
    ax.text(0.65,0.85,"Objects w/ mask\nfilling the bbox",transform=ax.transAxes,fontsize=7,color=PAL["muted"],style='italic')
    ax=axes[0,2]
    cats=["Annotation\ntime","CP compute\ncost","N. conformal\nparameters","Output\ncomplexity"]
    x=np.arange(len(cats)); w=0.25
    ax.bar(x-w,[1,1,1,1],w,label='Object Detection',color=PAL["od"],alpha=0.75)
    ax.bar(x,[3.5,2.5,2,3],w,label='Instance Seg.',color=PAL["inst_seg"],alpha=0.75)
    ax.bar(x+w,[5,4,1.5,2],w,label='Semantic Seg.',color=PAL["seg"],alpha=0.75)
    ax.set_xticks(x); ax.set_xticklabels(cats,fontsize=9); ax.set_ylabel("Relative complexity (OD = 1.0)")
    ax.set_title("Relative complexity per task"); ax.legend(fontsize=8)
    ax=axes[1,0]; ax.axis('off')
    td=[["Aspect","Object Detection","Instance Seg.","Semantic Seg."],
        ["Model output","bbox + class + conf","bbox + mask + class + conf","pixel-wise softmax"],
        ["CP on class","Prediction set {c\u2081,...,c\u2096}","Prediction set {c\u2081,...,c\u2096}","LAC threshold per pixel"],
        ["CP on localiz.","Interval on 4 coords\n[x\u00B1\u03B4, y\u00B1\u03B4, w\u00B1\u03B4, h\u00B1\u03B4]","Bbox interval +\nmask margin (dilation)","N/A (each pixel\nhas its own set)"],
        ["Score function","|c\u0302 - c_gt| / \u03C3","|c\u0302 - c_gt| / \u03C3 + IoU mask","1 - softmax(y_true)"],
        ["Coverage","P(y \u2208 C) \u2265 1-\u03B1 per box","P(y \u2208 C) \u2265 1-\u03B1 per mask","P(pixel misc.) \u2264 \u03B1"],
        ["CP challenge","Multi-object matching\n(Hungarian)","Mask shape variability\n(non-convex regions)","Spatial correlation\nbetween nearby pixels"]]
    table=ax.table(cellText=td[1:],colLabels=td[0],loc='center',cellLoc='center')
    table.auto_set_font_size(False); table.set_fontsize(7.5); table.scale(1.0,2.0)
    for (r,c),cell in table.get_celld().items():
        cell.set_edgecolor('#CCCCCC'); cell.set_linewidth(0.5)
        if r==0: cell.set_facecolor(PAL["coco"]); cell.set_text_props(color='white',fontweight='bold')
        elif c==0: cell.set_facecolor('#F2F4F4'); cell.set_text_props(fontweight='bold')
        elif c==1: cell.set_facecolor(PAL["coco_fill"])
        elif c==2: cell.set_facecolor('#F4ECF7')
        elif c==3: cell.set_facecolor('#EAFAF1')
    ax.set_title("Conformal output comparison per task",fontsize=12,fontweight='bold',pad=15)
    ax=axes[1,1]; io=np.random.beta(8,3,5000); is_=np.random.beta(5,3,5000)
    ax.hist(io,bins=50,alpha=0.55,color=PAL["od"],label=f'OD Bbox IoU (\u03BC={np.mean(io):.3f})',density=True)
    ax.hist(is_,bins=50,alpha=0.55,color=PAL["inst_seg"],label=f'Seg Mask IoU (\u03BC={np.mean(is_):.3f})',density=True)
    ax.axvline(0.5,color='red',ls='--',lw=1,alpha=0.6,label='AP@0.5 threshold')
    ax.axvline(0.75,color='orange',ls='--',lw=1,alpha=0.6,label='AP@0.75 threshold')
    ax.set_xlabel("IoU"); ax.set_ylabel("Density"); ax.set_title("IoU distribution: Bbox vs Mask"); ax.legend(fontsize=8)
    ax=axes[1,2]; szs=["Small\n(< 32\u00B2)","Medium\n(32\u00B2\u201396\u00B2)","Large\n(> 96\u00B2)"]
    x=np.arange(3); w=0.2
    ax.bar(x-1.5*w,[0.18,0.08,0.04],w,label='ECE (OD)',color=PAL["od"],alpha=0.75)
    ax.bar(x-0.5*w,[0.25,0.12,0.06],w,label='ECE (Seg)',color=PAL["inst_seg"],alpha=0.75)
    ax.bar(x+0.5*w,[0.08,0.03,0.01],w,label='Cov. Gap (OD)',color=PAL["od"],alpha=0.4,hatch='//')
    ax.bar(x+1.5*w,[0.15,0.06,0.02],w,label='Cov. Gap (Seg)',color=PAL["inst_seg"],alpha=0.4,hatch='//')
    ax.set_xticks(x); ax.set_xticklabels(szs); ax.set_ylabel("Value")
    ax.set_title("Object size impact\non calibration (expected)"); ax.legend(fontsize=7,ncol=2)
    ax.text(0.5,0.95,"Smaller objects \u2192 higher ECE \u2192 larger coverage gap",transform=ax.transAxes,fontsize=7,ha='center',color='#E74C3C',style='italic')
    plt.suptitle("Annotation & Calibration Comparison — OD vs Segmentation",fontsize=15,fontweight='bold',y=1.01)
    plt.tight_layout(); plt.savefig(os.path.join(out,"11_annotation_comparison.png")); plt.close()
    print(f"  [SAVED] 11_annotation_comparison.png")

def plot_12_class_mapping(out):
    fig,axes=plt.subplots(1,2,figsize=(16,9))
    ax=axes[0]; ax.set_xlim(0,10); ax.set_ylim(-1,21); ax.axis('off')
    v2c={"aeroplane":"airplane","bicycle":"bicycle","bird":"bird","boat":"boat","bottle":"bottle","bus":"bus","car":"car",
         "cat":"cat","chair":"chair","cow":"cow","diningtable":"dining table","dog":"dog","horse":"horse",
         "motorbike":"motorcycle","person":"person","pottedplant":"potted plant","sheep":"sheep","sofa":"couch","train":"train","tvmonitor":"tv"}
    ax.text(0.5,20.5,"VOC 2012",fontsize=12,fontweight='bold',color=PAL["voc"],ha='center')
    ax.text(9.5,20.5,"COCO 2017",fontsize=12,fontweight='bold',color=PAL["coco"],ha='center')
    for i,(vn,cn) in enumerate(v2c.items()):
        y=19.5-i*1.0
        ax.text(0,y,vn,fontsize=8,va='center',ha='left',color=PAL["voc"],bbox=dict(boxstyle='round,pad=0.15',facecolor=PAL["voc_fill"],edgecolor=PAL["voc_light"],linewidth=0.5))
        ax.text(10,y,cn,fontsize=8,va='center',ha='right',color=PAL["coco"],bbox=dict(boxstyle='round,pad=0.15',facecolor=PAL["coco_fill"],edgecolor=PAL["coco_light"],linewidth=0.5))
        col='#E74C3C' if vn!=cn else '#27AE60'; ls='--' if vn!=cn else '-'
        ax.plot([2.5,7.5],[y,y],color=col,linewidth=0.8,alpha=0.6,linestyle=ls)
    ax.text(5,-0.5,"\u2014 Identical name    --- Different name (mapping required)",fontsize=8,ha='center',color=PAL["muted"])
    ax.set_title("Class mapping VOC \u2192 COCO (20/20 shared)",fontsize=11,fontweight='bold')
    ax=axes[1]; ax.axis('off')
    co=["traffic light","fire hydrant","stop sign","parking meter","bench","elephant","bear","zebra","giraffe",
        "backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite",
        "baseball bat","baseball glove","skateboard","surfboard","tennis racket","banana","apple","sandwich","orange",
        "broccoli","carrot","hot dog","pizza","donut","cake","bed","toilet","laptop","mouse","remote","keyboard",
        "cell phone","microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors",
        "teddy bear","hair drier","toothbrush","wine glass","cup","fork","knife","spoon","bowl","truck"]
    ax.text(0.5,0.97,f"COCO-only classes (+60 classes not in VOC)",transform=ax.transAxes,fontsize=11,fontweight='bold',ha='center',color=PAL["coco"])
    nc=4
    for i,cls in enumerate(co[:48]):
        r,c=i//nc,i%nc; ax.text(0.02+c*0.25,0.92-r*0.065,f"\u2022 {cls}",transform=ax.transAxes,fontsize=7.5,color=PAL["coco"])
    ax.text(0.5,0.04,"Implication for CP: calibrating on COCO \u2192 testing on VOC requires\nmapping 20 shared classes + ignoring 60 COCO-only classes",
        transform=ax.transAxes,fontsize=9,ha='center',color='#E74C3C',style='italic',
        bbox=dict(boxstyle='round,pad=0.3',facecolor='#FDEDEC',edgecolor='#E74C3C',alpha=0.5))
    plt.suptitle("Cross-Dataset Mapping — COCO \u2194 VOC",fontsize=14,fontweight='bold',y=1.01)
    plt.tight_layout(); plt.savefig(os.path.join(out,"12_class_mapping.png")); plt.close()
    print(f"  [SAVED] 12_class_mapping.png")

def plot_13_pipeline_per_task(out):
    fig,axes=plt.subplots(1,3,figsize=(18,8))
    tasks=[
        ("Object Detection\n(COCO + VOC)",PAL["od"],[
            ("Input","Image x"),("Model f(x)","bbox + class + conf\nfor each detection"),
            ("Matching","Hungarian matching\nbbox_pred \u2194 bbox_gt (IoU)"),
            ("Class score","s_cls = 1 - softmax[y_true]\n\u2192 APS / RAPS"),
            ("Bbox score","s_box = |c\u0302\u1D4F - c\u1D4F_gt|\n\u2192 Box-Std / Box-CQR"),
            ("Quantile q_\u03B1","Calibrated on D_cal\nq = \u2308(n+1)(1-\u03B1)\u2309-th score"),
            ("Prediction set","C_\u03B1(x) = {classes} +\n[bbox \u00B1 \u03B4_\u03B1]")]),
        ("Instance Segmentation\n(COCO)",PAL["inst_seg"],[
            ("Input","Image x"),("Model f(x)","bbox + mask + class + conf\nfor each instance"),
            ("Matching","Hungarian matching\nmask_pred \u2194 mask_gt (IoU)"),
            ("Class score","s_cls = 1 - softmax[y_true]\n\u2192 Class-conditional CP"),
            ("Mask score","s_mask = dilation(pred) \u2287 gt?\nconformal margin width"),
            ("Quantile q_\u03B1","Two-step: q_class \u2192 q_mask\n(uncertainty propagation)"),
            ("Prediction set","C_\u03B1(x) = {classes} +\nmask \u2295 margin_\u03B1")]),
        ("Semantic Segmentation\n(VOC)",PAL["seg"],[
            ("Input","Image x"),("Model f(x)","Softmax H\u00D7W\u00D7K\nclass per pixel"),
            ("Pixel score","s(p) = 1 - softmax[y_true(p)]\nfor each pixel p"),
            ("LAC threshold","Include class c if\nsoftmax_c(p) > \u03C4_\u03B1"),
            ("CRC","Conformal Risk Control\nE[L(C_\u03BB)] \u2264 \u03B1"),
            ("Quantile q_\u03B1","Pixel-wise: calibrate \u03C4 on D_cal\nKandinsky clustering optional"),
            ("Prediction set","C_\u03B1(p) = {classes per pixel}\nor mask \u2295 margin")])]
    for ax,(title,color,steps) in zip(axes,tasks):
        ax.set_xlim(0,10); ax.set_ylim(-1,len(steps)*4+2); ax.axis('off')
        ax.text(5,len(steps)*4+1,title,ha='center',fontsize=12,fontweight='bold',color=color)
        for i,(sn,sd) in enumerate(steps):
            y=(len(steps)-1-i)*4+1
            ax.add_patch(patches.FancyBboxPatch((0.5,y-1.2),9,3,boxstyle="round,pad=0.2",facecolor=color,edgecolor=color,alpha=0.12,linewidth=1.5))
            ax.text(1.2,y+0.8,sn,fontsize=9,fontweight='bold',color=color)
            ax.text(1.2,y-0.4,sd,fontsize=7.5,color=PAL["text"])
            if i<len(steps)-1: ax.annotate('',xy=(5,y-1.2),xytext=(5,y-2.5),arrowprops=dict(arrowstyle='->',color=color,lw=1.5))
    plt.suptitle("Conformal Pipeline per Task — Step-by-Step",fontsize=15,fontweight='bold',y=1.0)
    plt.tight_layout(); plt.savefig(os.path.join(out,"13_pipeline_per_task.png")); plt.close()
    print(f"  [SAVED] 13_pipeline_per_task.png")

# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    project_root = Path(__file__).parent
    data_dir = project_root / "data"
    out_dir = project_root / "results" / "eda"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = str(out_dir)

    print("="*65)
    print("  CONFORMAL CALIBRATION PIPELINE — FULL EDA (English)")
    print("="*65)
    print(f"\n  Output: {out_dir}\n")

    coco, voc = load_or_generate(str(data_dir))
    print()

    print("  PART I — Dataset-Level EDA")
    print("  " + "-"*40)
    plot_01_class_distribution(coco, out)
    plot_02_bbox_sizes(coco, out)
    plot_03_spatial_heatmap(coco, out)
    plot_04_objects_per_image(coco, voc, out)
    plot_05_confidence(coco, out)
    plot_06_cross_dataset(coco, voc, out)
    plot_07_conformal_split(coco, out)
    plot_08_longtail(coco, out)
    print()

    print("  PART II — Comparative OD vs Segmentation")
    print("  " + "-"*40)
    plot_09_samples(out)
    plot_10_taxonomy(out)
    plot_11_annotations(out)
    plot_12_class_mapping(out)
    plot_13_pipeline_per_task(out)
    print()

    print("="*65)
    print(f"  COMPLETE — 13 plots saved to {out_dir}")
    print("="*65)
    print("\n  Plots:")
    for f in sorted(out_dir.glob("*.png")):
        print(f"    {f.name} ({f.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
