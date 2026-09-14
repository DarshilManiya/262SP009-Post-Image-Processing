import numpy as np, cv2, json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle, Arc

F = "fig"; os.makedirs(F, exist_ok=True)
BLUE="#2479C6"; LBLUE="#68A8E2"; ORANGE="#ED7D31"; GRAY="#404040"; LGRAY="#D9D9D9"
plt.rcParams["font.family"]="DejaVu Sans"

def save(fig, name, dpi=190):
    fig.savefig(f"{F}/{name}.png", dpi=dpi, bbox_inches="tight",
                facecolor="white", pad_inches=0.06)
    plt.close(fig); print("wrote", name)

def box(ax,x,y,w,h,txt,fc,tc="white",fs=12,bold=True):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.012,rounding_size=0.02",
        fc=fc,ec="none"))
    ax.text(x+w/2,y+h/2,txt,ha="center",va="center",color=tc,fontsize=fs,
            fontweight="bold" if bold else "normal",linespacing=1.45)

def arrow(ax,x1,y1,x2,y2,c=ORANGE,lw=2.6):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle="-|>",mutation_scale=20,
        lw=lw,color=c,shrinkA=0,shrinkB=0))

raw = cv2.imread("img/us_raw.png",0)
gt  = cv2.imread("img/us_gt.png",0)

# ---------------- 1. How ultrasound works ----------------
fig,ax=plt.subplots(figsize=(9,4.4)); ax.set_xlim(0,10); ax.set_ylim(0,5); ax.axis("off")
ax.add_patch(Rectangle((0.4,0.4),4.6,4.2,fc="#FCE9D8",ec=ORANGE,lw=2))
ax.text(2.7,4.25,"Inside the body",ha="center",fontsize=12,color=GRAY,fontweight="bold")
ax.add_patch(FancyBboxPatch((2.1,3.35),1.2,0.45,boxstyle="round,pad=0.03",fc=BLUE,ec="none"))
ax.text(2.7,3.57,"PROBE",ha="center",va="center",color="white",fontsize=9.5,fontweight="bold")
for i,r in enumerate([0.45,0.85,1.25]):
    ax.add_patch(Arc((2.7,3.3),r*2,r*1.5,theta1=200,theta2=340,color=BLUE,lw=2.2))
ax.add_patch(Circle((2.7,1.6),0.62,fc="#B9D9F5",ec=BLUE,lw=2))
ax.text(2.7,1.6,"organ",ha="center",va="center",fontsize=9.5,color=GRAY)
for i,r in enumerate([0.5,0.85]):
    ax.add_patch(Arc((2.7,2.3),r*2,r*1.4,theta1=20,theta2=160,color=ORANGE,lw=2.2,ls="--"))
ax.text(4.85,3.1,"sound\nsent in",ha="left",va="center",fontsize=9.5,color=BLUE,fontweight="bold")
ax.text(4.85,1.95,"echoes\ncome back",ha="left",va="center",fontsize=9.5,color=ORANGE,fontweight="bold")
arrow(ax,5.05,2.5,6.25,2.5)
im=cv2.resize(raw,(300,190))
ax.imshow(im,cmap="gray",extent=[6.4,9.8,0.9,4.1],aspect="auto",zorder=3)
ax.add_patch(Rectangle((6.4,0.9),3.4,3.2,fc="none",ec=GRAY,lw=1.6,zorder=4))
ax.text(8.1,4.3,"The picture we see",ha="center",fontsize=12,color=GRAY,fontweight="bold")
save(fig,"how_us_works")

# ---------------- 2. Why ultrasound matters ----------------
fig,ax=plt.subplots(figsize=(11,2.8)); ax.set_xlim(0,12); ax.set_ylim(0,3); ax.axis("off")
items=[("No radiation","Safe even in\npregnancy"),("Low cost","Far cheaper\nthan CT / MRI"),
       ("Portable","Can be taken to\nthe bedside"),("Real time","Live moving\nimages")]
for i,(t,s) in enumerate(items):
    x=0.25+i*2.95
    ax.add_patch(FancyBboxPatch((x,0.25),2.6,2.4,boxstyle="round,pad=0.02,rounding_size=0.08",
        fc="#EAF2FB",ec=LBLUE,lw=1.8))
    ax.text(x+1.3,2.05,t,ha="center",fontsize=14,color=BLUE,fontweight="bold")
    ax.text(x+1.3,1.1,s,ha="center",fontsize=11.5,color=GRAY,linespacing=1.5)
save(fig,"why_us")

# ---------------- 3. Problem: grain zoom ----------------
fig,ax=plt.subplots(figsize=(10,4.2)); ax.set_xlim(0,10); ax.set_ylim(0,4.3); ax.axis("off")
ax.imshow(raw,cmap="gray",extent=[0.2,5.6,0.25,3.9],aspect="auto")
ax.add_patch(Rectangle((0.2,0.25),5.4,3.65,fc="none",ec=GRAY,lw=1.5))
H,W=raw.shape
crop=raw[int(0.30*H):int(0.52*H), int(0.52*W):int(0.78*W)]
crop=cv2.resize(crop,(340,300),interpolation=cv2.INTER_NEAREST)
ax.imshow(crop,cmap="gray",extent=[6.3,9.8,0.5,3.7],aspect="auto")
ax.add_patch(Rectangle((6.3,0.5),3.5,3.2,fc="none",ec=ORANGE,lw=3))
ax.add_patch(Rectangle((3.0,1.6),1.4,0.9,fc="none",ec=ORANGE,lw=2.5))
ax.plot([4.4,6.3],[2.5,3.7],color=ORANGE,lw=1.6,ls="--")
ax.plot([4.4,6.3],[1.6,0.5],color=ORANGE,lw=1.6,ls="--")
ax.text(2.9,4.05,"Simulated ultrasound image",ha="center",fontsize=12.5,color=GRAY,fontweight="bold")
ax.text(8.05,3.95,"Zoomed in: this grain is SPECKLE",ha="center",fontsize=12.5,color=ORANGE,fontweight="bold")
save(fig,"problem_zoom")

# ---------------- 4. Speckle formation ----------------
fig,ax=plt.subplots(figsize=(10,3.9)); ax.set_xlim(0,10.4); ax.set_ylim(0,4); ax.axis("off")
ax.add_patch(Rectangle((0.25,0.45),3.0,3.1,fc="#F3F6FA",ec=LBLUE,lw=1.8))
rng=np.random.default_rng(3)
pts=rng.uniform([0.45,0.65],[3.05,3.35],size=(90,2))
ax.scatter(pts[:,0],pts[:,1],s=13,color=BLUE)
ax.text(1.75,3.72,"1. Tiny structures in tissue",ha="center",fontsize=11.5,color=GRAY,fontweight="bold")
ax.text(1.75,0.18,"too small to see one by one",ha="center",fontsize=10,color=GRAY,style="italic")
arrow(ax,3.42,2.0,4.0,2.0)
ax.add_patch(Rectangle((4.15,0.45),2.9,3.1,fc="#FFF6EE",ec=ORANGE,lw=1.8))
t=np.linspace(0,1,400)
for k,(ph,cc) in enumerate([(0,BLUE),(1.9,ORANGE),(3.6,"#70AD47")]):
    y=0.85+k*0.95+0.22*np.sin(2*np.pi*7*t+ph)
    ax.plot(4.3+t*2.6,y,color=cc,lw=1.9)
ax.text(5.6,3.72,"2. Their echoes overlap",ha="center",fontsize=11.5,color=GRAY,fontweight="bold")
ax.text(5.6,0.18,"waves add up / cancel out",ha="center",fontsize=10,color=GRAY,style="italic")
arrow(ax,7.22,2.0,7.8,2.0)
sp=raw[int(0.35*H):int(0.62*H), int(0.30*W):int(0.58*W)]
ax.imshow(cv2.resize(sp,(300,300)),cmap="gray",extent=[7.95,10.15,0.55,3.45],aspect="auto")
ax.add_patch(Rectangle((7.95,0.55),2.2,2.9,fc="none",ec=GRAY,lw=1.6))
ax.text(9.05,3.72,"3. Result: random grain",ha="center",fontsize=11.5,color=GRAY,fontweight="bold")
ax.text(9.05,0.18,"called speckle",ha="center",fontsize=10,color=ORANGE,style="italic",fontweight="bold")
save(fig,"speckle_formation")

# ---------------- 5. Why it matters (hidden lesion) ----------------
fig,axes=plt.subplots(1,2,figsize=(10,3.9))
for a,im,t,c in zip(axes,[raw,gt],
        ["What the doctor sees","Where the lesions actually are"],[GRAY,ORANGE]):
    a.imshow(im,cmap="gray"); a.axis("off")
    a.set_title(t,fontsize=13,color=c,fontweight="bold",pad=8)
for a in axes:
    a.add_patch(Circle((0.40*W,0.36*H),46,fc="none",ec=ORANGE,lw=2.4,ls="--"))
    a.add_patch(Circle((0.60*W,0.48*H),44,fc="none",ec="#70AD47",lw=2.4,ls="--"))
    a.add_patch(Circle((0.44*W,0.70*H),40,fc="none",ec="#C00000",lw=2.4,ls="--"))
fig.tight_layout(); save(fig,"why_matters")

# ---------------- 6. Where I fit ----------------
fig,ax=plt.subplots(figsize=(11.5,2.9)); ax.set_xlim(0,12); ax.set_ylim(0,3); ax.axis("off")
labs=[("Send sound &\ncollect echoes",BLUE),("Form the raw\ngrayscale image",BLUE),
      ("MY WORK\nClean up the image",ORANGE),("Final image\nfor the doctor","#70AD47")]
for i,(t,c) in enumerate(labs):
    x=0.2+i*3.0
    box(ax,x,0.75,2.5,1.55,t,c,fs=12.5)
    if i<3: arrow(ax,x+2.58,1.52,x+2.92,1.52,c=GRAY,lw=2.2)
ax.text(6.2,0.35,"16-member team project  •  8 modules  •  mine is the final stage",
        ha="center",fontsize=11,color=GRAY,style="italic")
save(fig,"where_i_fit")

# ---------------- 7. The trade-off ----------------
srad=cv2.imread("img/f_srad.png",0)
over=cv2.GaussianBlur(raw,(0,0),7); over=np.where(raw>0,over,0)
fig,axes=plt.subplots(1,3,figsize=(11,3.7))
for a,im,t,sub,c in zip(axes,[raw,srad,over],
    ["TOO LITTLE","JUST RIGHT","TOO MUCH"],
    ["Grain hides\nsmall details","Grain reduced,\nedges kept","Details are\nwiped out"],
    ["#C00000","#70AD47","#C00000"]):
    a.imshow(im,cmap="gray"); a.axis("off")
    a.set_title(t,fontsize=14,color=c,fontweight="bold",pad=6)
    a.text(0.5,-0.10,sub,transform=a.transAxes,ha="center",va="top",
           fontsize=11.5,color=GRAY,linespacing=1.4)
fig.tight_layout(); save(fig,"tradeoff")

# ---------------- 8. Sliding window ----------------
fig,ax=plt.subplots(figsize=(10,4.0)); ax.set_xlim(0,10.6); ax.set_ylim(0,4.4); ax.axis("off")
rng=np.random.default_rng(11)
g=rng.integers(20,230,size=(7,7))
cell=0.44
for i in range(7):
    for j in range(7):
        v=g[i,j]/255
        ax.add_patch(Rectangle((0.4+j*cell,3.5-i*cell),cell,cell,
                     fc=(v,v,v),ec="#BBBBBB",lw=0.6))
ax.add_patch(Rectangle((0.4+2*cell,3.5-4*cell),3*cell,3*cell,fc="none",ec=ORANGE,lw=3.2))
ax.text(1.95,3.85,"Image = grid of brightness values",fontsize=11.5,color=GRAY,fontweight="bold")
ax.text(1.95,0.28,"orange box = the neighbourhood",fontsize=10.5,color=ORANGE,fontweight="bold",ha="center")
arrow(ax,3.85,2.2,4.7,2.2)
box(ax,4.95,1.55,2.5,1.3,"Calculate a new\nvalue from the\n9 neighbours",BLUE,fs=11.5)
arrow(ax,7.6,2.2,8.4,2.2)
ax.add_patch(Rectangle((8.7,1.75),cell,cell,fc="#888888",ec=ORANGE,lw=3))
ax.text(9.35,1.97,"centre pixel\nis replaced",fontsize=11,color=GRAY,va="center",linespacing=1.4)
ax.text(5.3,0.55,"Then slide the box one step and repeat — across the whole image.",
        ha="center",fontsize=11.5,color=GRAY,style="italic")
save(fig,"sliding_window")

# ---------------- 9. My pipeline ----------------
fig,ax=plt.subplots(figsize=(11.5,3.2)); ax.set_xlim(0,12); ax.set_ylim(0,3.2); ax.axis("off")
st=[("STEP 1\nEven out\nbrightness",BLUE),("STEP 2\nRemove the\ngrain",ORANGE),
    ("STEP 3\nRestore\ncontrast",BLUE),("STEP 4\nCheck quality\nwith numbers","#70AD47")]
for i,(t,c) in enumerate(st):
    x=0.35+i*2.92
    box(ax,x,0.95,2.4,1.6,t,c,fs=12)
    if i<3: arrow(ax,x+2.48,1.75,x+2.84,1.75,c=GRAY,lw=2.2)
ax.text(0.35,0.45,"Input: raw grayscale image",fontsize=11,color=GRAY,fontweight="bold")
ax.text(11.65,0.45,"Output: clean image",fontsize=11,color=GRAY,fontweight="bold",ha="right")
save(fig,"pipeline")

# ---------------- 10-14. before/after pairs ----------------
pairs=[("median","Median filter"),("lee","Lee filter"),("srad","SRAD"),
       ("nlm","Non-Local Means"),("wavelet","Wavelet filter")]
for key,title in pairs:
    im=cv2.imread(f"img/f_{key}.png",0)
    fig,axes=plt.subplots(1,2,figsize=(9.2,3.6))
    for a,x,t,c in zip(axes,[raw,im],["BEFORE","AFTER — "+title],[GRAY,ORANGE]):
        a.imshow(x,cmap="gray"); a.axis("off")
        a.set_title(t,fontsize=13.5,color=c,fontweight="bold",pad=6)
    fig.tight_layout(); save(fig,f"ba_{key}")

# ---------------- 15. CLAHE ----------------
enh=cv2.imread("img/f_enhanced.png",0)
fig,axes=plt.subplots(1,2,figsize=(9.2,3.6))
for a,x,t,c in zip(axes,[srad,enh],
        ["After despeckling (flat)","After contrast enhancement"],[GRAY,ORANGE]):
    a.imshow(x,cmap="gray"); a.axis("off")
    a.set_title(t,fontsize=13,color=c,fontweight="bold",pad=6)
fig.tight_layout(); save(fig,"ba_clahe")

# ---------------- 16. comparison grid ----------------
fig,axes=plt.subplots(2,3,figsize=(11,5.4))
show=[(raw,"Original (raw)"),(cv2.imread("img/f_median.png",0),"Median"),
      (cv2.imread("img/f_lee.png",0),"Lee"),(srad,"SRAD"),
      (cv2.imread("img/f_nlm.png",0),"Non-Local Means"),
      (cv2.imread("img/f_wavelet.png",0),"Wavelet")]
for a,(x,t) in zip(axes.ravel(),show):
    a.imshow(x,cmap="gray"); a.axis("off")
    a.set_title(t,fontsize=12.5,color=GRAY,fontweight="bold",pad=4)
axes.ravel()[3].set_title("SRAD  ★ best balance",fontsize=12.5,color=ORANGE,fontweight="bold",pad=4)
fig.tight_layout(); save(fig,"comparison")

# ---------------- 17. metrics ROI ----------------
fig,ax=plt.subplots(figsize=(9,4.0)); ax.axis("off")
ax.imshow(raw,cmap="gray")
ax.add_patch(Circle((0.40*W,0.36*H),30,fc="none",ec="#C00000",lw=3))
ax.add_patch(Circle((0.66*W,0.36*H),30,fc="none",ec="#70AD47",lw=3))
ax.annotate("inside the lesion",(0.40*W,0.36*H),(0.14*W,0.10*H),color="#C00000",
    fontsize=12.5,fontweight="bold",arrowprops=dict(arrowstyle="->",color="#C00000",lw=2))
ax.annotate("normal tissue",(0.66*W,0.36*H),(0.74*W,0.10*H),color="#70AD47",
    fontsize=12.5,fontweight="bold",arrowprops=dict(arrowstyle="->",color="#70AD47",lw=2))
ax.text(0.5,-0.07,"CONTRAST = how far apart these two look, compared to how noisy they are",
    transform=ax.transAxes,ha="center",fontsize=12,color=GRAY,fontweight="bold")
save(fig,"metrics_roi")

# ---------------- 18. results bar chart ----------------
m=json.load(open("metrics.json"))
order=["Original (raw)","Median","Lee","Wavelet","NLM","SRAD"]
cnr=[m[k]["CNR"] for k in order]; snr=[m[k]["SNR"] for k in order]
epi=[m[k]["EPI"] for k in order]
fig,axes=plt.subplots(1,3,figsize=(12,3.5))
data=[(cnr,"Contrast (higher = better)",BLUE),
      (snr,"Grain removed (higher = better)",ORANGE),
      (epi,"Edges kept (higher = better)","#70AD47")]
for a,(vals,t,c) in zip(axes,data):
    cols=[LGRAY]+[c]*5
    cols[order.index("SRAD")]="#C00000" if False else c
    b=a.bar(range(6),vals,color=cols,edgecolor="none")
    b[0].set_color("#BFBFBF")
    a.set_xticks(range(6)); a.set_xticklabels(order,rotation=32,ha="right",fontsize=10,color=GRAY)
    a.set_title(t,fontsize=12,color=GRAY,fontweight="bold",pad=8)
    a.spines[["top","right"]].set_visible(False)
    a.tick_params(axis="y",labelsize=9,colors=GRAY)
    for r,v in zip(b,vals):
        a.text(r.get_x()+r.get_width()/2,v*1.02,f"{v:.2f}",ha="center",
               fontsize=9.5,color=GRAY,fontweight="bold")
fig.tight_layout(); save(fig,"bar_results")

# ---------------- 19. limitation: oversmoothing ----------------
strong=cv2.GaussianBlur(raw,(0,0),5); strong=np.where(raw>0,strong,0)
fig,axes=plt.subplots(1,3,figsize=(11,3.6))
for a,x,t,c in zip(axes,[raw,srad,strong],
    ["Original","Good setting","Filter pushed too hard"],[GRAY,"#70AD47","#C00000"]):
    a.imshow(x,cmap="gray"); a.axis("off")
    a.set_title(t,fontsize=13,color=c,fontweight="bold",pad=6)
axes[2].add_patch(Circle((0.60*W,0.48*H),46,fc="none",ec="#C00000",lw=2.6,ls="--"))
axes[2].text(0.5,-0.08,"small lesion has almost disappeared",transform=axes[2].transAxes,
             ha="center",fontsize=11,color="#C00000",fontweight="bold")
fig.tight_layout(); save(fig,"limits")

# ---------------- 20. summary hero ----------------
fig,axes=plt.subplots(1,2,figsize=(9.6,3.8))
for a,x,t,c in zip(axes,[raw,enh],["Before my stage","After my stage"],[GRAY,ORANGE]):
    a.imshow(x,cmap="gray"); a.axis("off")
    a.set_title(t,fontsize=14,color=c,fontweight="bold",pad=6)
fig.tight_layout(); save(fig,"summary_hero")
print("ALL FIGURES DONE")
