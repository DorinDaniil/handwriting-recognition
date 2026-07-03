#!/usr/bin/env python3
"""Fast 4-point polygon annotator for handwritten line detection (local web app).

Per image it preloads the sibling <stem>.txt (one transcription per line, the manual text
markup), shows the lines in reading order, and you click the 4 corners of each line's quad.
After the 4th click the polygon closes and the next text line becomes active, so you just
click-click-click-click down the page. Output is saved next to the image as <stem>.json:

    {"image": "...", "width": W, "height": H,
     "lines": [{"order": 1, "polygon": [[x,y],[x,y],[x,y],[x,y]], "text": "..."}]}

which already pairs geometry with text (feed straight to crops + TsvLineDataset).

On every save it ALSO rebuilds a central PaddleOCR-style <root>/labels.txt — the exact
format the DBNet++ detector trains on (detection/src/dataset.py):

    <rel_path>\t[{"transcription": "...", "points": [[x,y],[x,y],[x,y],[x,y]], "score": 1.0}, ...]

so to fine-tune the detector you only point it at this root + labels.txt and run
detection/split_dataset.py to make train/val splits. Only lines with a full 4-point polygon
are exported.

    pip install flask
    python annotate.py                      # serves ./data on http://127.0.0.1:5000
    python annotate.py --root data --port 5001
Then open the URL in your browser.

Shortcuts (in the page): click=add corner, drag a corner to move it, [z] undo last point,
[r] reset active line, [Tab]/[Shift+Tab] or click=select line, [a] add line, [x] delete line,
[s] save, [n]/[p] next/prev image (autosaves), wheel=zoom, right-drag or Space+drag=pan,
[f] fit, [h] toggle help.
"""
import argparse
import json
from pathlib import Path

from flask import Flask, Response, request, send_file

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
app = Flask(__name__)
ROOT = Path("data")
SKIP_DIRS = {".ipynb_checkpoints", "data_cropped"}


def _images():
    out = []
    for p in sorted(ROOT.rglob("*")):
        if p.suffix.lower() in IMG_EXT and not any(d in p.parts for d in SKIP_DIRS):
            out.append(p.relative_to(ROOT).as_posix())
    return out


def _txt_lines(img: Path):
    t = img.with_suffix(".txt")
    if not t.exists():
        return []
    return [ln.rstrip("\n") for ln in t.read_text(encoding="utf-8").splitlines() if ln.strip()]


@app.get("/")
def index():
    return Response(PAGE, mimetype="text/html")


@app.get("/api/images")
def api_images():
    imgs = _images()
    done = {p: (ROOT / p).with_suffix(".json").exists() for p in imgs}
    return {"images": imgs, "done": done}


@app.get("/image")
def image():
    p = (ROOT / request.args["path"]).resolve()
    if ROOT.resolve() not in p.parents and p != ROOT.resolve():
        return "forbidden", 403
    return send_file(p)


@app.get("/api/ann")
def get_ann():
    rel = request.args["path"]
    img = ROOT / rel
    jp = img.with_suffix(".json")
    if jp.exists():
        data = json.loads(jp.read_text(encoding="utf-8"))
        # keep texts in sync with the .txt if lengths still match and a polygon is missing
        return data
    # build a fresh annotation from the .txt: one line per transcription, no polygons yet
    texts = _txt_lines(img)
    return {"image": img.name, "width": 0, "height": 0,
            "lines": [{"order": i + 1, "polygon": [], "text": t} for i, t in enumerate(texts)]}


@app.post("/api/ann")
def save_ann():
    rel = request.args["path"]
    img = ROOT / rel
    data = request.get_json(force=True)
    lines = data.get("lines", [])
    for i, ln in enumerate(lines):
        ln["order"] = i + 1
    img.with_suffix(".json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    # mirror the text targets back to <stem>.txt: one line per annotation line, in order
    # (a newline/tab inside a single line's text is collapsed to a space)
    if lines:
        txt = "\n".join((ln.get("text") or "").replace("\r", " ").replace("\n", " ").replace("\t", " ")
                        for ln in lines)
        img.with_suffix(".txt").write_text(txt + "\n", encoding="utf-8")
    n_imgs, n_boxes = rebuild_labels()
    return {"ok": True, "label_images": n_imgs, "label_boxes": n_boxes}


@app.get("/api/export")
def export():
    n_imgs, n_boxes = rebuild_labels()
    return {"labels_txt": str((ROOT / "labels.txt").resolve()),
            "images": n_imgs, "boxes": n_boxes}


def rebuild_labels():
    """Scan every <stem>.json under ROOT and (re)write ROOT/labels.txt in PaddleOCR format.
    A box is exported only if its polygon has 4 points. Returns (n_images, n_boxes)."""
    lines_out, n_imgs, n_boxes = [], 0, 0
    for jp in sorted(ROOT.rglob("*.json")):
        if jp.name == "labels.txt" or any(d in jp.parts for d in SKIP_DIRS):
            continue
        try:
            data = json.loads(jp.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(data, dict) or "lines" not in data:
            continue
        img = None
        for ext in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"):
            if jp.with_suffix(ext).exists():
                img = jp.with_suffix(ext)
                break
        if img is None:
            continue
        boxes = []
        for ln in data["lines"]:
            poly = ln.get("polygon") or []
            if len(poly) != 4:
                continue
            boxes.append({
                "transcription": (ln.get("text") or "").replace("\t", " ").replace("\n", " "),
                "points": [[int(round(x)), int(round(y))] for x, y in poly],
                "score": 1.0,
            })
        if not boxes:
            continue
        rel = img.relative_to(ROOT).as_posix()
        lines_out.append(rel + "\t" + json.dumps(boxes, ensure_ascii=False))
        n_imgs += 1
        n_boxes += len(boxes)
    (ROOT / "labels.txt").write_text("\n".join(lines_out) + ("\n" if lines_out else ""),
                                     encoding="utf-8")
    return n_imgs, n_boxes


PAGE = r"""<!doctype html><html><head><meta charset="utf-8"><title>Line annotator</title>
<style>
  :root{--bg:#1e1e1e;--panel:#252526;--line:#333;--accent:#4ec9b0;--muted:#888;--text:#ddd;}
  *{box-sizing:border-box}
  body{margin:0;font:13px/1.4 system-ui,Segoe UI,sans-serif;background:var(--bg);color:var(--text);height:100vh;overflow:hidden}
  #top{height:40px;display:flex;align-items:center;gap:10px;padding:0 10px;background:var(--panel);border-bottom:1px solid var(--line)}
  #top b{color:var(--accent)} #top .sp{flex:1}
  button{background:#333;color:var(--text);border:1px solid #444;border-radius:4px;padding:4px 9px;cursor:pointer}
  button:hover{background:#3a3a3a}
  #wrap{display:flex;height:calc(100vh - 40px)}
  #files{width:230px;overflow:auto;background:var(--panel);border-right:1px solid var(--line)}
  #files div{padding:5px 9px;cursor:pointer;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;border-bottom:1px solid #2a2a2a;font-size:12px}
  #files div:hover{background:#2d2d2d}
  #files div.cur{background:#094771} #files div.done::before{content:"✓ ";color:var(--accent)}
  #mid{flex:1;position:relative;overflow:hidden;background:#111}
  canvas{position:absolute;top:0;left:0;cursor:crosshair}
  #side{width:320px;overflow:auto;background:var(--panel);border-left:1px solid var(--line)}
  .ln{padding:6px 9px;border-bottom:1px solid #2a2a2a;cursor:pointer;display:flex;gap:7px;align-items:flex-start}
  .ln:hover{background:#2d2d2d} .ln.active{background:#37373d;outline:1px solid var(--accent);outline-offset:-1px}
  .ln .num{color:var(--accent);min-width:20px;font-weight:bold;cursor:grab}
  .ln .num:active{cursor:grabbing}
  .ln .tx{flex:1;white-space:pre-wrap;word-break:break-word;padding:2px 3px;border-radius:3px}
  .ln .tx:focus{outline:1px solid var(--accent);background:#1e1e1e}
  .ln.full .num::after{content:" ●";color:#6a9955} .ln .cnt{color:var(--muted);font-size:11px}
  .ln.drop{border-top:2px solid var(--accent)}
  .ln .mv{display:flex;flex-direction:column;gap:1px}
  .ln .mv button{padding:0 4px;font-size:9px;line-height:12px;border-radius:3px}
  .ln .mv button.del{color:#e06060;border-color:#5a3a3a}
  .ln .mv button.del:hover{background:#4a2a2a}
  #help{position:absolute;right:10px;bottom:10px;background:#000a;padding:10px 12px;border-radius:6px;font-size:12px;max-width:420px;color:#ccc;display:none;white-space:pre-line}
  #status{color:var(--muted)}
</style></head><body>
<div id="top">
  <button onclick="prev()">◀ Prev (p)</button>
  <button onclick="next()">Next (n) ▶</button>
  <button onclick="save()">Save (s)</button>
  <button onclick="fit()">Fit (f)</button>
  <button onclick="addLine()">+ Line (a)</button>
  <button onclick="exportLabels()">Export labels.txt</button>
  <b id="title"></b><span id="status"></span><span class="sp"></span>
  <span id="prog"></span>
  <button onclick="toggleHelp()">? help (h)</button>
</div>
<div id="wrap">
  <div id="files"></div>
  <div id="mid"><canvas id="cv"></canvas>
    <div id="help"></div>
  </div>
  <div id="side"></div>
</div>
<script>
const HELP=`Click = add a corner of the active line (4 corners -> line done, next line auto-selected).
Drag a corner = move it.   z = undo last point   r = reset active line   x = delete active line
Tab / Shift+Tab or click # = select line    a = add a blank line
Reorder lines: drag the #number, use ▲▼ buttons, or Alt+↑ / Alt+↓ (numbering, polygon & text move together).
Click the text in the right panel to EDIT it; on save it's written back to <stem>.txt.
wheel = zoom at cursor   right-drag or Space+drag = pan   f = fit   s = save   n/p = next/prev (autosaves)`;
let images=[], done={}, idx=-1, ann=null, active=0, dirty=false, dragIdx=-1;
let img=new Image(), scale=1, ox=0, oy=0, panning=false, space=false, drag=null, px=0,py=0;
const cv=document.getElementById('cv'), ctx=cv.getContext('2d'), mid=document.getElementById('mid');

function resize(){cv.width=mid.clientWidth;cv.height=mid.clientHeight;draw();}
window.addEventListener('resize',resize);

async function loadList(){
  const r=await fetch('/api/images'); const j=await r.json();
  images=j.images; done=j.done; renderFiles();
  if(images.length) open(0);
}
function renderFiles(){
  const f=document.getElementById('files'); f.innerHTML='';
  images.forEach((p,i)=>{const d=document.createElement('div');d.textContent=p;
    if(done[p])d.classList.add('done'); if(i===idx)d.classList.add('cur');
    d.onclick=()=>open(i); f.appendChild(d);});
  document.getElementById('prog').textContent=
    Object.values(done).filter(Boolean).length+' / '+images.length+' done';
}
async function open(i){
  if(ann && idx>=0) await save(true);
  idx=i; active=0; const p=images[i];
  document.getElementById('title').textContent=p;
  const r=await fetch('/api/ann?path='+encodeURIComponent(p)); ann=await r.json();
  img=new Image();
  img.onload=()=>{ann.width=img.naturalWidth;ann.height=img.naturalHeight;fit();renderSide();};
  img.src='/image?path='+encodeURIComponent(p)+'&t='+Date.now();
  renderFiles();
}
function fit(){
  if(!img.naturalWidth)return;
  scale=Math.min(cv.width/img.naturalWidth, cv.height/img.naturalHeight)*0.98;
  ox=(cv.width-img.naturalWidth*scale)/2; oy=(cv.height-img.naturalHeight*scale)/2; draw();
}
const S=(x,y)=>[x*scale+ox, y*scale+oy];
const I=(x,y)=>[(x-ox)/scale,(y-oy)/scale];

function draw(){
  ctx.clearRect(0,0,cv.width,cv.height);
  if(img.naturalWidth) ctx.drawImage(img,ox,oy,img.naturalWidth*scale,img.naturalHeight*scale);
  if(!ann)return;
  ann.lines.forEach((ln,li)=>{
    const pts=ln.polygon; if(!pts.length)return;
    const act=li===active;
    ctx.lineWidth=act?2.5:1.5;
    ctx.strokeStyle=act?'#4ec9b0':(pts.length===4?'#e0a030':'#c05050');
    ctx.fillStyle=act?'rgba(78,201,176,.12)':'rgba(224,160,48,.07)';
    ctx.beginPath();
    pts.forEach((pt,k)=>{const[x,y]=S(pt[0],pt[1]); k?ctx.lineTo(x,y):ctx.moveTo(x,y);});
    if(pts.length===4)ctx.closePath();
    if(pts.length===4)ctx.fill(); ctx.stroke();
    pts.forEach((pt,k)=>{const[x,y]=S(pt[0],pt[1]);
      ctx.fillStyle=act?'#4ec9b0':'#e0a030';
      ctx.beginPath();ctx.arc(x,y,act?5:3.5,0,7);ctx.fill();
      if(act){ctx.fillStyle='#000';ctx.font='10px sans-serif';ctx.fillText(k+1,x-3,y+3);}});
    const[lx,ly]=S(pts[0][0],pts[0][1]);
    ctx.fillStyle=act?'#4ec9b0':'#e0a030';ctx.font='bold 14px sans-serif';
    ctx.fillText('#'+(li+1),lx+4,ly-6);
  });
}
function renderSide(){
  const s=document.getElementById('side'); s.innerHTML='';
  ann.lines.forEach((ln,i)=>{
    const row=document.createElement('div');
    row.className='ln'+(i===active?' active':'')+(ln.polygon.length===4?' full':'');
    row.ondragover=e=>{e.preventDefault();row.classList.add('drop');};
    row.ondragleave=()=>row.classList.remove('drop');
    row.ondrop=e=>{e.preventDefault();row.classList.remove('drop');moveTo(dragIdx,i);};

    const num=document.createElement('span'); num.className='num'; num.textContent=i+1;
    num.title='drag to reorder / click to select'; num.draggable=true;
    num.ondragstart=e=>{dragIdx=i;e.dataTransfer.effectAllowed='move';};
    num.onclick=()=>{active=i;renderSide();draw();};

    const mv=document.createElement('div'); mv.className='mv';
    const up=document.createElement('button'); up.textContent='▲'; up.title='move up (Alt+↑)';
    up.onclick=()=>nudge(i,-1);
    const dn=document.createElement('button'); dn.textContent='▼'; dn.title='move down (Alt+↓)';
    dn.onclick=()=>nudge(i,1);
    const del=document.createElement('button'); del.textContent='✕'; del.title='delete line (x)';
    del.className='del'; del.onclick=()=>{active=i; delLine();};
    mv.appendChild(up); mv.appendChild(dn); mv.appendChild(del);

    const col=document.createElement('div'); col.style.flex='1';
    const tx=document.createElement('div'); tx.className='tx'; tx.contentEditable='true'; tx.spellcheck=false;
    tx.textContent=ln.text||'';
    tx.oninput=()=>{ln.text=tx.innerText; dirty=true;};
    tx.onfocus=()=>{active=i;
      document.querySelectorAll('#side .ln').forEach(r=>r.classList.remove('active'));
      row.classList.add('active'); draw();};
    const cnt=document.createElement('div'); cnt.className='cnt'; cnt.textContent=ln.polygon.length+'/4 pts';
    col.appendChild(tx); col.appendChild(cnt);
    row.appendChild(num); row.appendChild(mv); row.appendChild(col);
    s.appendChild(row);
  });
}
function reindex(){ann.lines.forEach((l,i)=>l.order=i+1);}
function nudge(from,delta){                 // move a line up/down by one
  const to=from+delta; if(to<0||to>=ann.lines.length)return;
  const it=ann.lines.splice(from,1)[0]; ann.lines.splice(to,0,it);
  active=to; dirty=true; reindex(); renderSide(); draw();
}
function moveTo(from,before){               // drag-drop: move `from` to sit before row `before`
  if(from<0||from===before)return;
  const it=ann.lines.splice(from,1)[0];
  let to=before; if(from<before)to--; to=Math.max(0,Math.min(ann.lines.length,to));
  ann.lines.splice(to,0,it); active=to; dirty=true; reindex(); renderSide(); draw();
}
function curLine(){return ann.lines[active];}
function hitPoint(mx,my){
  const ln=curLine(); if(!ln)return -1;
  for(let k=0;k<ln.polygon.length;k++){const[x,y]=S(ln.polygon[k][0],ln.polygon[k][1]);
    if(Math.hypot(x-mx,y-my)<9)return k;} return -1;
}
cv.addEventListener('mousedown',e=>{
  const r=cv.getBoundingClientRect(),mx=e.clientX-r.left,my=e.clientY-r.top;
  if(e.button===2||space){panning=true;px=mx;py=my;return;}
  const h=hitPoint(mx,my);
  if(h>=0){drag=h;return;}                         // grab a corner to move
  const ln=curLine(); if(!ln)return;
  if(ln.polygon.length>=4){                         // start fresh if already 4: ignore (use r to reset)
    return;
  }
  const[ix,iy]=I(mx,my); ln.polygon.push([Math.round(ix),Math.round(iy)]);
  if(ln.polygon.length===4){                        // line done -> jump to next empty line
    let nx=ann.lines.findIndex((l,i)=>i>active && l.polygon.length<4);
    if(nx<0)nx=ann.lines.findIndex(l=>l.polygon.length<4);
    if(nx>=0)active=nx;
  }
  renderSide();draw();
});
cv.addEventListener('mousemove',e=>{
  const r=cv.getBoundingClientRect(),mx=e.clientX-r.left,my=e.clientY-r.top;
  if(panning){ox+=mx-px;oy+=my-py;px=mx;py=my;draw();return;}
  if(drag!==null){const[ix,iy]=I(mx,my);curLine().polygon[drag]=[Math.round(ix),Math.round(iy)];draw();}
});
window.addEventListener('mouseup',()=>{panning=false;drag=null;});
cv.addEventListener('contextmenu',e=>e.preventDefault());
cv.addEventListener('wheel',e=>{
  e.preventDefault();
  const r=cv.getBoundingClientRect(),mx=e.clientX-r.left,my=e.clientY-r.top;
  const[ix,iy]=I(mx,my); const f=e.deltaY<0?1.15:1/1.15; scale*=f;
  ox=mx-ix*scale; oy=my-iy*scale; draw();
},{passive:false});

function undo(){
  let ln=curLine();
  // if the active line is empty (we just auto-jumped after closing a quad),
  // step back to the previous line that has points and undo there
  if(ln && ln.polygon.length===0){
    for(let i=active-1;i>=0;i--){ if(ann.lines[i].polygon.length){active=i;ln=ann.lines[i];break;} }
  }
  if(ln && ln.polygon.length){ln.polygon.pop();renderSide();draw();}
}
function reset(){const ln=curLine();if(ln){ln.polygon=[];renderSide();draw();}}
function addLine(){ann.lines.push({order:ann.lines.length+1,polygon:[],text:''});active=ann.lines.length-1;renderSide();draw();}
function delLine(){if(!ann.lines.length)return;ann.lines.splice(active,1);active=Math.max(0,active-1);renderSide();draw();}
function prev(){if(idx>0)open(idx-1);}
function next(){if(idx<images.length-1)open(idx+1);}
async function save(silent){
  if(!ann)return;
  const st=document.getElementById('status'); st.textContent=' saving…';
  const r=await fetch('/api/ann?path='+encodeURIComponent(images[idx]),
    {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(ann)});
  const j=await r.json();
  done[images[idx]]=ann.lines.some(l=>l.polygon.length===4); renderFiles();
  st.textContent=' saved  ·  labels.txt: '+j.label_images+' imgs / '+j.label_boxes+' boxes';
  if(!silent)setTimeout(()=>st.textContent='',2000);
}
async function exportLabels(){
  const r=await fetch('/api/export'); const j=await r.json();
  document.getElementById('status').textContent=' labels.txt -> '+j.images+' imgs / '+j.boxes+' boxes';
}
function toggleHelp(){const h=document.getElementById('help');h.style.display=h.style.display==='block'?'none':'block';h.textContent=HELP;}

document.addEventListener('keydown',e=>{
  if(e.code==='Space'){space=true;return;}
  if(e.target.isContentEditable||e.target.tagName==='INPUT'||e.target.tagName==='TEXTAREA'){
    if((e.ctrlKey||e.metaKey)&&e.key.toLowerCase()==='s'){e.preventDefault();e.target.blur();save();}
    return;
  }
  if((e.ctrlKey||e.metaKey)&&e.key.toLowerCase()==='z'){e.preventDefault();undo();}
  else if(e.key==='z'){undo();}
  else if(e.key==='r'){reset();}
  else if(e.key==='x'){delLine();}
  else if(e.key==='a'){addLine();}
  else if(e.key==='s'){e.preventDefault();save();}
  else if(e.key==='n'){next();}
  else if(e.key==='p'){prev();}
  else if(e.key==='f'){fit();}
  else if(e.key==='h'){toggleHelp();}
  else if(e.altKey&&e.key==='ArrowUp'){e.preventDefault();nudge(active,-1);}
  else if(e.altKey&&e.key==='ArrowDown'){e.preventDefault();nudge(active,1);}
  else if(e.key==='Tab'){e.preventDefault();active=(active+(e.shiftKey?-1:1)+ann.lines.length)%ann.lines.length;renderSide();draw();}
});
document.addEventListener('keyup',e=>{if(e.code==='Space')space=false;});
resize(); loadList();
</script></body></html>"""


def main():
    global ROOT
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data", help="folder with images (+ sibling .txt)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5000)
    args = ap.parse_args()
    ROOT = Path(args.root)
    if not ROOT.exists():
        raise SystemExit(f"root not found: {ROOT.resolve()}")
    print(f"serving {ROOT.resolve()}  ->  http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
