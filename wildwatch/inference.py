"""Reusable pretrained YOLO11n image/video inference and CSV export."""
from __future__ import annotations
import argparse
import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

IMAGE_SUFFIXES={".jpg",".jpeg",".png",".bmp",".tif",".tiff",".webp"}
VIDEO_SUFFIXES={".mp4",".avi",".mov",".mkv",".mpeg",".mpg",".m4v"}
CSV_FIELDS=["source","frame_index","detection_index","class_id","class_name","confidence","x1","y1","x2","y2"]

@dataclass(frozen=True)
class Detection:
    """One prediction with pixel-coordinate xyxy bounds."""
    source: str
    frame_index: Optional[int]
    detection_index: int
    class_id: int
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float
    def to_row(self):
        row=asdict(self)
        if row["frame_index"] is None: row["frame_index"]=""
        return row

def validate_confidence(value: float)->float:
    value=float(value)
    if not 0.0<=value<=1.0: raise ValueError("confidence threshold must be between 0 and 1")
    return value

def xyxy_from_values(values):
    if len(values)!=4: raise ValueError("a bounding box must contain exactly four xyxy values")
    x1,y1,x2,y2=map(float,values)
    if x2<x1 or y2<y1: raise ValueError("invalid xyxy box")
    return x1,y1,x2,y2

def detections_from_result(result,source: str,frame_index=None):
    boxes=getattr(result,"boxes",None)
    if boxes is None or len(boxes)==0: return []
    names=getattr(result,"names",{}) or {}
    rows=[]
    for i,box in enumerate(boxes):
        x1,y1,x2,y2=xyxy_from_values(box.xyxy[0].tolist())
        class_id=int(box.cls[0].item()); confidence=float(box.conf[0].item())
        if isinstance(names,dict): class_name=str(names.get(class_id,class_id))
        else: class_name=str(names[class_id]) if class_id<len(names) else str(class_id)
        rows.append(Detection(source,frame_index,i,class_id,class_name,confidence,x1,y1,x2,y2))
    return rows

def write_detections_csv(rows,path: Path)->int:
    path.parent.mkdir(parents=True,exist_ok=True); count=0
    with path.open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=CSV_FIELDS); writer.writeheader()
        for row in rows: writer.writerow(row.to_row()); count+=1
    return count

def load_yolo_model(model_path: str):
    try: from ultralytics import YOLO
    except ImportError as exc: raise RuntimeError("Install dependencies with: pip install -r requirements.txt") from exc
    return YOLO(model_path)

def run_image(model,image_path: Path,output_path: Path,confidence: float,device=None):
    import cv2
    kwargs={"source":str(image_path),"conf":confidence,"verbose":False}
    if device: kwargs["device"]=device
    results=model.predict(**kwargs)
    if not results: raise RuntimeError(f"No inference result returned for {image_path}")
    result=results[0]; output_path.parent.mkdir(parents=True,exist_ok=True)
    if not cv2.imwrite(str(output_path),result.plot()): raise OSError(f"Could not write {output_path}")
    return detections_from_result(result,image_path.name)

def run_video(model,video_path: Path,output_path: Path,confidence: float,device=None):
    import cv2
    cap=cv2.VideoCapture(str(video_path))
    if not cap.isOpened(): cap.release(); raise OSError(f"Could not open video: {video_path}")
    fps=cap.get(cv2.CAP_PROP_FPS) or 25.0
    width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width<=0 or height<=0: cap.release(); raise ValueError("Video reports invalid frame dimensions")
    output_path.parent.mkdir(parents=True,exist_ok=True)
    writer=cv2.VideoWriter(str(output_path),cv2.VideoWriter_fourcc(*"mp4v"),fps,(width,height))
    if not writer.isOpened(): cap.release(); writer.release(); raise OSError(f"Could not create {output_path}")
    rows=[]; frame_index=0
    try:
        while True:
            ok,frame=cap.read()
            if not ok: break
            kwargs={"source":frame,"conf":confidence,"verbose":False}
            if device: kwargs["device"]=device
            results=model.predict(**kwargs)
            if results:
                result=results[0]; writer.write(result.plot())
                rows.extend(detections_from_result(result,video_path.name,frame_index))
            else: writer.write(frame)
            frame_index+=1
    finally: cap.release(); writer.release()
    if frame_index==0: raise ValueError(f"Video contains no readable frames: {video_path}")
    return rows

def process_media(input_path: Path,output_dir: Path,model_path="yolo11n.pt",confidence=0.25,device=None):
    confidence=validate_confidence(confidence)
    input_path=input_path.expanduser().resolve()
    if not input_path.is_file(): raise FileNotFoundError(input_path)
    output_dir=output_dir.expanduser().resolve(); output_dir.mkdir(parents=True,exist_ok=True)
    model=load_yolo_model(model_path); suffix=input_path.suffix.lower()
    if suffix in IMAGE_SUFFIXES:
        output_media=output_dir/f"{input_path.stem}_detections{suffix}"
        rows=run_image(model,input_path,output_media,confidence,device)
    elif suffix in VIDEO_SUFFIXES:
        output_media=output_dir/f"{input_path.stem}_detections.mp4"
        rows=run_video(model,input_path,output_media,confidence,device)
    else: raise ValueError(f"Unsupported media extension: {suffix}")
    csv_path=output_dir/f"{input_path.stem}_detections.csv"
    count=write_detections_csv(rows,csv_path)
    print(f"Annotated output: {output_media}"); print(f"Detection records: {count} -> {csv_path}")
    return csv_path

def main(argv=None):
    parser=argparse.ArgumentParser(description="WildWatch: pretrained YOLO11n detection and localization")
    parser.add_argument("--input",required=True,type=Path)
    parser.add_argument("--output-dir",type=Path,default=Path("runs/wildwatch"))
    parser.add_argument("--model",default="yolo11n.pt")
    parser.add_argument("--confidence",type=validate_confidence,default=0.25)
    parser.add_argument("--device",default=None,help="Optional device, e.g. cpu or 0")
    args=parser.parse_args(argv)
    process_media(args.input,args.output_dir,args.model,args.confidence,args.device)
    return 0

if __name__=="__main__": raise SystemExit(main())
