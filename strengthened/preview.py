from pathlib import Path
import pypdfium2 as pdfium
from PIL import Image,ImageDraw
from pypdf import PdfReader
p=Path('D:/Xai/paper/emse');out=p/'preview';out.mkdir(exist_ok=True);doc=pdfium.PdfDocument(str(p/'main.pdf'));thumbs=[]
for i,page in enumerate(doc):
 im=page.render(scale=1.3).to_pil().convert('RGB');im.save(out/f'page_{i+1:02}.png');im.thumbnail((280,400));c=Image.new('RGB',(290,425),'#dddddd');c.paste(im,((290-im.width)//2,20));ImageDraw.Draw(c).text((6,3),str(i+1),fill='black');thumbs.append(c)
sheet=Image.new('RGB',(1160,425*((len(thumbs)+3)//4)),'white')
for i,im in enumerate(thumbs):sheet.paste(im,((i%4)*290,(i//4)*425))
sheet.save(out/'contact_sheet.png');reader=PdfReader(p/'main.pdf');text='\n'.join(pg.extract_text() for pg in reader.pages);(out/'extracted_text.txt').write_text(text,encoding='utf-8');print(len(reader.pages),'pages')
