import os












directory = r'/content/drive/MyDrive/PFE/Données/image_facture'
for filename in os.listdir(directory):
    if filename.endswith(".jpg"):
        file.append(os.path.join(directory, filename))
        print(os.path.join(directory, filename))

    else:
        continue
        
        
def detect_text_blocks(img_path):
    detection_result = reader.detect(img_path,
                                 width_ths=0.7, 
                                 mag_ratio=1.5
                                 )
    text_coordinates = detection_result[0][0]
    return text_coordinates

def draw_bounds(img_path, bbox):
    image = Image.open(img_path)  
    draw = ImageDraw.Draw(image)
    for b in bbox:
        p0, p1, p2, p3 = [b[0], b[2]], [b[1], b[2]], \
                         [b[1], b[3]], [b[0], b[3]]
        draw.line([*p0, *p1, *p2, *p3, *p0], fill='red', width=2)
    return np.asarray(image)

# Draw bounding boxes
def draw_boxes(image, bounds, color='yellow', width=2):
    draw = ImageDraw.Draw(image)
    for b in bounds:
        p0, p1, p2, p3 = [b[0], b[2]], [b[1], b[2]], \
                         [b[1], b[3]], [b[0], b[3]]
        draw.line([*p0, *p1, *p2, *p3, *p0], fill='red', width=2)
    return image


def appartient(elt, lst):
    for e in lst:
        if ((e != elt) & (e not in liste)) :
            print("soupçon de fraude")
        else:
            print("Absence de fraude")



tr = []
for txt in recognition_results:
    tr.append((txt[1]))

    
for i in file:
    
    text_coordinates = detect_text_blocks(i)
    recognition_results = reader.recognize(i,
                                 horizontal_list=text_coordinates,
                                 free_list=[]
                                 )
    tr=[]
        for txt in recognition_results:
            tr.append((txt[1]))
        if appartient(["Part Ro","Ro","Régime obligatoire"], tr) == "soupçon de fraude":
            shutil.move(str(i), file_destination)
        else:
            print("continue")