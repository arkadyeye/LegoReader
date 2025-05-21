import os
import fitz  # PyMuPDF
from dataclasses import dataclass
from typing import Tuple, Optional, Literal

import numpy as np
import cv2

@dataclass
class ContentBlock:
    pos: Tuple[int, int, int, int]                      # (x1, y1, x2, y2)
    type: Literal["image", "text"]                      # 'image' or 'text'
    image_extension: Optional[str] = None               # e.g., 'png', 'jpg'
    xref: Optional[str] = None
    text: Optional[str] = None                          # extracted or associated text
    font: Optional[int] = None
    image_data: Optional[bytes] = None  # binary image data

page_images = []
page_text = []

results_to_draw = []

def inspect_pdf(pdf_path,page_number):
    pdf_document = fitz.open(pdf_path)
    current_page_pdf = pdf_document[page_number]

    print(f"Size: {current_page_pdf.rect.width} x {current_page_pdf.rect.height}")
    print(f"Rotation: {current_page_pdf.rotation}")

    # get images list

    # --- Images ---
    print("\nImages:")
    images = current_page_pdf.get_images(full=True)
    if images:
        for i, img in enumerate(images):
            xref = img[0]
            base_image = pdf_document.extract_image(xref)
            rect = current_page_pdf.get_image_bbox(img)

            # drop lego text on the bottom corner
            # !!!!!!!!!!1 it works only for right side lego sign
            if rect.x1 >= current_page_pdf.rect.width or rect.y1 >= current_page_pdf.rect.height:
                continue

            image_block = ContentBlock(
                pos=(int(rect.x0),int(rect.y0),int(rect.x1),int(rect.y1)),
                type="image",
                xref = xref,
                image_data= base_image["image"],
                image_extension=base_image["ext"]
            )
            page_images.append(image_block)


    # get texts with ___x
    print("\nText:")
    text_dict = current_page_pdf.get_text("dict")
    for block in text_dict["blocks"]:
        for line in block.get("lines", []):
            for span in line["spans"]:
                text = span["text"].strip()
                x0, y0, x1, y1 = [int(v) for v in span["bbox"]]
                font_size = span['size']

                if len(text) < 5 and 'x' in text and font_size == 6.0:
                    text_block = ContentBlock(
                        pos = (x0,y0,x1,y1),
                        type="text",
                        text= span["text"].strip(),
                        font = span['size']
                    )
                    page_text.append(text_block)



    folder = f"parts/images"
    os.makedirs(folder, exist_ok=True)

    file = open('parts/parts.txt', 'a')

    # scan for large numbers (part numbers)
    #for each number get it's ___x and image
    text_dict = current_page_pdf.get_text("dict")
    for block in text_dict["blocks"]:
        for line in block.get("lines", []):
            for span in line["spans"]:
                text = span["text"].strip()

                if text.isdigit():
                    number = int(text)
                    if number > 1000 : #means it's lego part number
                        tx0, ty0, tx1, ty1 = [int(v) for v in span["bbox"]]

                        # look for parts amount
                        amount_found = False
                        for each in page_text:
                            x0,y0,x1,y1 = each.pos
                            if x0 < tx0+2 < x1 and y0 < ty0-5 < y1:
                                amount = each.text
                                amount_found = True
                                break

                        if not amount_found:
                            print (f"part number {number} without amount !!!!!!!!")
                            continue

                        # look for it's image
                        image_counter = -1
                        for each in page_images:
                            x0,y0,x1,y1 = each.pos
                            if x0 < tx0+5 < x1 and y0 < ty0-12 < y1:
                               results_to_draw.append((x0,y0,max(tx1,x1),ty1))

                               image_counter += 1
                               image_name = f"img_{number}_{image_counter}.{each.image_extension}"
                               image_path = os.path.join(folder, image_name)

                               with open(image_path, "wb") as img_file:
                                   img_file.write(each.image_data)

                        if image_counter == -1:
                            print (f"part number {number} without image !!!!!!!!")

                        amount = amount[:-1]
                        print (f"part number {number} , amount {amount}")

                        file.write(f"{number},{amount}\n")

    file.close()

    mat = fitz.Matrix(1, 1)
    pix = current_page_pdf.get_pixmap(matrix=mat)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n).copy()

    img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)

    for res in results_to_draw:
        x0, y0, x1, y1 = res
        cv2.rectangle(img, (x0, y0), (x1, y1), (255, 0, 255), 2)

    cv2.imshow("debug",img)
    key = cv2.waitKey(0) & 0xFF
    cv2.destroyAllWindows()




# Example usage
# inspect_pdf("Manuals/6186243.pdf",38)
# inspect_pdf("Manuals/6420974.pdf",157) #157,158
# inspect_pdf("Manuals/6208467.pdf",393) #393,394
inspect_pdf("Manuals/6566113.pdf",169) #169,170






