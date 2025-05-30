import os
import fitz  # PyMuPDF
from dataclasses import dataclass
from typing import Tuple, Optional, Literal

import numpy as np
import cv2

import pandas as pd

@dataclass
class ContentBlock:
    pos: Tuple[int, int, int, int]                      # (x1, y1, x2, y2)
    type: Literal["image","part_id","amount"]           # 'image' or 'part_id' or 'amount'
    image_extension: Optional[str] = None               # e.g., 'png', 'jpg'
    xref: Optional[str] = None
    text: Optional[str] = None                          # extracted or associated text
    font: Optional[int] = None
    image_data: Optional[bytes] = None  # binary image data

@dataclass
class FinalPart:
    image_data: Optional[bytes] = None  # binary image data
    element_id: Optional[str] = None  # extracted or associated text
    part_num: Optional[str] = None
    color_id: Optional[str] = None



class PartsProcessor:
    def __init__(self,debug = False):
        self.page_images = []
        self.element_ids = []
        self.amount_texts = []
        self.parts_list_font_size = 6.0
        self.debug = debug

        # element_id,part_num,color_id,design_id
        self.df = pd.read_csv('elements.csv')
        self.df['element_id'] = self.df['element_id'].astype(str)

    def __get_pdf_texts(self,pdf_page):

        amount_texts = []
        parts_ids = []
        text_dict = pdf_page.get_text("dict")
        for block in text_dict["blocks"]:
            for line in block.get("lines", []):
                for span in line["spans"]:
                    text = span["text"].strip()
                    x0, y0, x1, y1 = [int(v) for v in span["bbox"]]
                    font_size = span['size']

                    # here we are getting all the amounts of the details
                    if len(text) < 5 and 'x' in text and font_size == self.parts_list_font_size:
                        text_block = ContentBlock(
                            pos=(x0, y0, x1, y1),
                            type="amount",
                            text=span["text"].strip(),
                            font=span['size']
                        )
                        amount_texts.append(text_block)

                    # here we want to get all parts number, that are a number bigger than 1000
                    elif text.isdigit():
                        number = int(text)
                        if number > 1000:
                            id_block = ContentBlock(
                                pos=(x0, y0, x1, y1),
                                type="part_id",
                                text=span["text"].strip(),
                                font=span['size']
                            )
                            parts_ids.append(id_block)

        return parts_ids,amount_texts

    def __get_pdf_images(self,pdf_doc,pdf_page):

        page_images = []

        images = pdf_page.get_images(full=True)
        if images:
            for i, img in enumerate(images):
                xref = img[0]
                base_image = pdf_doc.extract_image(xref)
                rect = pdf_page.get_image_bbox(img)

                # drop lego text on the bottom corner
                # !!!!!!!!!!1 it works only for right side lego sign
                if rect.x1 >= pdf_page.rect.width or rect.y1 >= pdf_page.rect.height:
                    continue

                image_block = ContentBlock(
                    pos=(int(rect.x0), int(rect.y0), int(rect.x1), int(rect.y1)),
                    type="image",
                    xref=xref,
                    image_data=base_image["image"],
                    image_extension=base_image["ext"]
                )
                page_images.append(image_block)

        return page_images

    def load_parts_list(self,top_folder):
        '''
        here we need to reasamble parts list, from files on disk.
        this is not efficient at all, but it allows manual editing of the list
        :return: list of objects of parts (image_data,element_id,amount,part_num,color)
        '''

        # reconstruct file paths
        fill_path = os.path.join(top_folder, "parts")
        fill_path_images = os.path.join(fill_path, "images")
        parts_path = os.path.join(fill_path, "parts.txt")

        try:
            df = pd.read_csv(parts_path)
        except FileNotFoundError:
            print("Warning: The parts CSV file was not found.")
            return None  # or handle differently depending on your needs


        # inner function !! cool ?
        def load_image(row):


            element_id = row['element_id']
            ext = row['ext']

            # Check if ext is missing (NaN or empty string)
            if pd.isna(ext) or str(ext).strip() == '':
                return None

            filename = f'img_{element_id}_0.{ext}'
            path = os.path.join(fill_path_images, filename)  # Update with actual image directory

            image = cv2.imread(path)
            return image  # None if file doesn't exist or failed to load

        # Add a new column with the image data
        df['image'] = df.apply(load_image, axis=1)

        print("Parts loaded from disk")

        return df


    def extract_parts(self,top_folder,pdf_doc,page_number):

        parts_ram_list = []

        current_page_pdf = pdf_doc[page_number]

        # does it really used ?
        # print(f"Size: {current_page_pdf.rect.width} x {current_page_pdf.rect.height}")
        # print(f"Rotation: {current_page_pdf.rotation}")

        # get texts and images from parts list
        self.element_ids,self.amount_texts = self.__get_pdf_texts(current_page_pdf)
        self.page_images = self.__get_pdf_images(pdf_doc,current_page_pdf)

        #prepare output data file
        # folder = f"parts/images"

        fill_path = os.path.join(top_folder, "parts")
        fill_path_images = os.path.join(fill_path, "images")

        os.makedirs(fill_path_images, exist_ok=True)
        parts_path = os.path.join(fill_path, "parts.txt")

        file = open(parts_path, 'a') # append, because they may be more than 1 parts page
        file.write("element_id,ext,amount,part_num,color_id\n")

        #this list used for debug
        results_to_draw = []


        # extracting each part
        for element_id in self.element_ids:
            tx0, ty0, tx1, ty1 = element_id.pos

            # look for parts amount
            amount_found = False
            for each in self.amount_texts:
                x0, y0, x1, y1 = each.pos
                if x0 < tx0 + 2 < x1 and y0 < ty0 - 5 < y1:
                    amount = each.text
                    amount_found = True
                    break

            if not amount_found:
                print(f"element id {element_id.text} without amount !!!!!!!!")
                continue

            # look for it's image
            image_counter = -1
            image_data = None
            image_extension = ""
            for each in self.page_images:
                x0, y0, x1, y1 = each.pos
                if x0 < tx0 + 5 < x1 and y0 < ty0 - 12 < y1:
                    results_to_draw.append((x0, y0, max(tx1, x1), ty1))

                    image_counter += 1
                    image_name = f"img_{element_id.text}_{image_counter}.{each.image_extension}"
                    image_path = os.path.join(fill_path_images, image_name)
                    image_data = each.image_data
                    image_extension = each.image_extension
                    with open(image_path, "wb") as img_file:
                        img_file.write(each.image_data)

            if image_counter == -1:
                print(f"element id {element_id.text} without image !!!!!!!!")

            amount = amount[:-1]
            if self.debug:
                print(f"element id {element_id.text} , amount {amount}")

            # look for lego part id and color
            filtered_df = self.df[self.df['element_id'] == element_id.text]
            # Check if any match is found
            if not filtered_df.empty:
                part_num = filtered_df.iloc[0]['part_num']
                color_id = filtered_df.iloc[0]['color_id']
                # print("Part Number:", part_num)
                # print("Color ID:", color_id)
            else:
                print("No matching element_id found.")

            part = FinalPart(
                image_data=image_data,
                element_id=element_id.text,
                part_num = part_num,
                color_id = color_id
            )

            parts_ram_list.append(part)

            file.write(f"{element_id.text},{image_extension},{amount},{part_num},{color_id}\n")

        file.close()
        print ("parts extracted successfully")


        if self.debug:
            mat = fitz.Matrix(1, 1)
            pix = current_page_pdf.get_pixmap(matrix=mat)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n).copy()

            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)

            for res in results_to_draw:
                x0, y0, x1, y1 = res
                cv2.rectangle(img, (x0, y0), (x1, y1), (255, 0, 255), 2)

            cv2.imshow("parts_debug", img)
            key = cv2.waitKey(0) & 0xFF
            cv2.destroyWindow('parts_debug')

        return parts_ram_list




# Example usage
# inspect_pdf("Manuals/6186243.pdf",38)
# inspect_pdf("Manuals/6420974.pdf",157) #157,158
# inspect_pdf("Manuals/6208467.pdf",393) #393,394
# inspect_pdf("Manuals/6566113.pdf",169) #169,170

# pp = PartsProcessor()
# pp.load_parts_list("processed/6186243")



